// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Per-pixel backward rasterizer: per gaussian, the tile reduces its per-pixel eval gradients and
// thread 0 runs the projection backward once on the sum.

#include "../../common/types.slang"
#include "../../common/math.slang"
#include "../../primitives/gaussian/project.slang"
#include "../../primitives/gaussian/eval.slang"

#include "common/execution.slang"

static const float ALPHA_MAX = 0.99;
static const float ALPHA_MIN_TRACK = 1.0 / 255.0;

static const uint CH_MEAN2D_X  = 0;
static const uint CH_MEAN2D_Y  = 1;
static const uint CH_COV2D_0   = 2;
static const uint CH_COV2D_1   = 3;
static const uint CH_COV2D_2   = 4;
static const uint CH_OPACITY   = 5;
static const uint CH_COLOR_R   = 6;
static const uint CH_COLOR_G   = 7;
static const uint CH_COLOR_B   = 8;
static const uint CH_ITER_X    = 9;
static const uint CH_ITER_Y    = 10;
static const uint NUM_CHANNELS = 11;

// One partial sum per (channel, native wave). A 64-lane wave covers the whole tile.
static const uint MAX_WARPS = TILE_AREA / 32u;
groupshared float gs_warp[NUM_CHANNELS * MAX_WARPS];

// Final atomic operands, one InterlockedAdd per thread 0..15 in the slot order used below.
static const uint NUM_ATOMICS = 16u;
groupshared float gs_atomic[NUM_ATOMICS];

groupshared uint gs_max_processed;
groupshared Gaussian3D gs_gauss;
groupshared Projected2DGaussian gs_proj;
groupshared uint gs_g_idx;

[shader("compute")]
[numthreads(TILE_WIDTH, TILE_HEIGHT, 1)]
void backwardRasterize(
    uint3 dtid: SV_DispatchThreadID,
    uint3 gid: SV_GroupID,
    uint3 gtid: SV_GroupThreadID,
    StructuredBuffer<Gaussian3D> gaussians,
    StructuredBuffer<Projected2DGaussian> projected_2d,
    StructuredBuffer<uint> tile_start,
    StructuredBuffer<uint> tile_end,
    StructuredBuffer<uint> sorted_gaussian_ids,
    StructuredBuffer<CameraData> camera,
    RWTexture2D<float4> output_image,
    RWTexture2D<uint> gaussian_count_per_pixel,
    RWTexture2D<float4> dL_dC_texture,
    AtomicI64Buffer grad_mean_x,
    AtomicI64Buffer grad_mean_y,
    AtomicI64Buffer grad_mean_z,
    AtomicI64Buffer grad_log_scale_x,
    AtomicI64Buffer grad_log_scale_y,
    AtomicI64Buffer grad_log_scale_z,
    AtomicI64Buffer grad_rot_w,
    AtomicI64Buffer grad_rot_x,
    AtomicI64Buffer grad_rot_y,
    AtomicI64Buffer grad_rot_z,
    AtomicI64Buffer grad_opacity,
    AtomicI64Buffer grad_color_r,
    AtomicI64Buffer grad_color_g,
    AtomicI64Buffer grad_color_b,
    AtomicI64Buffer grad_2d_x_iter,
    AtomicI64Buffer grad_2d_y_iter)
{
    uint2 pixel = dtid.xy;
    uint tid = gtid.y * TILE_WIDTH + gtid.x;
    uint2 frame_dim_px = uint2(camera[0].image_size);

    bool valid_pixel = (pixel.x < frame_dim_px.x && pixel.y < frame_dim_px.y);

    uint2 num_tiles = (frame_dim_px + uint2(TILE_WIDTH, TILE_HEIGHT) - 1u)
                      / uint2(TILE_WIDTH, TILE_HEIGHT);
    uint tile_idx = gid.y * num_tiles.x + gid.x;
    uint t_start = tile_start[tile_idx];

    uint processed = valid_pixel ? gaussian_count_per_pixel[pixel] : 0;

    if (tid == 0) gs_max_processed = 0;
    GroupMemoryBarrierWithGroupSync();
    InterlockedMax(gs_max_processed, processed);
    GroupMemoryBarrierWithGroupSync();

    uint max_processed = gs_max_processed;
    if (max_processed == 0) return;

    float2 pixel_center = float2(pixel) + 0.5;
    float4 dL_dC_full = valid_pixel ? dL_dC_texture[pixel] : float4(0.0);
    float3 dL_dC = dL_dC_full.rgb * camera[0].lensGain(pixel);
    float dL_dA = dL_dC_full.a;

    // Final transmittance T = 1 - alpha (forward wrote output_image.a = 1-T).
    float T_post = valid_pixel ? (1.0 - output_image[pixel].a) : 1.0;
    float T_final = T_post;
    float3 C_behind = float3(0.0, 0.0, 0.0);

    float4x4 world_to_view = camera[0].view_matrix;
    float4x4 view_to_clip  = camera[0].proj_matrix;
    float2 frame_dim = camera[0].image_size;

    for (int i = int(max_processed) - 1; i >= 0; --i)
    {
        if (tid == 0)
        {
            uint idx = sorted_gaussian_ids[t_start + uint(i)];
            gs_g_idx = idx;
            gs_gauss = gaussians[idx];
            gs_proj = projected_2d[idx];
        }
        GroupMemoryBarrierWithGroupSync();

        uint g_idx = gs_g_idx;
        Gaussian3D g = gs_gauss;
        Projected2DGaussian proj = gs_proj;

        bool contributes = valid_pixel && (uint(i) < processed);

        float2 local_dp_mean_2d = float2(0.0);
        float3 local_dp_cov2d = float3(0.0);
        float local_dp_opacity = 0.0;
        float3 local_dp_color = float3(0.0);
        float local_abs_mean2d_x = 0.0;
        float local_abs_mean2d_y = 0.0;

        if (contributes)
        {
            float alpha_raw = evalGaussian2DFromCov(pixel_center, proj.mean, proj.cov2d, g.opacity, proj.opacity_compensation);
            float alpha = min(alpha_raw, ALPHA_MAX);
            float alpha_grad_scale = (alpha_raw <= ALPHA_MAX) ? 1.0 : 0.0;

            // Must match the forward's alpha-min skip: a skipped splat is a no-op (denom = 1, zero gradient).
            if (alpha < ALPHA_MIN_TRACK)
            {
                alpha = 0.0;
                alpha_grad_scale = 0.0;
            }

            float denom = 1.0 - alpha;
            float T_i = T_post / denom;

            var dp_sh0 = diffPair(g.sh0);
            float3 c_i = shDC(dp_sh0.p);

            float3 dL_dc_i = dL_dC * (alpha * T_i);
            bwd_diff(shDC)(dp_sh0, dL_dc_i);

            float3 dC_dalpha = c_i * T_i - C_behind / denom;

            float dL_dalpha = dot(dL_dC, dC_dalpha) + dL_dA * (T_final / denom);
            
            float dL_d_alpha_raw = dL_dalpha * alpha_grad_scale;

            var dp_mean_2d = diffPair(proj.mean);
            var dp_cov2d   = diffPair(proj.cov2d);
            var dp_opacity = diffPair(g.opacity);
            var dp_comp    = diffPair(proj.opacity_compensation);
            bwd_diff(evalGaussian2DFromCov)(
                pixel_center, dp_mean_2d, dp_cov2d, dp_opacity, dp_comp, dL_d_alpha_raw);

            local_dp_mean_2d = dp_mean_2d.d;
            local_dp_cov2d = dp_cov2d.d;
            local_dp_opacity = dp_opacity.d;
            local_dp_color = dp_sh0.d;

            if (alpha >= ALPHA_MIN_TRACK)
            {
                local_abs_mean2d_x = abs(dp_mean_2d.d.x);
                local_abs_mean2d_y = abs(dp_mean_2d.d.y);
            }

            C_behind += c_i * alpha * T_i;
            T_post = T_i;
        }

        uint lane = WaveGetLaneIndex();
        uint warp_idx = tid / WaveGetLaneCount();

        float w_mean2d_x = WaveActiveSum(local_dp_mean_2d.x);
        float w_mean2d_y = WaveActiveSum(local_dp_mean_2d.y);
        float w_cov2d_0  = WaveActiveSum(local_dp_cov2d.x);
        float w_cov2d_1  = WaveActiveSum(local_dp_cov2d.y);
        float w_cov2d_2  = WaveActiveSum(local_dp_cov2d.z);
        float w_opacity  = WaveActiveSum(local_dp_opacity);
        float w_color_r  = WaveActiveSum(local_dp_color.x);
        float w_color_g  = WaveActiveSum(local_dp_color.y);
        float w_color_b  = WaveActiveSum(local_dp_color.z);
        float w_iter_x   = WaveActiveSum(local_abs_mean2d_x);
        float w_iter_y   = WaveActiveSum(local_abs_mean2d_y);

        if (lane == 0u)
        {
            gs_warp[CH_MEAN2D_X * MAX_WARPS + warp_idx] = w_mean2d_x;
            gs_warp[CH_MEAN2D_Y * MAX_WARPS + warp_idx] = w_mean2d_y;
            gs_warp[CH_COV2D_0  * MAX_WARPS + warp_idx] = w_cov2d_0;
            gs_warp[CH_COV2D_1  * MAX_WARPS + warp_idx] = w_cov2d_1;
            gs_warp[CH_COV2D_2  * MAX_WARPS + warp_idx] = w_cov2d_2;
            gs_warp[CH_OPACITY  * MAX_WARPS + warp_idx] = w_opacity;
            gs_warp[CH_COLOR_R  * MAX_WARPS + warp_idx] = w_color_r;
            gs_warp[CH_COLOR_G  * MAX_WARPS + warp_idx] = w_color_g;
            gs_warp[CH_COLOR_B  * MAX_WARPS + warp_idx] = w_color_b;
            gs_warp[CH_ITER_X   * MAX_WARPS + warp_idx] = w_iter_x;
            gs_warp[CH_ITER_Y   * MAX_WARPS + warp_idx] = w_iter_y;
        }
        GroupMemoryBarrierWithGroupSync();

        // Cross-warp reduction: thread c sums channel c's warp partials.
        float total = 0.0;
        if (tid < NUM_CHANNELS)
        {
            for (uint w = 0u; w < (TILE_AREA + WaveGetLaneCount() - 1u) / WaveGetLaneCount(); ++w)
                total += gs_warp[tid * MAX_WARPS + w];
        }
        // Slot mapping matches the InterlockedAdd order below.
        if (tid == CH_OPACITY) gs_atomic[0] = total;
        if (tid == CH_COLOR_R) gs_atomic[1] = total;
        if (tid == CH_COLOR_G) gs_atomic[2] = total;
        if (tid == CH_COLOR_B) gs_atomic[3] = total;
        if (tid == CH_ITER_X)  gs_atomic[4] = total;
        if (tid == CH_ITER_Y)  gs_atomic[5] = total;
        // Projection-input totals reuse gs_warp[0..4]; the per-warp partial writes are already complete.
        if (tid == CH_MEAN2D_X) gs_warp[0] = total;
        if (tid == CH_MEAN2D_Y) gs_warp[1] = total;
        if (tid == CH_COV2D_0)  gs_warp[2] = total;
        if (tid == CH_COV2D_1)  gs_warp[3] = total;
        if (tid == CH_COV2D_2)  gs_warp[4] = total;
        GroupMemoryBarrierWithGroupSync();

        // The projection backward is small serial compute that doesn't parallelize, so thread 0 runs it.
        if (tid == 0)
        {
            float2 sum_dp_mean_2d = float2(gs_warp[0], gs_warp[1]);
            float3 sum_dp_cov2d   = float3(gs_warp[2], gs_warp[3], gs_warp[4]);

            var dp_mean      = diffPair(g.mean);
            var dp_log_scale = diffPair(g.log_scale);
            var dp_rot_quat  = diffPair(g.rot_quat);
            // Vanilla projection only (filter_3D = 0, kernel_size = 0.3): Mip-Splatting needs the per-splat backward.
            bwd_diff(projectGaussian3DTo2D)(
                dp_mean, dp_log_scale, dp_rot_quat,
                world_to_view, view_to_clip, frame_dim,
                0.0, 0.3, 0u,
                sum_dp_mean_2d, sum_dp_cov2d, 0.0, 0.0, 0.0);

            gs_atomic[6]  = dp_mean.d.x;
            gs_atomic[7]  = dp_mean.d.y;
            gs_atomic[8]  = dp_mean.d.z;
            gs_atomic[9]  = dp_log_scale.d.x;
            gs_atomic[10] = dp_log_scale.d.y;
            gs_atomic[11] = dp_log_scale.d.z;
            gs_atomic[12] = dp_rot_quat.d.x;
            gs_atomic[13] = dp_rot_quat.d.y;
            gs_atomic[14] = dp_rot_quat.d.z;
            gs_atomic[15] = dp_rot_quat.d.w;
        }
        GroupMemoryBarrierWithGroupSync();

        if (tid < NUM_ATOMICS)
        {
            float v = gs_atomic[tid];
            int64_t fx = toFixed(v);
            if      (tid == 0)  atomicAddI64(grad_opacity,     g_idx, fx);
            else if (tid == 1)  atomicAddI64(grad_color_r,     g_idx, fx);
            else if (tid == 2)  atomicAddI64(grad_color_g,     g_idx, fx);
            else if (tid == 3)  atomicAddI64(grad_color_b,     g_idx, fx);
            else if (tid == 4)  atomicAddI64(grad_2d_x_iter,   g_idx, fx);
            else if (tid == 5)  atomicAddI64(grad_2d_y_iter,   g_idx, fx);
            else if (tid == 6)  atomicAddI64(grad_mean_x,      g_idx, fx);
            else if (tid == 7)  atomicAddI64(grad_mean_y,      g_idx, fx);
            else if (tid == 8)  atomicAddI64(grad_mean_z,      g_idx, fx);
            else if (tid == 9)  atomicAddI64(grad_log_scale_x, g_idx, fx);
            else if (tid == 10) atomicAddI64(grad_log_scale_y, g_idx, fx);
            else if (tid == 11) atomicAddI64(grad_log_scale_z, g_idx, fx);
            else if (tid == 12) atomicAddI64(grad_rot_w,       g_idx, fx);
            else if (tid == 13) atomicAddI64(grad_rot_x,       g_idx, fx);
            else if (tid == 14) atomicAddI64(grad_rot_y,       g_idx, fx);
            else if (tid == 15) atomicAddI64(grad_rot_z,       g_idx, fx);
        }

        GroupMemoryBarrierWithGroupSync();
    }
}
