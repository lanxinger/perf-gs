// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Transposed (per-splat) backward: each warp owns a bucket and each lane a splat, accumulating its
// gradient over the tile's pixels; T_i is rebuilt from forward checkpoints by wave prefix scans.

#include "../../common/types.slang"
#include "../../common/math.slang"
#include "../../primitives/gaussian/project.slang"
#include "../../primitives/gaussian/eval.slang"
#include "../../primitives/gaussian/blend.slang"
#include "../../core/compositor.slang"
#include "../../common/stbn.slang"
#include "checkpoint.slang"

#include "common/execution.slang"
// Pixel-in-tile index is row-major with row stride TILE_WIDTH (tiles may be non-square).
static const uint NUM_WARPS = TILE_AREA / RASTER_BUCKET_SIZE;  // 32-splat buckets per group

// A native wave may hold two independent buckets. Shuffle sources are native lane indices.
uint bucketWaveLane(uint wave_lane, uint bucket_lane)
{
    return (wave_lane & ~(RASTER_BUCKET_SIZE - 1u)) + bucket_lane;
}

static const float ALPHA_MAX = 0.99;
static const float ALPHA_MIN_TRACK = 1.0 / 255.0;

// Equivalent gradients:
//   0 = fused affine-map scan (blendPrefixExclusive, fast on Vulkan/D3D12 SHFL),
//   1 = separate wave scans for T and rgb C_in_front,
//   2 = hardware WavePrefixProduct(T) + scalar WavePrefixSum(dot(dLdC, c) * alpha * T),
//       fastest on Apple Silicon Metal (2 hardware SIMD prefix ops vs 20 cross-lane shuffles).
#ifndef BACKWARD_BLEND_SCAN_MODE
#ifdef TRAINER_METAL_BACKEND
#define BACKWARD_BLEND_SCAN_MODE 2
#else
#define BACKWARD_BLEND_SCAN_MODE 0
#endif
#endif

#if BACKWARD_BLEND_SCAN_MODE != 0
// On a 64-lane wave make the lower bucket contribute the identity to the upper bucket's prefix.
float bucketPrefixProduct(float x, uint wave_lane)
{
    float prefix = WavePrefixProduct(x);
    if (trainerWaveLaneCount() > RASTER_BUCKET_SIZE)
    {
        bool upper = wave_lane >= RASTER_BUCKET_SIZE;
        float upper_prefix = WavePrefixProduct(upper ? x : 1.0);
        return upper ? upper_prefix : prefix;
    }
    return prefix;
}

float3 bucketPrefixSum(float3 x, uint wave_lane)
{
    float3 prefix = WavePrefixSum(x);
    if (trainerWaveLaneCount() > RASTER_BUCKET_SIZE)
    {
        bool upper = wave_lane >= RASTER_BUCKET_SIZE;
        float3 upper_prefix = WavePrefixSum(upper ? x : float3(0.0));
        return upper ? upper_prefix : prefix;
    }
    return prefix;
}

float bucketPrefixSumScalar(float x, uint wave_lane)
{
    float prefix = WavePrefixSum(x);
    if (trainerWaveLaneCount() > RASTER_BUCKET_SIZE)
    {
        bool upper = wave_lane >= RASTER_BUCKET_SIZE;
        float upper_prefix = WavePrefixSum(upper ? x : 0.0);
        return upper ? upper_prefix : prefix;
    }
    return prefix;
}
#endif

// Exclusive bucket scan of the affine maps T -> T * T_step, S -> S + dLdC_weight * T, where
// (a1, b1) then (a2, b2) composes to (a1*a2, b1 + a1*b2): one Hillis-Steele pass yields both prefixes.
// Each of the N components scans one pixel; their shuffle chains are independent and interleave.
void blendPrefixExclusive<let N : int>(vector<float, N> T_step, vector<float, N> dLdC_weight,
                                       uint wave_lane, uint lane,
                                       out vector<float, N> T_prefix,
                                       out vector<float, N> dLdC_C_in_front_prefix)
{
    // Inclusive scan: lane i holds the composed map over splats [0..i].
    vector<float, N> a = T_step;
    vector<float, N> b = dLdC_weight;
    [unroll]
    for (uint d = 1u; d < RASTER_BUCKET_SIZE; d <<= 1u)
    {
        // Clamped source is unused when lane < d; all 32 lanes are active (warp-uniform region).
        uint src = bucketWaveLane(wave_lane, (lane >= d) ? (lane - d) : 0u);
        vector<float, N> a_lo = WaveReadLaneAt(a, src);
        vector<float, N> b_lo = WaveReadLaneAt(b, src);
        if (lane >= d)
        {
            b = b_lo + a_lo * b;   // b1 + a1*b2  (before overwriting a)
            a = a_lo * a;          // a1 * a2
        }
    }
#if COMPOSITOR_HARDNESS_ONE
    // Exclusive = inclusive with this lane's own step taken out, saving a dependent shuffle round:
    // T_step = 1 - alpha >= 1 - ALPHA_MAX, so the division is well conditioned.
    T_prefix = a / T_step;
    dLdC_C_in_front_prefix = b - T_prefix * dLdC_weight;
#else
    // Hardened extinction can drive T_step toward 0, so shift by a lane instead of dividing.
    uint prev = bucketWaveLane(wave_lane, (lane >= 1u) ? (lane - 1u) : 0u);
    vector<float, N> a_prev = WaveReadLaneAt(a, prev);
    vector<float, N> b_prev = WaveReadLaneAt(b, prev);
    T_prefix        = (lane >= 1u) ? a_prev : vector<float, N>(1.0);
    dLdC_C_in_front_prefix = (lane >= 1u) ? b_prev : vector<float, N>(0.0);
#endif
}

void blendPrefixExclusive(float T_step, float dLdC_weight, uint wave_lane, uint lane,
                          out float T_prefix, out float dLdC_C_in_front_prefix)
{
    vector<float, 1> t, s;
    blendPrefixExclusive<1>(vector<float, 1>(T_step), vector<float, 1>(dLdC_weight), wave_lane, lane, t, s);
    T_prefix = t.x;
    dLdC_C_in_front_prefix = s.x;
}

// Stochastic pixel subsampling: visit 1/STRIDE of the pixels and scale grads by STRIDE; the host
// rotates pixel_phase so the estimate stays unbiased. STRIDE 1 is the exact backward.
#ifndef BACKWARD_PIXEL_STRIDE
#define BACKWARD_PIXEL_STRIDE 1u
#endif

// Mirrors the host check (config_types.validate_training_config) against a stale define pair.
#if STBN_ENABLED && (BACKWARD_PIXEL_STRIDE > 1u)
#error "STBN_ENABLED and BACKWARD_PIXEL_STRIDE > 1 both subsample the backward's pixels; enable only one."
#endif

// STBN walks a compacted list of the k sampled pixels (trip count k); otherwise all TILE_AREA pixels.
#if STBN_ENABLED
static const uint STBN_VISIT_CAPACITY = STBN_MAX_TILE_PIXELS;
// Visit slot -> pixel-in-tile, per warp: flat-backward warps can own different tiles (per-tile kernel uses row 0).
groupshared uint gs_stbn_visit[NUM_WARPS][STBN_VISIT_CAPACITY];
#endif

// Per-pixel constants staged in LDS once per tile rather than re-read every bucket.
groupshared float4 gs_dLdC[TILE_AREA];   // rgb = dL_dC, a = dL_dA
groupshared float  gs_Tfinal[TILE_AREA];
groupshared float3 gs_Ctotal[TILE_AREA];
groupshared uint   gs_processed[TILE_AREA];
groupshared uint   gs_max_processed;     // deepest splat any pixel processed

// Per-warp LDS stage of the bucket's checkpoints for visited pixels only: slot k = (p - phase) / STRIDE.
#if STBN_ENABLED
static const uint STAGED_PIXELS = STBN_VISIT_CAPACITY;
#else
static const uint STAGED_PIXELS = TILE_AREA / BACKWARD_PIXEL_STRIDE;
#endif

groupshared uint2  gs_checkpoints[NUM_WARPS][STAGED_PIXELS]; // (T,RGB) as fp16

// Pixels the per-bucket backward advances together (independent scans that overlap their latency).
static const int PIXELS_PER_STEP = 2;
typealias PixelStepVector = vector<float, PIXELS_PER_STEP>;

[shader("compute")]
[numthreads(TILE_AREA, 1, 1)]
void backwardRasterizePerSplat(
    uint3 gid: SV_GroupID,
    uint3 gtid: SV_GroupThreadID,
    // Compact blend record packed by bin_count: the random (sorted-order) gather reads only what it uses.
    StructuredBuffer<BlendData> blend_data,
    StructuredBuffer<uint> tile_start,
    StructuredBuffer<uint> tile_end,
    StructuredBuffer<uint> sorted_gaussian_ids,
    StructuredBuffer<uint> bucket_offsets,
    StructuredBuffer<uint2> raster_checkpoints,
    StructuredBuffer<CameraData> camera,
    Texture2D<float4> output_image,
    Texture2D<uint> gaussian_count_per_pixel,
    Texture2D<float4> dL_dC_texture,
    // Tile-summed BlendData seeds for projection_backward.slang; the color grads below are final.
    AtomicGradBuffer grad_mean2d_x,
    AtomicGradBuffer grad_mean2d_y,
    AtomicGradBuffer grad_conic_a,
    AtomicGradBuffer grad_conic_b,
    AtomicGradBuffer grad_conic_c,
    AtomicGradBuffer grad_amplitude,
    AtomicGradBuffer grad_opacity,   // opaque_core's direct falloff term only
    AtomicGradBuffer grad_color_r,
    AtomicGradBuffer grad_color_g,
    AtomicGradBuffer grad_color_b,
    AtomicGradBuffer grad_2d_x_iter,
    AtomicGradBuffer grad_2d_y_iter,
    // Densify-window sum of each splat's blend weight alpha * T; written only when track_blend_weight != 0.
    AtomicI64Buffer blend_weight_accum,
    uniform uint track_blend_weight,
    // Screen-gradient tracking for densification (grad_2d_*_iter); off once densification ends.
    uniform uint track_screen_grad,
    // Visited pixel subset in [0, BACKWARD_PIXEL_STRIDE); the host rotates it per iteration.
    uniform uint pixel_phase,
#if STBN_ENABLED
    // stbn_slice is the image's visit count + phase (not the iteration), walking the mask's temporal axis.
    StructuredBuffer<uint> stbn_table,
    uniform uint2 stbn_dims,
    uniform uint stbn_slice,
    uniform uint stbn_tile_pixels,
#endif
    // Compositor extinction hardness k (1 = alpha-over). See core/compositor.slang.
    uniform float extinction_hardness)
{
    uint tid = gtid.x;
    uint2 frame_dim_px = uint2(camera[0].image_size);
    uint2 num_tiles = (frame_dim_px + uint2(TILE_WIDTH, TILE_HEIGHT) - 1u)
                      / uint2(TILE_WIDTH, TILE_HEIGHT);
    // 1D dispatch: gid.x is the linear tile index, unlike the per-pixel backward's 2D grid.
    uint tile_idx = gid.x;
    uint tile_x = tile_idx % num_tiles.x;
    uint tile_y = tile_idx / num_tiles.x;
    if (tile_y >= num_tiles.y) return;

    uint t_start = tile_start[tile_idx];
    uint t_end   = tile_end[tile_idx];
    uint tile_count = (t_end > t_start) ? (t_end - t_start) : 0u;
    if (tile_count == 0u) return;

    uint tile_bucket_base = bucket_offsets[tile_idx];

    float4x4 world_to_view = camera[0].view_matrix;
    float4x4 view_to_clip  = camera[0].proj_matrix;
    float2 frame_dim = camera[0].image_size;

    uint tile_origin_x = tile_x * TILE_WIDTH;
    uint tile_origin_y = tile_y * TILE_HEIGHT;
    uint2 my_pixel = uint2(tile_origin_x + (tid % TILE_WIDTH),
                           tile_origin_y + (tid / TILE_WIDTH));
    bool my_pixel_valid = (my_pixel.x < frame_dim_px.x && my_pixel.y < frame_dim_px.y);
#if STBN_ENABLED
    // Tile-uniform: the workgroup shares one mask block; slot j is the j-th sampled pixel.
    uint stbn_base = stbnBlockBase(uint2(tile_origin_x, tile_origin_y),
                                   stbn_dims, stbn_slice);
    // Mask on: the table's per-tile count (it varies under a global slice threshold), not the uniform.
    // Mask off: never the table (the loss weights every pixel); the list permutes the whole tile.
    uint visit_count = stbnActive(stbn_tile_pixels)
        ? min(stbnTileCount(stbn_table, stbn_base), STBN_VISIT_CAPACITY)
        : TILE_AREA;
    if (tid < visit_count)
        gs_stbn_visit[0][tid] = stbnPixelAtSlot(stbn_table, stbn_base, tid);
#else
    uint visit_count = TILE_AREA;
#endif
    if (my_pixel_valid)
    {
        float gain = camera[0].lensGain(my_pixel);
        gs_dLdC[tid]      = dL_dC_texture[my_pixel] * float4(gain.xxx, 1.0);
        // T_final = 1 - alpha (forward wrote output_image.a = 1-T).
        float4 out_px     = output_image[my_pixel];
        gs_Tfinal[tid]    = 1.0 - out_px.a;
        gs_Ctotal[tid]    = out_px.rgb / gain;
        gs_processed[tid] = gaussian_count_per_pixel[my_pixel];
    }
    else
    {
        gs_dLdC[tid]      = float4(0.0);
        gs_Tfinal[tid]    = 1.0;
        gs_Ctotal[tid]    = float3(0.0);
        gs_processed[tid] = 0u;   // no splat contributes to an OOB pixel
    }
    GroupMemoryBarrierWithGroupSync();

    // Saturation early-out: buckets past ceil(max_processed / P) have no contributing pixel.
    if (tid == 0u) gs_max_processed = 0u;
    GroupMemoryBarrierWithGroupSync();
#if STBN_ENABLED
    // Sampled pixels only: an unsampled deeper ray would keep zero-contribution buckets alive.
    if (stbnSelected(stbn_table, my_pixel, stbn_dims, stbn_slice, stbn_tile_pixels))
        InterlockedMax(gs_max_processed, gs_processed[tid]);
#else
    InterlockedMax(gs_max_processed, gs_processed[tid]);
#endif
    GroupMemoryBarrierWithGroupSync();
    uint max_processed = gs_max_processed;

    uint wave_lane = trainerWaveLaneIndex(tid);
    uint lane = wave_lane % RASTER_BUCKET_SIZE;
    uint warp_idx = tid / RASTER_BUCKET_SIZE;
    // Truncate to the buckets some pixel actually reached (vs. tile_count).
    uint num_buckets = (max_processed + RASTER_BUCKET_SIZE - 1u) / RASTER_BUCKET_SIZE;

    [loop]
    for (uint base_bucket = 0u; base_bucket < num_buckets; base_bucket += NUM_WARPS)
    {
        uint bucket = base_bucket + warp_idx;
        bool bucket_valid = (bucket < num_buckets);

        uint local_splat = bucket * RASTER_BUCKET_SIZE + lane;
        bool splat_valid = bucket_valid && (local_splat < tile_count);

        // Masked lanes hold junk but never write; they are scan identities via alpha = 0.
        uint g_idx = 0u;
        BlendData blend_only_data;
        if (splat_valid)
        {
            g_idx = sorted_gaussian_ids[t_start + local_splat];
            blend_only_data = blend_data[g_idx];
        }

        float2 acc_mean2d = float2(0.0);
        float3 acc_conic  = float3(0.0);
        float  acc_amplitude = 0.0;
        float  acc_opacity = 0.0;
        float3 acc_color  = float3(0.0);
        float  acc_absx = 0.0;
        float  acc_absy = 0.0;
        float  acc_weight = 0.0;

        // Same GaussianPrimitive as the forward; masked lanes get a finite zero prim.
        GaussianPrimitive prim = splat_valid ? GaussianPrimitive.make(blend_only_data)
                                             : GaussianPrimitive.none();
        float3 c_i = prim.color();

        uint cp_bucket = tile_bucket_base + bucket;
        uint bucket_first_splat = bucket * RASTER_BUCKET_SIZE;

        // Barriers are legal here (block-uniform trip count): WAR against the previous iteration's
        // reads, then RAW after staging because the pixel loop reads entries staged by other lanes.
        GroupMemoryBarrierWithGroupSync();
        if (bucket_valid)
        {
            uint cp_base = checkpointIndex(cp_bucket, 0u);
#if STBN_ENABLED
            // Slot k holds the k-th sampled pixel's checkpoint, indexed by visit slot below.
            for (uint k = lane; k < visit_count; k += RASTER_BUCKET_SIZE)
                gs_checkpoints[warp_idx][k] =
                    raster_checkpoints[cp_base + gs_stbn_visit[0][k]];
#else
            uint phase = pixel_phase % BACKWARD_PIXEL_STRIDE;  // 0 at STRIDE=1
            // Slot k holds pixel p = phase + k * STRIDE, exactly the pixels the loop visits.
            for (uint k = lane; k < STAGED_PIXELS; k += RASTER_BUCKET_SIZE)
                gs_checkpoints[warp_idx][k] = raster_checkpoints[cp_base + phase + k * BACKWARD_PIXEL_STRIDE];
#endif
        }
        GroupMemoryBarrierWithGroupSync();

        // All 32 lanes iterate in lockstep for the wave scans; v is the visit slot (the pixel without STBN).
        [loop]
        for (uint v = 0u; v < visit_count; ++v)
        {
#if STBN_ENABLED
            // Warp-uniform (tile-uniform list); slot v is also the checkpoint slot.
            uint p = gs_stbn_visit[0][v];
            uint checkpoint_slot = v;
#else
            uint p = v;
#endif
            // Saturation skip (ray ended before this bucket); warp-uniform, so the wave scans stay in sync.
            uint processed = gs_processed[p];
            if (processed <= bucket_first_splat)
                continue;

#if !STBN_ENABLED
            // Stride subsampling; warp-uniform (p, pixel_phase), so the wave scans stay valid.
            if (BACKWARD_PIXEL_STRIDE > 1u &&
                (p % BACKWARD_PIXEL_STRIDE) != (pixel_phase % BACKWARD_PIXEL_STRIDE))
                continue;
#endif

            // Lanes past the pixel's early break get alpha = 0: a scan identity (T_step = 1, c*alpha*T = 0).
            bool contributes = splat_valid && (local_splat < processed);

#if !STBN_ENABLED
            // Visited pixels are compacted to slot (p - phase) / STRIDE.
            uint checkpoint_slot = (p - (pixel_phase % BACKWARD_PIXEL_STRIDE))
                                   / BACKWARD_PIXEL_STRIDE;
#endif
            float T_checkpoint; float3 C_checkpoint;
            unpackCheckpoint(gs_checkpoints[warp_idx][checkpoint_slot], T_checkpoint, C_checkpoint);

            float2 pixel_center = float2(tile_origin_x + (p % TILE_WIDTH),
                                         tile_origin_y + (p / TILE_WIDTH)) + 0.5;

            float alpha = 0.0;
            float alpha_grad_scale = 0.0;
            if (contributes)
            {
                float alpha_raw = prim.coverage(pixel_center);
                alpha = min(alpha_raw, ALPHA_MAX);
                alpha_grad_scale = (alpha_raw <= ALPHA_MAX) ? 1.0 : 0.0;
                // Must match the forward's alpha-min skip, or the T reconstruction diverges from it.
                if (alpha < ALPHA_MIN_TRACK)
                {
                    alpha = 0.0;
                    alpha_grad_scale = 0.0;
                }
            }

            // hardness 1 = alpha-over, k > 1 sharpens extinction (core/compositor.slang).
            Compositor comp;
            comp.hardness = extinction_hardness;
            float T_step = comp.transmittance(alpha);

            float3 dL_dC = gs_dLdC[p].rgb;

#if BACKWARD_BLEND_SCAN_MODE == 0
            // dL_dC is warp-uniform, so dot(dL_dC, .) commutes through the prefix sum: a scalar scan suffices.
            float dLdC_weight = dot(dL_dC, c_i) * comp.weight(alpha);   // 0 when masked
            float T_prefix, dLdC_C_in_front_prefix;
            blendPrefixExclusive(T_step, dLdC_weight, wave_lane, lane, T_prefix, dLdC_C_in_front_prefix);
            float T_i     = T_checkpoint * T_prefix;                   // pre-splat transmittance
            float alpha_T = comp.weight(alpha) * T_i;          // this splat's blend weight
            // dot(dL_dC, c_i * alpha * T_i): this splat's share of dot(dL_dC, C_in_front_incl).
            float dLdC_contrib = dLdC_weight * T_i;
            // dot(dL_dC, C_in_front_excl(i)) = dot(dL_dC, C_checkpoint) + prefixsum.
            float dLdC_C_in_front = dot(dL_dC, C_checkpoint) + T_checkpoint * dLdC_C_in_front_prefix;   // exclusive
#elif BACKWARD_BLEND_SCAN_MODE == 2
            float T_prefix = bucketPrefixProduct(T_step, wave_lane);
            float T_i     = T_checkpoint * T_prefix;
            float alpha_T = comp.weight(alpha) * T_i;          // this splat's blend weight
            float dLdC_contrib = dot(dL_dC, c_i) * alpha_T;
            float dLdC_C_in_front = dot(dL_dC, C_checkpoint) + bucketPrefixSumScalar(dLdC_contrib, wave_lane);
#else
            float T_prefix = bucketPrefixProduct(T_step, wave_lane);
            float T_i     = T_checkpoint * T_prefix;
            float alpha_T = comp.weight(alpha) * T_i;          // this splat's blend weight
            float3 contrib = c_i * alpha_T;                    // 0 when masked
            // C_in_front(i) = C_checkpoint + sum of contrib for lower lanes.
            float3 C_in_front = C_checkpoint + bucketPrefixSum(contrib, wave_lane);
#endif

            if (contributes)
            {
                float  dL_dA = gs_dLdC[p].a;
                float  T_final = gs_Tfinal[p];
                float3 C_total = gs_Ctotal[p];

                float oneMinusAlpha = 1.0 - alpha;

                // shDC is affine, so the color seed is summed here and scaled by SH_C0 after the loop.
                acc_color += dL_dC * alpha_T;
                acc_weight += alpha_T;

#if BACKWARD_BLEND_SCAN_MODE == 0 || BACKWARD_BLEND_SCAN_MODE == 2
                // dot(dL_dC, C_behind) with C_behind = C_total - C_in_front(i) - c_i * alpha_i * T_i.
                float dLdC_Cbehind = dot(dL_dC, C_total) - dLdC_C_in_front - dLdC_contrib;
                // The compositor owns the alpha-over vs hardened-extinction math.
                float dL_dalpha = comp.coverageGrad(alpha, dot(dL_dC, c_i), T_i,
                                                    dLdC_Cbehind, dL_dA, T_final);
#else
                // C_behind(i) = sum_{j>i, j<processed} c_j * alpha_j * T_j.
                float3 C_behind = C_total - C_in_front - contrib;
                // Same Compositor as the fused path, so both scan modes honor extinction_hardness.
                float dL_dalpha = comp.coverageGrad(alpha, dot(dL_dC, c_i), T_i,
                                                    dot(dL_dC, C_behind), dL_dA, T_final);
#endif
                float dL_d_alpha_raw = dL_dalpha * alpha_grad_scale;

                var dp_prim = diffPair(prim);

                bwd_diff(primitiveCoverage<GaussianPrimitive>)(dp_prim, pixel_center, dL_d_alpha_raw);

                acc_mean2d += dp_prim.d.mean;
                acc_conic  += dp_prim.d.conic;
                acc_amplitude += dp_prim.d.amplitude;
                acc_opacity += dp_prim.d.opacity;

                if (alpha >= ALPHA_MIN_TRACK)
                {
                    acc_absx += abs(dp_prim.d.mean.x);
                    acc_absy += abs(dp_prim.d.mean.y);
                }
            }
        }

        if (splat_valid)
        {
            // shDC = 0.5 + SH_C0 * sh0, so dL/dsh0 = SH_C0 * the accumulated seed.
            acc_color *= SH_C0;

            // Scale all sums by STRIDE (unbiased); STBN's TILE_AREA/k compensation already rides in dL_dC.
            if (BACKWARD_PIXEL_STRIDE > 1u)
            {
                float s = float(BACKWARD_PIXEL_STRIDE);
                acc_mean2d *= s;
                acc_conic  *= s;
                acc_amplitude *= s;
                acc_opacity *= s;
                acc_color  *= s;
                acc_absx *= s;
                acc_absy *= s;
                acc_weight *= s;
            }
            if (track_blend_weight != 0u)
            {
                // STBN visited visit_count of the tile's pixels; scale to all.
                float coverage = float(TILE_AREA) / float(max(visit_count, 1u));
                atomicAddI64(blend_weight_accum, g_idx, toFixed(acc_weight * coverage));
            }

            // projection_backward.slang applies J^T once per gaussian to these tile sums.
            atomicAddGrad(grad_mean2d_x, g_idx, acc_mean2d.x);
            atomicAddGrad(grad_mean2d_y, g_idx, acc_mean2d.y);
            atomicAddGrad(grad_conic_a, g_idx, acc_conic.x);
            atomicAddGrad(grad_conic_b, g_idx, acc_conic.y);
            atomicAddGrad(grad_conic_c, g_idx, acc_conic.z);
            atomicAddGrad(grad_amplitude, g_idx, acc_amplitude);
#if SPLAT_KERNEL == 3
            atomicAddGrad(grad_opacity, g_idx, acc_opacity);
#endif
            atomicAddGrad(grad_color_r, g_idx, acc_color.x);
            atomicAddGrad(grad_color_g, g_idx, acc_color.y);
            atomicAddGrad(grad_color_b, g_idx, acc_color.z);
            if (track_screen_grad != 0u)
            {
                atomicAddGrad(grad_2d_x_iter, g_idx, acc_absx);
                atomicAddGrad(grad_2d_y_iter, g_idx, acc_absy);
            }
        }

        // No barrier here: each warp's scans are self-contained and the per-pixel LDS is read-only.
    }
}


// The backward in use: flat, one warp per live bucket (see live_buckets.slang).
[shader("compute")]
[numthreads(TILE_AREA, 1, 1)]
void backwardRasterizePerBucket(
    uint3 gid: SV_GroupID,
    uint3 gtid: SV_GroupThreadID,
    StructuredBuffer<BlendData> blend_data,
    StructuredBuffer<uint> tile_start,
    StructuredBuffer<uint> tile_end,
    StructuredBuffer<uint> sorted_gaussian_ids,
    StructuredBuffer<uint> bucket_offsets,
    StructuredBuffer<uint2> live_buckets,        // (tile, local_bucket) per slot
    StructuredBuffer<uint> live_bucket_total,    // [1]: number of live buckets
    StructuredBuffer<uint2> raster_checkpoints,
    StructuredBuffer<CameraData> camera,
    Texture2D<float4> output_image,
    Texture2D<uint> gaussian_count_per_pixel,
    Texture2D<float4> dL_dC_texture,
    AtomicGradBuffer grad_mean2d_x,
    AtomicGradBuffer grad_mean2d_y,
    AtomicGradBuffer grad_conic_a,
    AtomicGradBuffer grad_conic_b,
    AtomicGradBuffer grad_conic_c,
    AtomicGradBuffer grad_amplitude,
    AtomicGradBuffer grad_opacity,
    AtomicGradBuffer grad_color_r,
    AtomicGradBuffer grad_color_g,
    AtomicGradBuffer grad_color_b,
    AtomicGradBuffer grad_2d_x_iter,
    AtomicGradBuffer grad_2d_y_iter,
    AtomicI64Buffer blend_weight_accum,
    uniform uint track_blend_weight,
    uniform uint track_screen_grad,
    uniform uint pixel_phase,
#if STBN_ENABLED
    StructuredBuffer<uint> stbn_table,
    uniform uint2 stbn_dims,
    uniform uint stbn_slice,
    uniform uint stbn_tile_pixels,
#endif
    uniform float extinction_hardness)
{
    uint tid = gtid.x;
    uint wave_lane = trainerWaveLaneIndex(tid);
    uint lane = wave_lane % RASTER_BUCKET_SIZE;
    uint warp_idx = tid / RASTER_BUCKET_SIZE;

    // Tail warps (slot >= live_total) must still reach the staging barrier, so they return after it.
    uint live_total = live_bucket_total[0];
    uint group = gid.x + gid.y * LINEAR_DISPATCH_WIDTH;
    // Skip whole padded groups before the staging barrier; a partial group must still reach it.
    if (live_total == 0u || group > (live_total - 1u) / NUM_WARPS) return;
    uint slot = group * NUM_WARPS + warp_idx;
    bool bucket_valid = (slot < live_total);

    uint2 frame_dim_px = uint2(camera[0].image_size);
    uint2 grid_tiles = (frame_dim_px + uint2(TILE_WIDTH, TILE_HEIGHT) - 1u)
                       / uint2(TILE_WIDTH, TILE_HEIGHT);

    uint tile_idx = 0u, local_bucket = 0u;
    uint t_start = 0u, tile_count = 0u;
    uint tile_origin_x = 0u, tile_origin_y = 0u;
    uint cp_bucket = 0u, bucket_first_splat = 0u;
    if (bucket_valid)
    {
        uint2 lb = live_buckets[slot];
        tile_idx = lb.x;
        local_bucket = lb.y;
        uint tile_x = tile_idx % grid_tiles.x;
        uint tile_y = tile_idx / grid_tiles.x;
        tile_origin_x = tile_x * TILE_WIDTH;
        tile_origin_y = tile_y * TILE_HEIGHT;
        t_start = tile_start[tile_idx];
        uint t_end = tile_end[tile_idx];
        tile_count = (t_end > t_start) ? (t_end - t_start) : 0u;
        cp_bucket = bucket_offsets[tile_idx] + local_bucket;
        bucket_first_splat = local_bucket * RASTER_BUCKET_SIZE;
    }

    uint phase = pixel_phase % BACKWARD_PIXEL_STRIDE;   // 0 at STRIDE=1
#if STBN_ENABLED
    // Per-warp visit list: warps in one workgroup can own buckets of different tiles.
    uint visit_count = 0u;
    if (bucket_valid)
    {
        uint stbn_base = stbnBlockBase(uint2(tile_origin_x, tile_origin_y),
                                       stbn_dims, stbn_slice);
        // Table count when the mask is on, the full tile when off (see backwardRasterizePerSplat).
        visit_count = stbnActive(stbn_tile_pixels)
            ? min(stbnTileCount(stbn_table, stbn_base), STBN_VISIT_CAPACITY)
            : TILE_AREA;
        uint cp_base = checkpointIndex(cp_bucket, 0u);
        for (uint k = lane; k < visit_count; k += RASTER_BUCKET_SIZE)
        {
            uint p = stbnPixelAtSlot(stbn_table, stbn_base, k);
            gs_stbn_visit[warp_idx][k] = p;
            gs_checkpoints[warp_idx][k] = raster_checkpoints[cp_base + p];
        }
    }
#else
    uint visit_count = TILE_AREA;
    if (bucket_valid)
    {
        uint cp_base = checkpointIndex(cp_bucket, 0u);
        for (uint k = lane; k < STAGED_PIXELS; k += RASTER_BUCKET_SIZE)
            gs_checkpoints[warp_idx][k] = raster_checkpoints[cp_base + phase + k * BACKWARD_PIXEL_STRIDE];
    }
#endif
    // RAW: the pixel loop reads checkpoints staged by other lanes; every warp must arrive first.
    GroupMemoryBarrierWithGroupSync();
    if (!bucket_valid) return;

    uint local_splat = local_bucket * RASTER_BUCKET_SIZE + lane;
    bool splat_valid = (local_splat < tile_count);
    uint g_idx = 0u;
    BlendData blend_only_data;
    if (splat_valid)
    {
        g_idx = sorted_gaussian_ids[t_start + local_splat];
        blend_only_data = blend_data[g_idx];
    }

    float2 acc_mean2d = float2(0.0);
    float3 acc_conic  = float3(0.0);
    float  acc_amplitude = 0.0;
    float  acc_opacity = 0.0;
    float3 acc_color  = float3(0.0);
    float  acc_absx = 0.0;
    float  acc_absy = 0.0;
    float  acc_weight = 0.0;

    GaussianPrimitive prim = splat_valid ? GaussianPrimitive.make(blend_only_data)
                                         : GaussianPrimitive.none();
    float3 c_i = prim.color();

    // hardness 1 = alpha-over, k > 1 sharpens extinction (core/compositor.slang).
    Compositor comp;
    comp.hardness = extinction_hardness;

    // The warp shares this 8x8 tile. Compute two gains per lane and reuse them for every visit.
    uint2 origin = uint2(tile_origin_x, tile_origin_y);
    uint2 first = origin + uint2(lane % TILE_WIDTH, lane / TILE_WIDTH);
    float2 gainPair = float2(camera[0].lensGain(first),
                             camera[0].lensGain(first + uint2(0u, RASTER_BUCKET_SIZE / TILE_WIDTH)));

    // Visit slot k is pixel phase + k * STRIDE (or the k-th STBN sample) and checkpoint k. PIXELS_PER_STEP
    // slots per iteration: their loads and wave scans are independent, so they hide each other's latency.
#if STBN_ENABLED
    uint slot_count = visit_count;
#else
    uint slot_count = STAGED_PIXELS; //unless we have the stochastic stride on, this is == TILE_AREA
#endif
    [loop]
    for (uint v = 0u; v < slot_count; v += PIXELS_PER_STEP)
    {
        uint slot[PIXELS_PER_STEP];
        uint2 pixel[PIXELS_PER_STEP];
        uint processed[PIXELS_PER_STEP];
        uint max_processed = 0u;
        [unroll]
        for (uint j = 0u; j < PIXELS_PER_STEP; ++j)
        {
            slot[j] = min(v + j, slot_count - 1u);
#if STBN_ENABLED
            uint p = gs_stbn_visit[warp_idx][slot[j]];
#else
            uint p = phase + slot[j] * BACKWARD_PIXEL_STRIDE;
#endif
            pixel[j] = uint2(tile_origin_x + (p % TILE_WIDTH), tile_origin_y + (p / TILE_WIDTH));
            bool visited = (v + j < slot_count)
                && pixel[j].x < frame_dim_px.x && pixel[j].y < frame_dim_px.y;
            processed[j] = visited ? gaussian_count_per_pixel[pixel[j]] : 0u;
            max_processed = max(max_processed, processed[j]);
            // An unvisited partner still loads (in bounds) but contributes nothing.
            pixel[j] = min(pixel[j], frame_dim_px - 1u);
        }
        // Saturation skip: every ray ended before this bucket. Warp-uniform, so the scans stay in lockstep.
        if (max_processed <= bucket_first_splat)
            continue;

        // Lanes past a pixel's early break get alpha = 0: a scan identity (T_step = 1, c*alpha*T = 0).
        bool contributes[PIXELS_PER_STEP];
        float2 pixel_center[PIXELS_PER_STEP];
        float alpha[PIXELS_PER_STEP];
        bool alpha_unclamped[PIXELS_PER_STEP];
        float T_checkpoint[PIXELS_PER_STEP];
        float3 C_checkpoint[PIXELS_PER_STEP];
        float dLdC_c[PIXELS_PER_STEP];
        float dLdC_C_checkpoint[PIXELS_PER_STEP];
        float4 dLdC4[PIXELS_PER_STEP];
        float4 C_total[PIXELS_PER_STEP];
        PixelStepVector T_step;
        PixelStepVector dLdC_weight;
        [unroll]
        for (uint j = 0u; j < PIXELS_PER_STEP; ++j)
        {
            contributes[j] = splat_valid && (local_splat < processed[j]);
            pixel_center[j] = float2(pixel[j]) + 0.5;
            unpackCheckpoint(gs_checkpoints[warp_idx][slot[j]], T_checkpoint[j], C_checkpoint[j]);
            dLdC4[j] = dL_dC_texture[pixel[j]];
            C_total[j] = output_image[pixel[j]];
            uint2 local = pixel[j] - origin;
            uint index = local.y * TILE_WIDTH + local.x;
            float gain = WaveReadLaneAt(index < RASTER_BUCKET_SIZE ? gainPair.x : gainPair.y,
                                        bucketWaveLane(wave_lane, index % RASTER_BUCKET_SIZE));
            dLdC4[j].rgb *= gain;
            C_total[j].rgb /= gain;  // Checkpoints and splat colors are in scene space.
            dLdC_c[j] = dot(dLdC4[j].rgb, c_i);
            dLdC_C_checkpoint[j] = dot(dLdC4[j].rgb, C_checkpoint[j]);

            alpha[j] = 0.0;
            alpha_unclamped[j] = false;
            if (contributes[j])
            {
                float alpha_raw = prim.coverage(pixel_center[j]);
                alpha[j] = min(alpha_raw, ALPHA_MAX);
                alpha_unclamped[j] = (alpha_raw <= ALPHA_MAX);
                if (alpha[j] < ALPHA_MIN_TRACK)
                {
                    alpha[j] = 0.0;
                    alpha_unclamped[j] = false;
                }
            }
            T_step[j] = comp.transmittance(alpha[j]);
            dLdC_weight[j] = dLdC_c[j] * comp.weight(alpha[j]);
        }

#if BACKWARD_BLEND_SCAN_MODE == 0
        PixelStepVector T_prefix, dLdC_C_in_front_prefix;
        blendPrefixExclusive(T_step, dLdC_weight, wave_lane, lane, T_prefix, dLdC_C_in_front_prefix);
#else
        PixelStepVector T_prefix;
        [unroll]
        for (uint j = 0u; j < PIXELS_PER_STEP; ++j)
            T_prefix[j] = bucketPrefixProduct(T_step[j], wave_lane);
#endif

        [unroll]
        for (uint j = 0u; j < PIXELS_PER_STEP; ++j)
        {
            float T_i     = T_checkpoint[j] * T_prefix[j];
            float alpha_T = comp.weight(alpha[j]) * T_i;
#if BACKWARD_BLEND_SCAN_MODE == 0
            float dLdC_contrib = dLdC_weight[j] * T_i;
            float dLdC_C_in_front = dLdC_C_checkpoint[j] + T_checkpoint[j] * dLdC_C_in_front_prefix[j];
#elif BACKWARD_BLEND_SCAN_MODE == 2
            float dLdC_contrib = dLdC_weight[j] * T_i;
            float dLdC_C_in_front = dLdC_C_checkpoint[j] + bucketPrefixSumScalar(dLdC_contrib, wave_lane);
#else
            float3 contrib = c_i * alpha_T;
            float3 C_in_front = C_checkpoint[j] + bucketPrefixSum(contrib, wave_lane);
#endif

            if (contributes[j])
            {
                float  T_final = 1.0 - C_total[j].a;

                acc_color += dLdC4[j].rgb * alpha_T;
                acc_weight += alpha_T;

#if BACKWARD_BLEND_SCAN_MODE == 0 || BACKWARD_BLEND_SCAN_MODE == 2
                float dLdC_Cbehind = dot(dLdC4[j].rgb, C_total[j].rgb) - dLdC_C_in_front - dLdC_contrib;
                float dL_dalpha = comp.coverageGrad(alpha[j], dLdC_c[j], T_i, dLdC_Cbehind, dLdC4[j].a,
                                                    T_final);
#else
                // C_total = C_in_front + contrib + C_behind.
                float3 C_behind = C_total[j].rgb - C_in_front - contrib;
                float dL_dalpha = comp.coverageGrad(alpha[j], dLdC_c[j], T_i,
                                                    dot(dLdC4[j].rgb, C_behind), dLdC4[j].a, T_final);
#endif
                var dp_prim = diffPair(prim);
                bwd_diff(primitiveCoverage<GaussianPrimitive>)(
                    dp_prim, pixel_center[j], alpha_unclamped[j] ? dL_dalpha : 0.0);

                acc_mean2d += dp_prim.d.mean;
                acc_conic  += dp_prim.d.conic;
                acc_amplitude += dp_prim.d.amplitude;
                acc_opacity += dp_prim.d.opacity;

                if (alpha[j] >= ALPHA_MIN_TRACK)
                {
                    acc_absx += abs(dp_prim.d.mean.x);
                    acc_absy += abs(dp_prim.d.mean.y);
                }
            }
        }
    }

    if (splat_valid)
    {
        // shDC = 0.5 + SH_C0 * sh0, so dL/dsh0 = SH_C0 * the accumulated seed.
        acc_color *= SH_C0;

        if (BACKWARD_PIXEL_STRIDE > 1u)
        {
            float s = float(BACKWARD_PIXEL_STRIDE);
            acc_mean2d *= s;
            acc_conic  *= s;
            acc_amplitude *= s;
            acc_opacity *= s;
            acc_color  *= s;
            acc_absx *= s;
            acc_absy *= s;
            acc_weight *= s;
        }
        if (track_blend_weight != 0u)
        {
            float coverage = float(TILE_AREA) / float(max(visit_count, 1u));
            atomicAddI64(blend_weight_accum, g_idx, toFixed(acc_weight * coverage));
        }
        atomicAddGrad(grad_mean2d_x, g_idx, acc_mean2d.x);
        atomicAddGrad(grad_mean2d_y, g_idx, acc_mean2d.y);
        atomicAddGrad(grad_conic_a, g_idx, acc_conic.x);
        atomicAddGrad(grad_conic_b, g_idx, acc_conic.y);
        atomicAddGrad(grad_conic_c, g_idx, acc_conic.z);
        atomicAddGrad(grad_amplitude, g_idx, acc_amplitude);
#if SPLAT_KERNEL == 3
        atomicAddGrad(grad_opacity, g_idx, acc_opacity);
#endif
        atomicAddGrad(grad_color_r, g_idx, acc_color.x);
        atomicAddGrad(grad_color_g, g_idx, acc_color.y);
        atomicAddGrad(grad_color_b, g_idx, acc_color.z);
        if (track_screen_grad != 0u)
        {
            atomicAddGrad(grad_2d_x_iter, g_idx, acc_absx);
            atomicAddGrad(grad_2d_y_iter, g_idx, acc_absy);
        }
    }
}
