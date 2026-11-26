// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Tile-binned forward rasterizer, front to back: one warp per tile with two pixels per lane (rows r and
// r + TILE_HEIGHT / 2), so each splat load serves two pixels; a workgroup holds two vertically adjacent
// tiles. Writes each pixel's early-break stop index so the backward replays the same entries.

#include "../../common/types.slang"
#include "../../common/math.slang"
#include "../../primitives/gaussian/eval.slang"
#include "../../primitives/gaussian/blend.slang"
#include "blend_forward.slang"
#include "common/execution.slang"

static const uint PIXELS_PER_THREAD = 2u;
static const uint TILE_ROWS_PER_PASS = TILE_HEIGHT / PIXELS_PER_THREAD;
#ifdef TRAINER_METAL_BACKEND
// On Apple Silicon (32-wide SIMD-group), 1 tile per 32-thread workgroup lets all 32 lanes
// cooperatively preload each 32-splat bucket into threadgroup memory in parallel and exit
// independently when the tile saturates (~2x faster than un-staged per-splat gathers).
static const uint TILES_PER_GROUP = 1u;
groupshared BlendData gs_forward_bucket[RASTER_BUCKET_SIZE];
#else
static const uint TILES_PER_GROUP = 2u;
#endif

[shader("compute")]
[numthreads(TILE_WIDTH, TILE_ROWS_PER_PASS * TILES_PER_GROUP, 1)]
void forwardRasterize(
    uint3 gid: SV_GroupID,
    uint3 gtid: SV_GroupThreadID,
    // Compact blend record packed by bin_count, gathered identically by the backward.
    StructuredBuffer<BlendData> blend_data,
    StructuredBuffer<uint> tile_start,
    StructuredBuffer<uint> tile_end,
    StructuredBuffer<uint> sorted_gaussian_ids,   // the value array after sort
    StructuredBuffer<CameraData> camera,
    // Ragged (CSR) (T, C_in_front) checkpoints, one per RASTER_BUCKET_SIZE splats, for the per-splat backward.
    StructuredBuffer<uint> bucket_offsets,
    RWStructuredBuffer<uint2> raster_checkpoints,
    RWTexture2D<float4> output_image,
    RWTexture2D<uint> gaussian_count_per_pixel,
    uniform float extinction_hardness)
{
    uint2 frame_dim_px = uint2(camera[0].image_size);
    uint2 tile_dim = uint2(TILE_WIDTH, TILE_HEIGHT);
    uint2 num_tiles = (frame_dim_px + tile_dim - 1u) / tile_dim;
    uint2 tile = uint2(gid.x, gid.y * TILES_PER_GROUP + gtid.y / TILE_ROWS_PER_PASS);
    if (tile.y >= num_tiles.y) return;

    uint tile_idx = tile.y * num_tiles.x + tile.x;
    uint t_start = tile_start[tile_idx];
    uint t_end   = tile_end[tile_idx];
    uint tile_count = (t_end > t_start) ? (t_end - t_start) : 0;
    uint tile_bucket_base = bucket_offsets[tile_idx];

    uint2 pixel[PIXELS_PER_THREAD];
    float2 pixel_center[PIXELS_PER_THREAD];
    bool valid_pixel[PIXELS_PER_THREAD];
    uint pixel_in_tile[PIXELS_PER_THREAD];   // row-major slot in each bucket's checkpoint block
    [unroll]
    for (uint j = 0u; j < PIXELS_PER_THREAD; ++j)
    {
        uint2 in_tile = uint2(gtid.x, gtid.y % TILE_ROWS_PER_PASS + j * TILE_ROWS_PER_PASS);
        pixel[j] = tile * tile_dim + in_tile;
        pixel_center[j] = float2(pixel[j]) + 0.5;
        valid_pixel[j] = pixel[j].x < frame_dim_px.x && pixel[j].y < frame_dim_px.y;
        pixel_in_tile[j] = in_tile.y * TILE_WIDTH + in_tile.x;
    }

    float3 C[PIXELS_PER_THREAD];
    float T[PIXELS_PER_THREAD];
    uint processed[PIXELS_PER_THREAD];
#ifdef TRAINER_METAL_BACKEND
    uint lane = gtid.y * TILE_WIDTH + gtid.x;
    bool active[PIXELS_PER_THREAD];
    [unroll]
    for (uint j = 0u; j < PIXELS_PER_THREAD; ++j)
    {
        C[j] = float3(0.0);
        T[j] = 1.0;
        processed[j] = 0u;
        active[j] = valid_pixel[j];
    }
    Compositor comp;
    comp.hardness = extinction_hardness;
    for (uint base = 0u; base < tile_count; base += RASTER_BUCKET_SIZE)
    {
        if (!WaveActiveAnyTrue(active[0] || active[1]))
            break;
        uint bucket = tile_bucket_base + base / RASTER_BUCKET_SIZE;
        [unroll]
        for (uint j = 0u; j < PIXELS_PER_THREAD; ++j)
        {
            if (active[j])
                raster_checkpoints[checkpointIndex(bucket, pixel_in_tile[j])] =
                    packCheckpoint(T[j], C[j]);
        }
        uint load_idx = base + lane;
        if (load_idx < tile_count)
            gs_forward_bucket[lane] = blend_data[sorted_gaussian_ids[t_start + load_idx]];
        GroupMemoryBarrierWithGroupSync();

        uint end_k = min(RASTER_BUCKET_SIZE, tile_count - base);
        for (uint k = 0u; k < end_k; ++k)
        {
            GaussianPrimitive prim = GaussianPrimitive.make(gs_forward_bucket[k]);
            [unroll]
            for (uint j = 0u; j < PIXELS_PER_THREAD; ++j)
            {
                if (!active[j])
                    continue;
                float alpha = min(prim.coverage(pixel_center[j]), BLEND_ALPHA_MAX);
                if (alpha >= BLEND_ALPHA_MIN)
                {
                    C[j] += prim.color() * comp.weight(alpha) * T[j];
                    T[j] *= comp.transmittance(alpha);
                    processed[j] = base + k + 1u;
                    active[j] = T[j] >= BLEND_T_EARLY_BREAK;
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
#else
    GaussianSource src;
    src.blend_data = blend_data;
    blendForward<GaussianSource, PIXELS_PER_THREAD>(
        src, t_start, tile_count, pixel_center, valid_pixel,
        pixel_in_tile, tile_bucket_base, sorted_gaussian_ids,
        raster_checkpoints, extinction_hardness, C, T, processed);
#endif

    [unroll]
    for (uint j = 0u; j < PIXELS_PER_THREAD; ++j)
    {
        if (!valid_pixel[j])
            continue;
        // The backward recovers final transmittance as T = 1 - output_image.a.
        output_image[pixel[j]] = float4(C[j] * camera[0].lensGain(pixel[j]), 1.0 - T[j]);
        gaussian_count_per_pixel[pixel[j]] = processed[j];
    }
}
