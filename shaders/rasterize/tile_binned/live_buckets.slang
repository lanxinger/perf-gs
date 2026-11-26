// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Live-bucket compaction: the flat (warp-per-bucket) backward dispatches only buckets some pixel reached.
// Host order: tileMaxProcessed, scan, scatterLiveBuckets, prepareBucketDispatchArgs.

#include "checkpoint.slang"   // RASTER_BUCKET_SIZE
#include "../../common/stbn.slang"

#include "common/execution.slang"
// Must match backwardRasterizePerBucket's numthreads(TILE_AREA) and slot = gid.x * NUM_WARPS + warp.
static const uint FLAT_WARPS_PER_WG = TILE_AREA / RASTER_BUCKET_SIZE;

groupshared uint gs_tile_max;

// Pass 1: live buckets = ceil(max processed / P); one TILE_AREA-thread workgroup per tile (tid = pixel).
[shader("compute")]
[numthreads(TILE_AREA, 1, 1)]
void tileMaxProcessed(
    uint3 gid: SV_GroupID,
    uint3 gtid: SV_GroupThreadID,
    Texture2D<uint> gaussian_count_per_pixel,
    RWStructuredBuffer<uint> live_bucket_count,
    // STBN mask, runtime-gated (not STBN_ENABLED) so the STBN_ENABLED=0 reference sees the same buckets.
    StructuredBuffer<uint> stbn_table,
    uniform uint2 stbn_dims,
    uniform uint stbn_slice,
    uniform uint stbn_tile_pixels,
    uniform uint2 frame_dim_px,
    uniform uint num_tiles_x,
    uniform uint num_tiles)
{
    uint tile_idx = gid.x;
    if (tile_idx >= num_tiles) return;

    uint tid = gtid.x;
    if (tid == 0u) gs_tile_max = 0u;
    GroupMemoryBarrierWithGroupSync();

    uint tile_x = tile_idx % num_tiles_x;
    uint tile_y = tile_idx / num_tiles_x;
    uint2 pixel = uint2(tile_x * TILE_WIDTH  + (tid % TILE_WIDTH),
                        tile_y * TILE_HEIGHT + (tid / TILE_WIDTH));
    bool contributes = pixel.x < frame_dim_px.x && pixel.y < frame_dim_px.y && stbnSelected(stbn_table, pixel, stbn_dims, stbn_slice, stbn_tile_pixels);
    uint processed = contributes ? gaussian_count_per_pixel[pixel] : 0u;
    InterlockedMax(gs_tile_max, processed);
    GroupMemoryBarrierWithGroupSync();

    if (tid == 0u)
    {
        uint m = gs_tile_max;
        live_bucket_count[tile_idx] = (m + RASTER_BUCKET_SIZE - 1u) / RASTER_BUCKET_SIZE;
    }
}

// Pass 3, after the exclusive scan of live_bucket_count: one thread per tile writes its live buckets.
[shader("compute")]
[numthreads(256, 1, 1)]
void scatterLiveBuckets(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<uint> live_bucket_count,
    StructuredBuffer<uint> live_bucket_offsets,
    RWStructuredBuffer<uint2> live_buckets,
    uniform uint num_tiles)
{
    uint t = dtid.x;
    if (t >= num_tiles) return;
    uint cnt = live_bucket_count[t];
    uint off = live_bucket_offsets[t];
    for (uint k = 0u; k < cnt; ++k)
        live_buckets[off + k] = uint2(t, k);
}

// Pass 4: indirect-dispatch args for the flat backward.
[shader("compute")]
[numthreads(1, 1, 1)]
void prepareBucketDispatchArgs(
    RWStructuredBuffer<uint> live_bucket_total,
    RWStructuredBuffer<uint> dispatch_args,
    uniform uint max_buckets)   // defensive cap: never queue more than we sized
{
    uint total = min(live_bucket_total[0], max_buckets);
    uint groups = (total + FLAT_WARPS_PER_WG - 1u) / FLAT_WARPS_PER_WG;
    // Vulkan/D3D12 require at least one group; the in-kernel slot >= total guard makes it a no-op.
    if (groups == 0u) groups = 1u;
    dispatch_args[0] = min(groups, LINEAR_DISPATCH_WIDTH);
    dispatch_args[1] = (groups - 1u) / LINEAR_DISPATCH_WIDTH + 1u;
    dispatch_args[2] = 1u;
}
