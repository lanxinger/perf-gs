// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Binning phase 1: project each live gaussian, count the tiles its ellipse touches (AccuTile) and
// key it by depth for the front-to-back splat sort that orders bin_emit's pairs.

#include "../../common/types.slang"
#include "../../common/math.slang"
#include "../../primitives/gaussian/project.slang"
#include "splat_tiles.slang"

// TODO: try merging this and bin_emit.slang into a single kernel.

#include "common/execution.slang"

// Splats that touch no tile sort after every real depth and emit nothing.
static const uint UNBINNED_DEPTH_KEY = 0xFFFFFFFFu;

// Order-preserving uint key; valid because the bin cull guarantees depth > z_near > 0.
uint depthToSortable(float depth)
{
    return asuint(depth) ^ 0x80000000u;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void binCount(
    uint3 dtid: SV_DispatchThreadID,
    RWStructuredBuffer<Gaussian3D> gaussians,
    StructuredBuffer<uint> num_gaussians_counter,
    StructuredBuffer<CameraData> camera,
    RWStructuredBuffer<Projected2DGaussian> projected_2d,
    RWStructuredBuffer<BlendData> blend_data,          // [max_splats], compact blend gather record
    RWStructuredBuffer<uint> tile_count_per_gaussian,  // [max_splats]
    RWStructuredBuffer<uint> depth_keys,               // [max_splats], sorted with depth_ids
    RWStructuredBuffer<uint> depth_ids,                // [max_splats]
    RWStructuredBuffer<uint> depth_count,              // [1] splats the depth sort orders
    StructuredBuffer<uint> cluster_visible,            // [ceil(max_splats/128)]
    StructuredBuffer<float> filter_3D,                 // [max_splats], mip 3D size (0=off)
    uniform uint mip_enabled,
    uniform float kernel_size,                         // 2D dilation (0.3 vanilla / 0.1 mip)
    uniform float z_near,
    uniform uint max_splats)
{
    uint gid = dtid.x;
    uint live = min(num_gaussians_counter[0], max_splats);
    bool in_bounds = (gid < max_splats);
    if (in_bounds)
    {
        // Written before any early skip so downstream kernels (e.g. score_structure_aware) never see stale counts.
        tile_count_per_gaussian[gid] = 0u;
    }

    bool visible = false;
    uint depth_key = UNBINNED_DEPTH_KEY;

    if (in_bounds && gid < live && cluster_visible[gid / CLUSTER_SIZE] != 0u)
    {
        Gaussian3D g = gaussians[gid];

        float2 mean_2d;
        float3 cov2d;
        float depth_vs;
        float cov2d_det;

        // Mip-Splatting 3D filter and opacity compensation (1.0 when off); f3d = 0, kernel_size = 0.3 is vanilla.
        float f3d = (mip_enabled != 0u) ? filter_3D[gid] : 0.0;
        float comp;

        GaussianModel model;
        model.mean = g.mean;
        model.log_scale = g.log_scale;
        model.rot_quat = g.rot_quat;
        model.projectInto(camera[0], f3d, kernel_size, mip_enabled,
                          mean_2d, cov2d, depth_vs, cov2d_det, comp);

        if (depth_vs >= z_near)
        {
            BlendData bd;
            bd.mean = mean_2d;
            bd.conic = conicFromCov(cov2d);
            bd.amplitude = splatAmplitude(g.opacity, comp);
            bd.color = shDC(g.sh0);
            bd.opacity = g.opacity;

            // bin_emit replays this exact walk, so its pairs match the scanned offsets.
            SplatTiles tiles;
            if (SplatTiles.make(bd, camera[0].image_size, tiles))
            {
                uint2 unused_tile;
                uint count = tiles.walk(0xFFFFFFFFu, unused_tile);
                if (count > 0u)
                {
                    Projected2DGaussian proj;
                    proj.mean = mean_2d;
                    proj.cov2d = cov2d;
                    proj.depth_vs = depth_vs;
                    proj.opacity_compensation = comp;
                    projected_2d[gid] = proj;
                    blend_data[gid] = bd;
                    tile_count_per_gaussian[gid] = count;
                    depth_key = depthToSortable(depth_vs);
                    visible = true;
                }
            }
        }
    }

    // Stream-compact only visible splats into depth_keys/depth_ids (one atomic per wave) so
    // depth_sort, sh_forward, scan_tile_offsets and bin_emit skip off-screen/culled splats.
    uint wave_vis = WaveActiveSum(visible ? 1u : 0u);
    uint wave_off = WavePrefixSum(visible ? 1u : 0u);
    uint base = 0u;
    if (WaveIsFirstLane() && wave_vis > 0u)
        InterlockedAdd(depth_count[0], wave_vis, base);
    base = WaveReadLaneFirst(base);
    if (visible)
    {
        uint dst = base + wave_off;
        depth_keys[dst] = depth_key;
        depth_ids[dst] = gid;
    }
}
