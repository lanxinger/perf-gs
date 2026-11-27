// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

#include "../common/types.slang"
#include "../common/execution.slang"
#include "../primitives/gaussian/project.slang"

// One group of THREADS_PER_SPLAT lanes per gaussian, each striding over cameras for a min
// view-space depth; this cuts per-splat latency when few live gaussians under-occupy the device.
static const uint THREADS_PER_SPLAT = 32;
static const uint MIP_DISPATCH_WIDTH = 65535u;

[shader("compute")]
[numthreads(1, 1, 1)]
void prepareMipFilterArgs(
    StructuredBuffer<uint> num_gaussians_counter,
    RWStructuredBuffer<uint> dispatch_args,
    RWStructuredBuffer<uint> previous_count,
    uniform uint max_splats)
{
    uint count = min(num_gaussians_counter[0], max_splats);
    uint old_count = previous_count[0];
    previous_count[0] = count;
    previous_count[1] = max(old_count, count);
    uint groups = max(count, 1u);
    dispatch_args[0] = min(groups, MIP_DISPATCH_WIDTH);
    dispatch_args[1] = (groups + MIP_DISPATCH_WIDTH - 1u) / MIP_DISPATCH_WIDTH;
    dispatch_args[2] = 1u;
    uint clear_count = old_count > count ? old_count - count : 0u;
    uint clear_groups = max((clear_count + 255u) / 256u, 1u);
    dispatch_args[3] = min(clear_groups, MIP_DISPATCH_WIDTH);
    dispatch_args[4] = (clear_groups + MIP_DISPATCH_WIDTH - 1u) / MIP_DISPATCH_WIDTH;
    dispatch_args[5] = 1u;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void clearMipFilterTail(
    uint3 group_id: SV_GroupID,
    uint3 thread_id: SV_GroupThreadID,
    StructuredBuffer<uint> previous_count,
    RWStructuredBuffer<float> filter_3D)
{
    uint i = previous_count[0] + (group_id.x + group_id.y * MIP_DISPATCH_WIDTH)
             * 256u + thread_id.x;
    if (i < previous_count[1]) filter_3D[i] = 0.0;
}

groupshared float s_min_z[THREADS_PER_SPLAT];
groupshared uint  s_any[THREADS_PER_SPLAT];   // bool as uint for the OR reduction

[shader("compute")]
[numthreads(THREADS_PER_SPLAT, 1, 1)]
void computeMip3DFilter(
    uint3 group_id: SV_GroupID,
    uint3 gtid: SV_GroupThreadID,
    StructuredBuffer<Gaussian3D> gaussians,
    StructuredBuffer<uint> num_gaussians_counter,
    StructuredBuffer<CameraData> cameras,
    RWStructuredBuffer<float> filter_3D,     // [max_splats], output
    uniform uint num_cameras,
    uniform float filter_coef,               // sqrt(0.2) / max_focal
    uniform uint max_splats)
{
    uint gid = group_id.x + group_id.y * MIP_DISPATCH_WIDTH;
    uint lane = gtid.x;               // 0 .. THREADS_PER_SPLAT-1
    if (gid >= max_splats) return;

    // Group-uniform early-outs (gid is per group) never split the group before a barrier;
    // clearMipFilterTail zeroes the dead tail.
    if (gid >= num_gaussians_counter[0]) return;

    float3 mean = gaussians[gid].mean;

    float min_z = 1e30;
    bool any_valid = false;

    // Strided so consecutive lanes touch consecutive cameras (coalesced reads).
    for (uint c = lane; c < num_cameras; c += THREADS_PER_SPLAT)
    {
        CameraData cam = cameras[c];

        float3 vs;
        float2 px;
        float clip_w;
        projectMeanToPixel(mean, cam.view_matrix, cam.proj_matrix, cam.image_size, vs, px, clip_w);
        float z = vs.z;
        if (z <= 0.2)
            continue;

        // In-screen test with Mip-Splatting's 1.15x margin.
        float W = cam.image_size.x;
        float H = cam.image_size.y;
        bool in_screen =
            px.x >= -0.15 * W && px.x <= 1.15 * W &&
            px.y >= -0.15 * H && px.y <= 1.15 * H;
        if (!in_screen)
            continue;

        min_z = min(min_z, z);
        any_valid = true;
    }

    // The group usually fits in one wave; narrower waves merge their results after one barrier.
    float group_min = WaveActiveMin(min_z);
    bool group_any = WaveActiveAnyTrue(any_valid);
    uint wave_size = trainerWaveLaneCount();
    if (wave_size < THREADS_PER_SPLAT)
    {
        s_min_z[lane] = group_min;
        s_any[lane] = group_any ? 1u : 0u;
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0u)
        {
            for (uint other = wave_size; other < THREADS_PER_SPLAT; other += wave_size)
            {
                group_min = min(group_min, s_min_z[other]);
                group_any = group_any || (s_any[other] != 0u);
            }
        }
    }

    if (lane == 0)
    {
        // A gaussian seen by no view needs no low-pass (and is a prune candidate).
        filter_3D[gid] = group_any ? (group_min * filter_coef) : 0.0;
    }
}
