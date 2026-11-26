// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Per-iter densification tracking fold (int64 fixed-point).

#include "../common/types.slang"

#include "common/execution.slang"

[shader("compute")]
[numthreads(256, 1, 1)]
void accumulateIter(
    uint3 dtid: SV_DispatchThreadID,
    RWStructuredBuffer<int64_t> grad_2d_x_iter,
    RWStructuredBuffer<int64_t> grad_2d_y_iter,
    AtomicI64Buffer grad_mean_2d_mag_accum,
    RWStructuredBuffer<uint> visibility_count_accum,
    StructuredBuffer<uint> num_gaussians_counter,
    // Cluster-cull mask: seeded all-ones when disabled.
    StructuredBuffer<uint> cluster_visible,
    uniform float2 frame_dim,
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;
    // Dead slots and culled clusters got no screen gradient (already folded to zero), so skip their reads.
    if (i >= num_gaussians_counter[0]) return;
    if (cluster_visible[i / CLUSTER_SIZE] == 0u) return;

    int64_t ix = grad_2d_x_iter[i];
    int64_t iy = grad_2d_y_iter[i];

    if (ix == int64_t(0) && iy == int64_t(0)) return;

    // Pixel -> NDC units before the norm, matching gsplat's densification signal.
    float fx = fromFixed(ix) * (frame_dim.x * 0.5);
    float fy = fromFixed(iy) * (frame_dim.y * 0.5);
    float m = sqrt(fx * fx + fy * fy);

    atomicAddI64(grad_mean_2d_mag_accum, i, toFixed(m));
    InterlockedAdd(visibility_count_accum[i], 1u);

    grad_2d_x_iter[i] = int64_t(0);
    grad_2d_y_iter[i] = int64_t(0);
}
