// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Clear the densification tracking buffers.

#include "../common/fixed_point.slang"

[shader("compute")]
[numthreads(256, 1, 1)]
void clearTracking(
    uint3 dtid: SV_DispatchThreadID,
    GradBuffer grad_mean_2d_mag_accum,
    RWStructuredBuffer<uint> visibility_count_accum,
    GradBuffer grad_2d_x_iter,
    GradBuffer grad_2d_y_iter,
    RWStructuredBuffer<fixed_t> blend_weight_accum,
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;
    grad_mean_2d_mag_accum[i] = GradScalar(0);
    visibility_count_accum[i] = 0u;
    grad_2d_x_iter[i] = GradScalar(0);
    grad_2d_y_iter[i] = GradScalar(0);
    blend_weight_accum[i] = fixed_t(0);
}
