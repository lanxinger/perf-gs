// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Opacity + scale regularization gradients (int64 fixed-point).

#include "../common/types.slang"
#include "../common/math.slang"

#include "common/execution.slang"

[shader("compute")]
[numthreads(256, 1, 1)]
void addRegularization(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<Gaussian3D> gaussians,
    StructuredBuffer<uint> num_gaussians_counter,
    AtomicI64Buffer grad_opacity,
    AtomicI64Buffer grad_log_scale_x,
    AtomicI64Buffer grad_log_scale_y,
    AtomicI64Buffer grad_log_scale_z,
    StructuredBuffer<uint> cluster_visible,
    uniform float w_opacity,
    uniform float w_scale)
{
    uint i = dtid.x;
    uint n = num_gaussians_counter[0];
    if (i >= n) return;
    if (cluster_visible[i / CLUSTER_SIZE] == 0u) return;

    float inv_n = 1.0 / float(max(n, 1u));

    Gaussian3D g = gaussians[i];

    if (w_opacity > 0.0)
    {
        float sig = sigmoid(g.opacity);
        float d_opa = w_opacity * sig * (1.0 - sig) * inv_n;
        atomicAddI64(grad_opacity, i, toFixed(d_opa));
    }

    if (w_scale > 0.0)
    {
        float3 scale = exp(g.log_scale);
        float3 d_scale = w_scale * scale * inv_n;
        atomicAddI64(grad_log_scale_x, i, toFixed(d_scale.x));
        atomicAddI64(grad_log_scale_y, i, toFixed(d_scale.y));
        atomicAddI64(grad_log_scale_z, i, toFixed(d_scale.z));
    }
}
