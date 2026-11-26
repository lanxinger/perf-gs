// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Optional view-dependent SH backward (sh_degree > 0), one thread per gaussian: turns the blend's
// grad_color (SH_C0 * dL/d(clamped rgb)) into higher-band and view-direction mean gradients.

#include "../../common/types.slang"
#include "../../primitives/gaussian/sh.slang"

#include "common/execution.slang"

[shader("compute")]
[numthreads(256, 1, 1)]
void shBackward(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<Gaussian3D> gaussians,
    StructuredBuffer<uint> num_gaussians_counter,
    StructuredBuffer<CameraData> camera,
    StructuredBuffer<uint> features_rest,
    RWStructuredBuffer<uint> features_rest_grad,
    // Accumulated color gradient (read; clamped channels zeroed for the DC term).
    GradBuffer grad_color_r,
    GradBuffer grad_color_g,
    GradBuffer grad_color_b,
    // Adds the direction->mean term after projection_backward's write (single writer per gid).
    GradBuffer grad_mean_x,
    GradBuffer grad_mean_y,
    GradBuffer grad_mean_z,
    StructuredBuffer<uint> cluster_visible,
    uniform uint active_sh_degree,
    uniform uint rest_floats,
    uniform uint max_splats)
{
    uint gid = dtid.x;
    if (gid >= max_splats) return;
    if (gid >= num_gaussians_counter[0]) return;
    // Culled clusters get no grad_color and the rest-Adam skips them too, so stale grads stay unread.
    if (cluster_visible[gid / CLUSTER_SIZE] == 0u) return;

    Gaussian3D g = gaussians[gid];
    float3 cam = cameraCenterWorld(camera[0].view_matrix);
    int deg = int(active_sh_degree);

    float3 gaussianColor = float3(fromGrad(grad_color_r[gid]),
                       fromGrad(grad_color_g[gid]),
                       fromGrad(grad_color_b[gid]));

    // Higher-band gradients packed fp16, 2 scalars per uint, into features_rest_grad (sole writer).
    uint fu = halfFeatUints(rest_floats);
    uint gbase = gid * fu;

    // No color gradient (e.g. no tile pairs this view): every output gradient is zero.
    if (all(gaussianColor == float3(0.0)))
    {
        for (uint u = 0u; u < fu; ++u)
            features_rest_grad[gbase + u] = 0u;
        return;
    }

    // Only the active bands (excluding DC): coeffs beyond deg are ignored and get zero gradient.
    uint active_coeffs = uint((deg + 1) * (deg + 1) - 1);
    float3 rest[SH_REST_COEFFS] = loadSHRestActive(features_rest, gid, rest_floats,
                                                   active_coeffs);

    // Primal (pre-clamp) color to reconstruct the forward's >=0 clamp mask.
    float3 rgb_raw = shColorFromMean(g.sh0, rest, g.mean, cam, deg);
    float3 mask = float3(rgb_raw.x > 0.0 ? 1.0 : 0.0,
                         rgb_raw.y > 0.0 ? 1.0 : 0.0,
                         rgb_raw.z > 0.0 ? 1.0 : 0.0);

    // grad_color == SH_C0 * dL/d(rgb_clamped); route through the clamp.
    float3 seed = (gaussianColor / SH_C0) * mask;

    var dp_rest = diffPair(rest);
    var dp_mean = diffPair(g.mean);
    bwd_diff(shColorFromMean)(g.sh0, dp_rest, dp_mean, cam, deg, seed);

    [ForceUnroll]
    for (uint u = 0u; u < uint(SH_FEAT_UINTS_MAX); ++u)
    {
        if (u >= fu) continue;
        uint s0 = u * 2u, s1 = s0 + 1u;
        uint k0 = s0 / 3u, c0 = s0 % 3u;
        float3 v0 = dp_rest.d[k0];
        float a = (c0 == 0u) ? v0.x : (c0 == 1u ? v0.y : v0.z);
        float b = 0.0;
        if (s1 < rest_floats)
        {
            // min() keeps the unrolled static index in bounds (s1 == 45 -> k1 == 15); no-op at runtime.
            uint k1 = min(s1 / 3u, uint(SH_REST_COEFFS) - 1u), c1 = s1 % 3u;
            float3 v1 = dp_rest.d[k1];
            b = (c1 == 0u) ? v1.x : (c1 == 1u ? v1.y : v1.z);
        }
        features_rest_grad[gbase + u] = packHalf2x(a, b);
    }

    // DC gradient is grad_color for unclamped channels; zero the clamped ones.
    if (mask.x == 0.0) grad_color_r[gid] = GradScalar(0);
    if (mask.y == 0.0) grad_color_g[gid] = GradScalar(0);
    if (mask.z == 0.0) grad_color_b[gid] = GradScalar(0);

    grad_mean_x[gid] = grad_mean_x[gid] + toGrad(dp_mean.d.x);
    grad_mean_y[gid] = grad_mean_y[gid] + toGrad(dp_mean.d.y);
    grad_mean_z[gid] = grad_mean_z[gid] + toGrad(dp_mean.d.z);
}
