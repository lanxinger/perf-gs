// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Optional view-dependent SH forward (sh_degree > 0): once per gaussian, overwrites bin_count's
// BlendData.color (the DC color) with the full SH color, clamped at 0.

#include "../../common/types.slang"
#include "../../primitives/gaussian/sh.slang"

#include "common/execution.slang"

[shader("compute")]
[numthreads(256, 1, 1)]
void shForward(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<Gaussian3D> gaussians,
    StructuredBuffer<uint> visible_ids,
    StructuredBuffer<uint> visible_count,
    StructuredBuffer<CameraData> camera,
    StructuredBuffer<uint> features_rest,
    RWStructuredBuffer<BlendData> blend_data,
    uniform uint active_sh_degree,
    uniform uint rest_floats,
    uniform uint max_splats)
{
    uint idx = dtid.x;
    if (idx >= min(visible_count[0], max_splats)) return;
    uint gid = visible_ids[idx];

    Gaussian3D g = gaussians[gid];
    float3 cam_center = cameraCenterWorld(camera[0].view_matrix);

    uint active_coeffs = (active_sh_degree + 1u) * (active_sh_degree + 1u) - 1u;
    float3 rest[SH_REST_COEFFS] = loadSHRestActive(features_rest, gid, rest_floats,
                                                   active_coeffs);
    float3 rgb = max(shColorFromMean(g.sh0, rest, g.mean, cam_center,
                                     int(active_sh_degree)), 0.0);

    blend_data[gid].color = rgb;
}
