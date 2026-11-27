// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Simple threshold-based densification (reference 3DGS strategy): prune transparent, oversized or
// underweight splats; densify high-gradient ones (split if max_scale > scale_thresh, else clone).

#include "../common/types.slang"
#include "../common/math.slang"
#include "blend_weight.slang"

static const uint ACTION_KEEP  = 0;
static const uint ACTION_SPLIT = 1;
static const uint ACTION_CLONE = 2;
static const uint ACTION_PRUNE = 3;

[shader("compute")]
[numthreads(256, 1, 1)]
void decideActionsSimple(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<Gaussian3D> gaussians,
    StructuredBuffer<uint> num_gaussians_counter,
    ReadGradBuffer grad_mean_2d_mag_accum,
    StructuredBuffer<uint> visibility_count_accum,
    StructuredBuffer<int64_t> blend_weight_accum,
    RWStructuredBuffer<uint> action,
    uniform float grad_thresh,
    uniform float scale_thresh,
    uniform float t_opacity,
    uniform float t_big_scale,      // world-space; caller passes prune_scale3d * scene_scale
    uniform uint allow_scale_prune, // 0 until after the first opacity reset (matches gsplat)
    uniform float min_blend_weight, // pixels per visible view; 0 disables
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats)
        return;

    uint n = num_gaussians_counter[0];
    if (i >= n)
    {
        action[i] = ACTION_KEEP;
        return;
    }

    Gaussian3D g = gaussians[i];
    float3 scale = exp(g.log_scale);
    float max_scale = max(scale.x, max(scale.y, scale.z));
    float opacity = sigmoid(g.opacity);

    // Prune: too transparent (always active, like gsplat's prune_opa).
    if (opacity < t_opacity)
    {
        action[i] = ACTION_PRUNE;
        return;
    }

    // Prune too-big splats only after the first opacity reset, so big init blobs get subdivided first.
    if (allow_scale_prune != 0u && max_scale > t_big_scale)
    {
        action[i] = ACTION_PRUNE;
        return;
    }

    // Densification signal: average 2D gradient magnitude.
    uint vis = visibility_count_accum[i];

    if (isUnderweight(blend_weight_accum[i], vis, min_blend_weight))
    {
        action[i] = ACTION_PRUNE;
        return;
    }
    
    float avg_grad = 0.0;
    if (vis > 0)
        avg_grad = fromGrad(grad_mean_2d_mag_accum[i]) / float(vis);

    if (avg_grad > grad_thresh)
    {
        if (max_scale > scale_thresh)
            action[i] = ACTION_SPLIT;
        else
            action[i] = ACTION_CLONE;
        return;
    }

    action[i] = ACTION_KEEP;
}
