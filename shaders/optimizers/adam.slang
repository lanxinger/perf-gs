// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Adam optimizer step
#include "../common/types.slang"

#include "common/execution.slang"

static const float ADAM_BETA1 = 0.9;
static const float ADAM_BETA2 = 0.999;
static const float ADAM_EPSILON = 1e-8;

struct AdamConfig
{
    float lr_mean;
    float lr_scale;
    float lr_rot;
    float lr_opacity;
    float lr_color;
    float bias_correction1;
    float bias_correction2;
    uint step;  // seeds bf16 stochastic rounding
};

float adam_scalar(
    float param, float grad,
    inout float m, inout float v,
    float lr, float bc1, float bc2)
{
    m = ADAM_BETA1 * m + (1.0 - ADAM_BETA1) * grad;
    v = ADAM_BETA2 * v + (1.0 - ADAM_BETA2) * grad * grad;
    float m_hat = m / bc1;
    float v_hat = v / bc2;
    return param - lr * m_hat / (sqrt(v_hat) + ADAM_EPSILON);
}


float3 adam_vec3(float3 p, float3 g, inout float3 m, inout float3 v,
                 float lr, float bc1, float bc2)
{
    return float3(
        adam_scalar(p.x, g.x, m.x, v.x, lr, bc1, bc2),
        adam_scalar(p.y, g.y, m.y, v.y, lr, bc1, bc2),
        adam_scalar(p.z, g.z, m.z, v.z, lr, bc1, bc2)
    );
}

float4 adam_vec4(float4 p, float4 g, inout float4 m, inout float4 v,
                 float lr, float bc1, float bc2)
{
    return float4(
        adam_scalar(p.x, g.x, m.x, v.x, lr, bc1, bc2),
        adam_scalar(p.y, g.y, m.y, v.y, lr, bc1, bc2),
        adam_scalar(p.z, g.z, m.z, v.z, lr, bc1, bc2),
        adam_scalar(p.w, g.w, m.w, v.w, lr, bc1, bc2)
    );
}

[shader("compute")]
[numthreads(256, 1, 1)]
void adamStep(
    uint3 dtid: SV_DispatchThreadID,
    RWStructuredBuffer<Gaussian3D> gaussians,
    RWStructuredBuffer<AdamState> adam_state,
    StructuredBuffer<uint> num_gaussians_counter,
    GradBuffer grad_mean_x,
    GradBuffer grad_mean_y,
    GradBuffer grad_mean_z,
    GradBuffer grad_log_scale_x,
    GradBuffer grad_log_scale_y,
    GradBuffer grad_log_scale_z,
    GradBuffer grad_rot_w,
    GradBuffer grad_rot_x,
    GradBuffer grad_rot_y,
    GradBuffer grad_rot_z,
    GradBuffer grad_opacity,
    GradBuffer grad_color_r,
    GradBuffer grad_color_g,
    GradBuffer grad_color_b,
    // Cluster-cull mask; all ones when culling is disabled.
    StructuredBuffer<uint> cluster_visible,
    uniform AdamConfig cfg,
    uniform float pixels_per_step)
{
    uint idx = dtid.x;
    if (idx >= num_gaussians_counter[0]) return;
    if (cluster_visible[idx / CLUSTER_SIZE] == 0u) return;

    float inv_pixels = 1.0 / max(pixels_per_step, 1.0);

    float3 g_mean = float3(
        fromGrad(grad_mean_x[idx]), fromGrad(grad_mean_y[idx]), fromGrad(grad_mean_z[idx])
    ) * inv_pixels;

    float3 g_log_scale = float3(
        fromGrad(grad_log_scale_x[idx]), fromGrad(grad_log_scale_y[idx]), fromGrad(grad_log_scale_z[idx])
    ) * inv_pixels;

    float4 g_rot = float4(
        fromGrad(grad_rot_w[idx]), fromGrad(grad_rot_x[idx]),
        fromGrad(grad_rot_y[idx]), fromGrad(grad_rot_z[idx])
    ) * inv_pixels;

    float g_op = fromGrad(grad_opacity[idx]) * inv_pixels;

    float3 g_color = float3(
        fromGrad(grad_color_r[idx]), fromGrad(grad_color_g[idx]), fromGrad(grad_color_b[idx])
    ) * inv_pixels;

    bool any_bad = any(isnan(g_mean)) || any(isinf(g_mean))
                || any(isnan(g_log_scale)) || any(isinf(g_log_scale))
                || any(isnan(g_rot)) || any(isinf(g_rot))
                || isnan(g_op) || isinf(g_op)
                || any(isnan(g_color)) || any(isinf(g_color));
    if (any_bad) return;

    Gaussian3D g = gaussians[idx];
    AdamMoments s = loadAdamMoments(adam_state[idx]);

    g.mean = adam_vec3(g.mean, g_mean, s.mean_m, s.mean_v,
                       cfg.lr_mean, cfg.bias_correction1, cfg.bias_correction2);

    g.log_scale = adam_vec3(g.log_scale, g_log_scale, s.log_scale_m, s.log_scale_v,
                            cfg.lr_scale, cfg.bias_correction1, cfg.bias_correction2);

    g.rot_quat = adam_vec4(g.rot_quat, g_rot, s.rot_m, s.rot_v,
                           cfg.lr_rot, cfg.bias_correction1, cfg.bias_correction2);

    normalizeGaussian3D(g);

    g.opacity = adam_scalar(g.opacity, g_op, s.opacity_m, s.opacity_v,
                            cfg.lr_opacity, cfg.bias_correction1, cfg.bias_correction2);

    g.sh0 = adam_vec3(g.sh0, g_color, s.color_m, s.color_v,
                              cfg.lr_color, cfg.bias_correction1, cfg.bias_correction2);

    gaussians[idx] = g;
    adam_state[idx] = storeAdamMoments(s, adamRoundingSeed(idx, cfg.step));

    if (any(g_mean != float3(0.0)))
    {
        grad_mean_x[idx] = GradScalar(0); grad_mean_y[idx] = GradScalar(0); grad_mean_z[idx] = GradScalar(0);
    }
    if (any(g_log_scale != float3(0.0)))
    {
        grad_log_scale_x[idx] = GradScalar(0); grad_log_scale_y[idx] = GradScalar(0); grad_log_scale_z[idx] = GradScalar(0);
    }
    if (any(g_rot != float4(0.0)))
    {
        grad_rot_w[idx] = GradScalar(0); grad_rot_x[idx] = GradScalar(0); grad_rot_y[idx] = GradScalar(0); grad_rot_z[idx] = GradScalar(0);
    }
    if (g_op != 0.0)
        grad_opacity[idx] = GradScalar(0);
    if (any(g_color != float3(0.0)))
    {
        grad_color_r[idx] = GradScalar(0); grad_color_g[idx] = GradScalar(0); grad_color_b[idx] = GradScalar(0);
    }
}

#include "primitives/gaussian/sh_layout.slang"
#include "../common/half_pack.slang"

// Guards only a literal zero: SH-rest v is ~1e-12 (g ~ 1e-6), so a larger floor would crush the step.
static const float SH_V_FLOOR = 1e-15;

[shader("compute")]
[numthreads(256, 1, 1)]
void adamStepRest(
    uint3 dtid: SV_DispatchThreadID,
    RWStructuredBuffer<uint> features_rest,
    StructuredBuffer<uint> features_rest_grad,
    RWStructuredBuffer<uint> adam_rest,
    StructuredBuffer<uint> num_gaussians_counter,
    StructuredBuffer<uint> cluster_visible,
    uniform float lr_rest,
    uniform float bias_correction1,
    uniform float bias_correction2,
    uniform float pixels_per_step,
    uniform uint rest_floats,
    uniform uint active_rest_floats,
    uniform uint max_splats,
    uniform uint step)
{
    uint idx = dtid.x;
    if (idx >= num_gaussians_counter[0]) return;
    if (cluster_visible[idx / CLUSTER_SIZE] == 0u) return;

    float inv_pixels = 1.0 / max(pixels_per_step, 1.0);
    uint rng = adamRoundingSeed(idx, step);
    uint active = min(active_rest_floats, rest_floats);
    uint fu = halfFeatUints(active);     // only the active coeff uints
    uint pbase = idx * halfFeatUints(rest_floats);  // packed features stride

    for (uint u = 0u; u < fu; ++u)
    {
        float2 g_pair = unpackHalf2x(features_rest_grad[pbase + u]);
        float2 p_pair = unpackHalf2x(features_rest[pbase + u]);
        float p_out[2] = { p_pair.x, p_pair.y };
        [unroll]
        for (uint lane = 0u; lane < 2u; ++lane)
        {
            uint s = u * 2u + lane;
            if (s >= active) break;
            float g = ((lane == 0u) ? g_pair.x : g_pair.y) * inv_pixels;
            if (isnan(g) || isinf(g)) continue;
            float2 mv = loadRestMoments(adam_rest, s, idx, max_splats);
            float m = mv.x;
            float v = mv.y;
            p_out[lane] = adam_scalar((lane == 0u) ? p_pair.x : p_pair.y, g, m, v, lr_rest, bias_correction1, bias_correction2);
            storeRestMoments(adam_rest, s, idx, max_splats, float2(m, max(v, SH_V_FLOOR)), rng);
        }
        features_rest[pbase + u] = packHalf2x(p_out[0], p_out[1]);
    }
}


