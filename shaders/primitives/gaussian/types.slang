// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Gaussian storage and optimizer-state structs. Python reads their layout by reflection.

#ifndef TRAINER_GAUSSIAN_TYPES
#define TRAINER_GAUSSIAN_TYPES

#ifdef TRAINER_METAL_BACKEND
// Tightly packed 56 B (14 floats) on Metal, avoiding MSL's 16-byte float3 alignment
// (80 B -> 56 B, matching Vulkan/D3D12 scalar layout and saving 30% Gaussian bandwidth).
struct Gaussian3D
{
    float _mean_x, _mean_y, _mean_z;
    float _log_scale_x, _log_scale_y, _log_scale_z;
    float _rot_w, _rot_x, _rot_y, _rot_z;
    float opacity;     // pre-sigmoid logit
    float _sh0_r, _sh0_g, _sh0_b;

    property float3 mean
    {
        get { return float3(_mean_x, _mean_y, _mean_z); }
        set { _mean_x = newValue.x; _mean_y = newValue.y; _mean_z = newValue.z; }
    }
    property float3 log_scale
    {
        get { return float3(_log_scale_x, _log_scale_y, _log_scale_z); }
        set { _log_scale_x = newValue.x; _log_scale_y = newValue.y; _log_scale_z = newValue.z; }
    }
    property float4 rot_quat
    {
        get { return float4(_rot_w, _rot_x, _rot_y, _rot_z); }
        set { _rot_w = newValue.x; _rot_x = newValue.y; _rot_y = newValue.z; _rot_z = newValue.w; }
    }
    property float3 sh0
    {
        get { return float3(_sh0_r, _sh0_g, _sh0_b); }
        set { _sh0_r = newValue.x; _sh0_g = newValue.y; _sh0_b = newValue.z; }
    }
};
#else
struct Gaussian3D
{
    float3 mean;
    float3 log_scale;
    float4 rot_quat;   // (w, x, y, z)
    float opacity;     // pre-sigmoid logit
    float3 sh0;        // linear RGB = 0.5 + C0 * sh0
};
#endif

// Offsets of each field in the flat gradient/optimizer channel order.
namespace GaussianChannel
{
    static const uint Mean = 0u;
    static const uint LogScale = 3u;
    static const uint Rotation = 6u;
    static const uint Opacity = 10u;
    static const uint Color = 11u;
    static const int Count = 14;
}

void normalizeGaussian3D(inout Gaussian3D primitive)
{
    primitive.rot_quat = normalize(primitive.rot_quat);
}

// Adam (m, v) per optimized field, in Gaussian3D field order.
struct AdamMoments
{
    float3 mean_m;
    float3 mean_v;
    float3 log_scale_m;
    float3 log_scale_v;
    float4 rot_m;
    float4 rot_v;
    float opacity_m;
    float opacity_v;
    float3 color_m;
    float3 color_v;
};

#include "../../common/adam_precision.slang"

#if defined(ADAM_MOMENTS_BF16)
#ifdef TRAINER_METAL_BACKEND
// Tightly packed 56 B (14 uints) on Metal, avoiding MSL's 16-byte uint3/uint4 padding.
struct AdamState
{
    uint _mean_mv_x, _mean_mv_y, _mean_mv_z;
    uint _log_scale_mv_x, _log_scale_mv_y, _log_scale_mv_z;
    uint _rot_mv_w, _rot_mv_x, _rot_mv_y, _rot_mv_z;
    uint opacity_mv;
    uint _color_mv_r, _color_mv_g, _color_mv_b;
};

AdamMoments loadAdamMoments(AdamState s)
{
    uint3 mean_mv = uint3(s._mean_mv_x, s._mean_mv_y, s._mean_mv_z);
    uint3 log_scale_mv = uint3(s._log_scale_mv_x, s._log_scale_mv_y, s._log_scale_mv_z);
    uint4 rot_mv = uint4(s._rot_mv_w, s._rot_mv_x, s._rot_mv_y, s._rot_mv_z);
    uint3 color_mv = uint3(s._color_mv_r, s._color_mv_g, s._color_mv_b);
    AdamMoments m;
    m.mean_m = bf16Lo(mean_mv);
    m.mean_v = bf16Hi(mean_mv);
    m.log_scale_m = bf16Lo(log_scale_mv);
    m.log_scale_v = bf16Hi(log_scale_mv);
    m.rot_m = bf16Lo(rot_mv);
    m.rot_v = bf16Hi(rot_mv);
    m.opacity_m = bf16Lo(s.opacity_mv);
    m.opacity_v = bf16Hi(s.opacity_mv);
    m.color_m = bf16Lo(color_mv);
    m.color_v = bf16Hi(color_mv);
    return m;
}

AdamState storeAdamMoments(AdamMoments m, uint seed)
{
    AdamState s;
    uint rng = seed;
    uint3 mean_mv = packBf16(m.mean_m, m.mean_v, rng);
    uint3 log_scale_mv = packBf16(m.log_scale_m, m.log_scale_v, rng);
    uint4 rot_mv = packBf16(m.rot_m, m.rot_v, rng);
    uint3 color_mv = packBf16(m.color_m, m.color_v, rng);
    s._mean_mv_x = mean_mv.x; s._mean_mv_y = mean_mv.y; s._mean_mv_z = mean_mv.z;
    s._log_scale_mv_x = log_scale_mv.x; s._log_scale_mv_y = log_scale_mv.y; s._log_scale_mv_z = log_scale_mv.z;
    s._rot_mv_w = rot_mv.x; s._rot_mv_x = rot_mv.y; s._rot_mv_y = rot_mv.z; s._rot_mv_z = rot_mv.w;
    s.opacity_mv = packBf16(m.opacity_m, m.opacity_v, rng);
    s._color_mv_r = color_mv.x; s._color_mv_g = color_mv.y; s._color_mv_b = color_mv.z;
    return s;
}
#else
struct AdamState
{
    uint3 mean_mv;       // bf16 (m, v)
    uint3 log_scale_mv;
    uint4 rot_mv;
    uint opacity_mv;
    uint3 color_mv;
};

AdamMoments loadAdamMoments(AdamState s)
{
    AdamMoments m;
    m.mean_m = bf16Lo(s.mean_mv);
    m.mean_v = bf16Hi(s.mean_mv);
    m.log_scale_m = bf16Lo(s.log_scale_mv);
    m.log_scale_v = bf16Hi(s.log_scale_mv);
    m.rot_m = bf16Lo(s.rot_mv);
    m.rot_v = bf16Hi(s.rot_mv);
    m.opacity_m = bf16Lo(s.opacity_mv);
    m.opacity_v = bf16Hi(s.opacity_mv);
    m.color_m = bf16Lo(s.color_mv);
    m.color_v = bf16Hi(s.color_mv);
    return m;
}

AdamState storeAdamMoments(AdamMoments m, uint seed)
{
    AdamState s;
    uint rng = seed;
    s.mean_mv = packBf16(m.mean_m, m.mean_v, rng);
    s.log_scale_mv = packBf16(m.log_scale_m, m.log_scale_v, rng);
    s.rot_mv = packBf16(m.rot_m, m.rot_v, rng);
    s.opacity_mv = packBf16(m.opacity_m, m.opacity_v, rng);
    s.color_mv = packBf16(m.color_m, m.color_v, rng);
    return s;
}
#endif
#else
#ifdef TRAINER_METAL_BACKEND
// Tightly packed 112 B (28 floats) on Metal, matching Vulkan/D3D12 scalar layout exactly.
struct AdamState
{
    float _mean_m_x, _mean_m_y, _mean_m_z;
    float _mean_v_x, _mean_v_y, _mean_v_z;
    float _log_scale_m_x, _log_scale_m_y, _log_scale_m_z;
    float _log_scale_v_x, _log_scale_v_y, _log_scale_v_z;
    float _rot_m_w, _rot_m_x, _rot_m_y, _rot_m_z;
    float _rot_v_w, _rot_v_x, _rot_v_y, _rot_v_z;
    float opacity_m;
    float opacity_v;
    float _color_m_r, _color_m_g, _color_m_b;
    float _color_v_r, _color_v_g, _color_v_b;
};

AdamMoments loadAdamMoments(AdamState s)
{
    AdamMoments m;
    m.mean_m = float3(s._mean_m_x, s._mean_m_y, s._mean_m_z);
    m.mean_v = float3(s._mean_v_x, s._mean_v_y, s._mean_v_z);
    m.log_scale_m = float3(s._log_scale_m_x, s._log_scale_m_y, s._log_scale_m_z);
    m.log_scale_v = float3(s._log_scale_v_x, s._log_scale_v_y, s._log_scale_v_z);
    m.rot_m = float4(s._rot_m_w, s._rot_m_x, s._rot_m_y, s._rot_m_z);
    m.rot_v = float4(s._rot_v_w, s._rot_v_x, s._rot_v_y, s._rot_v_z);
    m.opacity_m = s.opacity_m;
    m.opacity_v = s.opacity_v;
    m.color_m = float3(s._color_m_r, s._color_m_g, s._color_m_b);
    m.color_v = float3(s._color_v_r, s._color_v_g, s._color_v_b);
    return m;
}

AdamState storeAdamMoments(AdamMoments m, uint seed)
{
    AdamState s;
    s._mean_m_x = m.mean_m.x; s._mean_m_y = m.mean_m.y; s._mean_m_z = m.mean_m.z;
    s._mean_v_x = m.mean_v.x; s._mean_v_y = m.mean_v.y; s._mean_v_z = m.mean_v.z;
    s._log_scale_m_x = m.log_scale_m.x; s._log_scale_m_y = m.log_scale_m.y; s._log_scale_m_z = m.log_scale_m.z;
    s._log_scale_v_x = m.log_scale_v.x; s._log_scale_v_y = m.log_scale_v.y; s._log_scale_v_z = m.log_scale_v.z;
    s._rot_m_w = m.rot_m.x; s._rot_m_x = m.rot_m.y; s._rot_m_y = m.rot_m.z; s._rot_m_z = m.rot_m.w;
    s._rot_v_w = m.rot_v.x; s._rot_v_x = m.rot_v.y; s._rot_v_y = m.rot_v.z; s._rot_v_z = m.rot_v.w;
    s.opacity_m = m.opacity_m;
    s.opacity_v = m.opacity_v;
    s._color_m_r = m.color_m.x; s._color_m_g = m.color_m.y; s._color_m_b = m.color_m.z;
    s._color_v_r = m.color_v.x; s._color_v_g = m.color_v.y; s._color_v_b = m.color_v.z;
    return s;
}
#else
struct AdamState
{
    AdamMoments moments;
};

AdamMoments loadAdamMoments(AdamState s)
{
    return s.moments;
}

AdamState storeAdamMoments(AdamMoments m, uint seed)
{
    AdamState s;
    s.moments = m;
    return s;
}
#endif
#endif

#endif // TRAINER_GAUSSIAN_TYPES
