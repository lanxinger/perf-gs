// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Shared type definitions.

#ifndef TRAINER_TYPES_SLANG
#define TRAINER_TYPES_SLANG

#include "camera.slang"
#include "fixed_point.slang"
#include "primitives/gaussian/types.slang"

#ifdef TRAINER_METAL_BACKEND
// MSL aligns float3 to 16 bytes (sizeof(float3) == 16), which would bloat Projected2DGaussian
// from 28 -> 48 B and BlendData from 40 -> 80 B. Pack as scalar floats with Slang properties
// so Metal matches Vulkan/D3D12 memory footprint and cuts forward/backward bandwidth in half.
struct Projected2DGaussian
{
    float _mean_x, _mean_y;
    float _cov2d_x, _cov2d_y, _cov2d_z;
    float depth_vs;
    float opacity_compensation;

    property float2 mean
    {
        get { return float2(_mean_x, _mean_y); }
        set { _mean_x = newValue.x; _mean_y = newValue.y; }
    }
    property float3 cov2d
    {
        get { return float3(_cov2d_x, _cov2d_y, _cov2d_z); }
        set { _cov2d_x = newValue.x; _cov2d_y = newValue.y; _cov2d_z = newValue.z; }
    }
};

struct BlendData
{
    float _mean_x, _mean_y;
    float _conic_x, _conic_y, _conic_z;
    float amplitude;
    float _color_r, _color_g, _color_b;
    float opacity;

    property float2 mean
    {
        get { return float2(_mean_x, _mean_y); }
        set { _mean_x = newValue.x; _mean_y = newValue.y; }
    }
    property float3 conic
    {
        get { return float3(_conic_x, _conic_y, _conic_z); }
        set { _conic_x = newValue.x; _conic_y = newValue.y; _conic_z = newValue.z; }
    }
    property float3 color
    {
        get { return float3(_color_r, _color_g, _color_b); }
        set { _color_r = newValue.x; _color_g = newValue.y; _color_b = newValue.z; }
    }
};
#else
struct Projected2DGaussian
{
    float2 mean;            // screen-space center (in pixels)
    float3 cov2d;           // 2x2 covariance packed as (cov_00, cov_01, cov_11)
    float depth_vs;         // view-space depth (positive-z is forward in our convention)
    float opacity_compensation; // Mip-Splatting alpha factor coef3d*coef2d (1.0 when off)
};

// Per-view splat record the blend passes gather, precomputed once by bin_count (and sh_forward for
// color) so the per-pixel loops evaluate only a quadratic form and one exp.
struct BlendData
{
    float2 mean;            // == Projected2DGaussian.mean
    float3 conic;           // conicFromCov(Projected2DGaussian.cov2d)
    float  amplitude;       // splatAmplitude(opacity logit, Mip compensation)
    float3 color;           // this view's color: shDC(sh0), or the SH evaluation
    float  opacity;         // == Gaussian3D.opacity (pre-sigmoid logit); shapes opaque_core only
};
#endif

#endif // TRAINER_TYPES_SLANG
