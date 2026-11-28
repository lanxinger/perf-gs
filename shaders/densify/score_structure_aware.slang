// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

#include "../common/types.slang"
#include "../common/math.slang"
#include "../primitives/gaussian/project.slang"

static const float ETA_HIGH = 1.0;    // eta > this ⟹ under-resolved (split vote)
static const float ETA_LOW  = 0.1;    // eta < this ⟹ over-resolved (prune vote)
static const float EPS = 1e-8;
static const float COV_DILATION = 1e-8;

// Wang-hash uniform[0,1), as in split_and_clone.slang, so sampling is deterministic per (gaussian, view).
uint _wang_hash(uint x)
{
    x = (x ^ 61u) ^ (x >> 16u);
    x *= 9u;
    x = x ^ (x >> 4u);
    x *= 0x27d4eb2du;
    x = x ^ (x >> 15u);
    return x;
}
float _rand01(uint seed) { return float(_wang_hash(seed)) / 4294967295.0; }

float2 _randn2(uint seed)
{
    float u1 = max(_rand01(seed), 1e-7);
    float u2 = _rand01(seed + 1u);
    float r = sqrt(-2.0 * log(u1));
    float a = 6.28318530718 * u2;
    return float2(r * cos(a), r * sin(a));
}

float3 _sample_structure_bilinear(Texture2D<float4> tex, float2 p, uint2 dim)
{
    float2 pf = floor(p);
    int2 i0 = int2(pf);
    float2 f = p - pf;                 // bilinear weights in [0,1)
    int W = int(dim.x), H = int(dim.y);

    float3 acc = float3(0.0, 0.0, 0.0);
    // 4 taps; each out-of-bounds tap contributes 0 (zero padding).
    [unroll]
    for (int dy = 0; dy < 2; ++dy)
    {
        [unroll]
        for (int dx = 0; dx < 2; ++dx)
        {
            int tx = i0.x + dx;
            int ty = i0.y + dy;
            if (tx < 0 || tx >= W || ty < 0 || ty >= H) continue;
            float wx = (dx == 0) ? (1.0 - f.x) : f.x;
            float wy = (dy == 0) ? (1.0 - f.y) : f.y;
            acc += (wx * wy) * tex[int2(tx, ty)].xyz;
        }
    }
    return acc;
}

// Top eigenvalue of the packed symmetric 2x2 tensor (Sxx, Sxy, Syy).
float _principal_eigenvalue(float3 s)
{
    float a = s.x, b = s.y, c = s.z;
    float mid = 0.5 * (a + c);
    float disc = sqrt(max((0.5 * (a - c)) * (0.5 * (a - c)) + b * b, 0.0));
    return mid + disc;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void scoreStructureAware(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<Gaussian3D> gaussians,
    StructuredBuffer<uint> num_gaussians_counter,
    StructuredBuffer<CameraData> camera,
    Texture2D<float4> structure_tensor,
    StructuredBuffer<uint> tile_count_per_gaussian,
    ReadGradBuffer grad_2d_x_iter,
    ReadGradBuffer grad_2d_y_iter,
    RWStructuredBuffer<uint> eta_n_high, RWStructuredBuffer<uint> eta_n_low,
    RWStructuredBuffer<uint> eta_max_x, RWStructuredBuffer<uint> eta_max_y, RWStructuredBuffer<uint> eta_max_z,
    // Count of views in which this gaussian was on-screen / valid.
    RWStructuredBuffer<uint> eta_n_total,
    uniform float opacity_threshold,  // SADGS active-visibility test: sigmoid(opacity) > this (0.05)
    uniform float per_view_grad_thresh,
    uniform uint pass_idx,       // view index, for jitter decorrelation across views
    uniform uint max_splats)
{
    uint gaussian_idx = dtid.x;
    if (gaussian_idx >= max_splats) return;
    if (gaussian_idx >= num_gaussians_counter[0]) return;

    if (tile_count_per_gaussian[gaussian_idx] == 0u) return;

    Gaussian3D g = gaussians[gaussian_idx];
    if (1.0 / (1.0 + exp(-g.opacity)) <= opacity_threshold) return;

    CameraData cam = camera[0];
    float2 frame_dim = cam.image_size;

    if (per_view_grad_thresh > 0.0)
    {
        float gx = fromGrad(grad_2d_x_iter[gaussian_idx]) * (frame_dim.x * 0.5);
        float gy = fromGrad(grad_2d_y_iter[gaussian_idx]) * (frame_dim.y * 0.5);
        if (sqrt(gx * gx + gy * gy) < per_view_grad_thresh) return;
    }

    float3 mean_vs;
    float2 mean_px;
    float clip_w;
    projectMeanToPixel(g.mean, cam.view_matrix, cam.proj_matrix, frame_dim, mean_vs, mean_px, clip_w);
    if (mean_vs.z <= 0.0) return;   // behind camera; no observation this view
    if (clip_w <= 0.0) return;      // behind clip plane

    float3x3 J = projectionJacobian(mean_vs, cam.proj_matrix, frame_dim);

    // A = J W R diag(scale). Its columns are the projected axis vectors v_k.
    float3x3 W = float3x3(cam.view_matrix);
    float3x3 R = matrixFromQuaternion(g.rot_quat);
    float3 scale = exp(g.log_scale);
    float3x3 RS = float3x3(
        R[0][0] * scale.x, R[0][1] * scale.y, R[0][2] * scale.z,
        R[1][0] * scale.x, R[1][1] * scale.y, R[1][2] * scale.z,
        R[2][0] * scale.x, R[2][1] * scale.y, R[2][2] * scale.z
    );
    float3x3 A = mul(J, mul(W, RS));

    float len_vx = sqrt(A[0][0] * A[0][0] + A[1][0] * A[1][0] + COV_DILATION);
    float len_vy = sqrt(A[0][1] * A[0][1] + A[1][1] * A[1][1] + COV_DILATION);
    float len_vz = sqrt(A[0][2] * A[0][2] + A[1][2] * A[1][2] + COV_DILATION);

    float cov_xx = A[0][0] * A[0][0] + A[0][1] * A[0][1] + A[0][2] * A[0][2] + 0.3;
    float cov_xy = A[0][0] * A[1][0] + A[0][1] * A[1][1] + A[0][2] * A[1][2];
    float cov_yy = A[1][0] * A[1][0] + A[1][1] * A[1][1] + A[1][2] * A[1][2] + 0.3;

    // Cholesky of the 2D covariance (as in SADGS).
    float L11 = sqrt(max(cov_xx, 1e-6));
    float L21 = cov_xy / L11;
    float L22 = sqrt(max(cov_yy - L21 * L21, 1e-6));

    uint seed = gaussian_idx * 2654435761u + pass_idx * 40503u;
    float2 eps = _randn2(seed);
    float2 sample_px = mean_px + float2(L11 * eps.x, L21 * eps.x + L22 * eps.y);

    uint2 dim;
    structure_tensor.GetDimensions(dim.x, dim.y);
    float3 sbar = _sample_structure_bilinear(structure_tensor, sample_px, dim);

    float lambda1 = _principal_eigenvalue(sbar);
    float inv_lambda_min = sqrt(max(lambda1, 0.0)) + EPS;  // 1 / Lambda_min

    float eta_x = len_vx * inv_lambda_min;
    float eta_y = len_vy * inv_lambda_min;
    float eta_z = len_vz * inv_lambda_min;
    float eta_max_scalar = max(eta_x, max(eta_y, eta_z));

    record_observations(
        gaussian_idx, eta_x, eta_y, eta_z, eta_max_scalar,
        eta_n_high, eta_n_low, eta_max_x, eta_max_y, eta_max_z, eta_n_total
    );
    
}

void record_observations(uint gaussian_idx, float eta_x, float eta_y, float eta_z, float eta_max_scalar,
    RWStructuredBuffer<uint> eta_n_high, RWStructuredBuffer<uint> eta_n_low,
    RWStructuredBuffer<uint> eta_max_x, RWStructuredBuffer<uint> eta_max_y, RWStructuredBuffer<uint> eta_max_z,
    RWStructuredBuffer<uint> eta_n_total
)
{
    InterlockedAdd(eta_n_total[gaussian_idx], 1u);

    if (eta_max_scalar > ETA_HIGH) InterlockedAdd(eta_n_high[gaussian_idx], 1u);
    else if (eta_max_scalar <= ETA_LOW) InterlockedAdd(eta_n_low[gaussian_idx], 1u);

    // Per-axis max over views feeds the anisotropic split decision.
    InterlockedMax(eta_max_x[gaussian_idx], uint(max(eta_x, 0.0) * ETA_FIXED_SCALE));
    InterlockedMax(eta_max_y[gaussian_idx], uint(max(eta_y, 0.0) * ETA_FIXED_SCALE));
    InterlockedMax(eta_max_z[gaussian_idx], uint(max(eta_z, 0.0) * ETA_FIXED_SCALE));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void clearStructureScores(
    uint3 dtid: SV_DispatchThreadID,
    RWStructuredBuffer<uint> eta_n_high, RWStructuredBuffer<uint> eta_n_low,
    RWStructuredBuffer<uint> eta_max_x, RWStructuredBuffer<uint> eta_max_y, RWStructuredBuffer<uint> eta_max_z,
    RWStructuredBuffer<uint> eta_n_total,
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;
    eta_n_high[i] = 0u; eta_n_low[i] = 0u;
    eta_max_x[i] = 0u; eta_max_y[i] = 0u; eta_max_z[i] = 0u;
    eta_n_total[i] = 0u;
}
