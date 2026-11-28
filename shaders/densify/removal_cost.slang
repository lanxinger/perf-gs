// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

#include "primitives/gaussian/blend.slang"
#include "rasterize/tile_binned/blend_forward.slang"
#include "common/fixed_point.slang"

#ifdef FIXED_POINT_INT32
typedef RWStructuredBuffer<fixed_t> RemovalScoreBuffer;
static const int REMOVAL_COST_MAX = 2147483647;
// Largest float32 strictly below 2^31 so float->int conversion never overflows.
static const float REMOVAL_COST_MAX_FLOAT = 2147483520.0;

// Saturate at INT32_MAX: pruning only ranks the lowest-scoring splats, so huge scores must never wrap.
void addRemovalScore(RemovalScoreBuffer scores, uint id, float scaled_energy)
{
    int add = (scaled_energy >= REMOVAL_COST_MAX_FLOAT)
        ? REMOVAL_COST_MAX
        : (scaled_energy > 1.0 ? int(scaled_energy) : 1);
    int cur = scores[id];
    while (cur < REMOVAL_COST_MAX)
    {
        int next = (cur > REMOVAL_COST_MAX - add) ? REMOVAL_COST_MAX : cur + add;
        int prev;
        InterlockedCompareExchange(scores[id], cur, next, prev);
        if (prev == cur) return;
        cur = prev;
    }
}
#else
typedef AtomicI64Buffer RemovalScoreBuffer;
void addRemovalScore(RemovalScoreBuffer scores, uint id, float scaled_energy)
{
    int64_t energy = int64_t(scaled_energy);
    // One fixed-point unit distinguishes measured zero contrast from never sampled.
    atomicAddI64(scores, id, max(energy, int64_t(1)));
}
#endif

// Replay one pixel per 4x4 block up to the actual forward pass's early stop.
[shader("compute")]
[numthreads(8, 8, 1)]
void accumulateRemovalCost(uint3 tid : SV_DispatchThreadID,
    StructuredBuffer<CameraData> camera, StructuredBuffer<BlendData> blend_data,
    StructuredBuffer<uint> tile_start, StructuredBuffer<uint> sorted_ids,
    RWTexture2D<uint> processed, RWTexture2D<float4> image, RemovalScoreBuffer scores,
    uniform uint phase, uniform float hardness)
{
    uint2 pixel = tid.xy * 4u + uint2(phase & 3u, phase >> 2u);
    uint2 dims = uint2(camera[0].image_size);
    if (any(pixel >= dims)) return;
    uint tile = pixel.x / TILE_WIDTH + (pixel.y / TILE_HEIGHT) * ((dims.x + TILE_WIDTH - 1u) / TILE_WIDTH);
    uint offset = tile_start[tile], stop = processed[pixel];
    Compositor comp;
    comp.hardness = hardness;
    float transmission = 1;
    float gain = camera[0].lensGain(pixel);
    float3 remaining = image[pixel].rgb;
    for (uint j = 0; j < stop; ++j)
    {
        uint id = sorted_ids[offset + j];
        GaussianPrimitive splat = GaussianPrimitive.make(blend_data[id]);
        float alpha = min(splat.coverage(float2(pixel) + 0.5), BLEND_ALPHA_MAX);
        if (alpha < BLEND_ALPHA_MIN) continue;
        float t = comp.transmittance(alpha);
        float weight = comp.weight(alpha);
        float3 source = splat.color() * (gain * transmission);
        // remaining = weight * source + t * background, in original pixel units.
        // Omitting this splat replaces remaining with background.
        // At the last entry there is no remaining background in this replay. This
        // form also stays finite when a hard, opaque splat drives t to zero.
        float3 delta = j + 1 == stop ? weight * source : (weight / t) * (source - remaining);
        addRemovalScore(scores, id, dot(delta, delta) * REMOVAL_COST_SCALE);
        remaining -= weight * source;
        transmission *= t;
    }
}
