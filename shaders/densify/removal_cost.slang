// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

#include "primitives/gaussian/blend.slang"
#include "rasterize/tile_binned/blend_forward.slang"

// Replay one pixel per 4x4 block up to the actual forward pass's early stop.
[shader("compute")]
[numthreads(8, 8, 1)]
void accumulateRemovalCost(uint3 tid : SV_DispatchThreadID,
    StructuredBuffer<CameraData> camera, StructuredBuffer<BlendData> blend_data,
    StructuredBuffer<uint> tile_start, StructuredBuffer<uint> sorted_ids,
    RWTexture2D<uint> processed, RWTexture2D<float4> image, AtomicI64Buffer scores,
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
        int64_t energy = int64_t(dot(delta, delta) * 1073741824.0);
        // One fixed-point unit distinguishes measured zero contrast from never sampled.
        atomicAddI64(scores, id, max(energy, int64_t(1)));
        remaining -= weight * source;
        transmission *= t;
    }
}
