// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Complete second-primitive proof: shared forward blend, autodiff backward,
// and generic Adam update. Deliberately imports no Gaussian file or type.

#include "blend.slang"
#include "parameters.slang"
#include "../../rasterize/tile_binned/blend_forward.slang"

[shader("compute")]
[numthreads(64, 1, 1)]
void diskForward(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<Disk2D> primitives,
    StructuredBuffer<uint> sorted_primitive_ids,
    StructuredBuffer<float2> pixels,
    RWStructuredBuffer<uint2> checkpoints,
    RWStructuredBuffer<float4> outputs,
    RWStructuredBuffer<uint> processed_out,
    uniform uint primitive_count,
    uniform uint pixel_count,
    uniform float extinction_hardness)
{
    uint index = dtid.x;
    if (index >= pixel_count) return;

    DiskSource source;
    source.primitives = primitives;
    float2 pixel_center[1] = { pixels[index] };
    bool valid_pixel[1] = { true };
    uint pixel_slot[1] = { index };
    float3 color[1];
    float transmittance[1];
    uint processed[1];
    blendForward<DiskSource, 1>(
        source, 0u, primitive_count, pixel_center, valid_pixel,
        pixel_slot, 0u, sorted_primitive_ids, checkpoints, extinction_hardness,
        color, transmittance, processed);
    outputs[index] = float4(color[0], 1.0 - transmittance[0]);
    processed_out[index] = processed[0];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void diskBackward(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<Disk2D> primitives,
    StructuredBuffer<float2> pixels,
    StructuredBuffer<float> coverage_seeds,
    StructuredBuffer<float3> color_seeds,
    RWStructuredBuffer<float> gradients,
    uniform uint count)
{
    uint index = dtid.x;
    if (index >= count) return;

    DiskPrimitive primitive = DiskPrimitive.make(primitives[index]);
    var differential = diffPair(primitive);
    bwd_diff(primitiveLinearObjective<DiskPrimitive>)(
        differential, pixels[index], coverage_seeds[index], color_seeds[index], 1.0);

    uint base = index * DiskChannel::Count;
    gradients[base + DiskChannel::Center + 0u] = differential.d.center.x;
    gradients[base + DiskChannel::Center + 1u] = differential.d.center.y;
    gradients[base + DiskChannel::LogRadius] = differential.d.log_radius;
    gradients[base + DiskChannel::Opacity] = differential.d.opacity;
    gradients[base + DiskChannel::Color + 0u] = differential.d.color_value.x;
    gradients[base + DiskChannel::Color + 1u] = differential.d.color_value.y;
    gradients[base + DiskChannel::Color + 2u] = differential.d.color_value.z;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void diskAdam(
    uint3 dtid: SV_DispatchThreadID,
    RWStructuredBuffer<Disk2D> primitives,
    RWStructuredBuffer<fixed_t> gradients,
    RWStructuredBuffer<float> moment1,
    RWStructuredBuffer<float> moment2,
    StructuredBuffer<uint> primitive_count,
    uniform ChannelLearningRates<DiskChannel::Count> learning_rates,
    uniform AdamConfig cfg,
    uniform float inverse_pixels,
    uniform uint channel_stride)
{
    uint index = dtid.x;
    if (index >= primitive_count[0]) return;

    DiskParameterStorage storage;
    storage.primitives = primitives;
    AdamPlanes<DiskChannel::Count> planes;
    planes.gradients = gradients;
    planes.moment1 = moment1;
    planes.moment2 = moment2;
    planes.channel_stride = channel_stride;
    AdamPass.step(storage, planes, learning_rates, index, cfg, inverse_pixels);
}
