// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Concrete Gaussian specialization of the generic Adam pass.

#include "parameters.slang"

#include "common/execution.slang"

[shader("compute")]
[numthreads(256, 1, 1)]
void adamStep(
    uint3 dtid: SV_DispatchThreadID,
    RWStructuredBuffer<Gaussian3D> primitives,
    RWStructuredBuffer<fixed_t> gradients,
    RWStructuredBuffer<float> moment1,
    RWStructuredBuffer<float> moment2,
    StructuredBuffer<uint> num_primitives_counter,
    StructuredBuffer<uint> cluster_visible,
    uniform ChannelLearningRates<GaussianChannel::Count> learning_rates,
    uniform AdamConfig cfg,
    uniform float pixels_per_step,
    uniform uint channel_stride)
{
    uint index = dtid.x;
    if (index >= num_primitives_counter[0]) return;
    if (cluster_visible[index / CLUSTER_SIZE] == 0u) return;

    GaussianParameterStorage primitive_storage;
    primitive_storage.primitives = primitives;

    AdamPlanes<GaussianChannel::Count> optimizer_storage;
    optimizer_storage.gradients = gradients;
    optimizer_storage.moment1 = moment1;
    optimizer_storage.moment2 = moment2;
    optimizer_storage.channel_stride = channel_stride;

    float inv_pixels = 1.0 / max(pixels_per_step, 1.0);
    AdamPass.step(primitive_storage, optimizer_storage, learning_rates,
                  index, cfg, inv_pixels);
}
