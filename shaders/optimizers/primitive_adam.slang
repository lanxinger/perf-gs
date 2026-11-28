// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

#ifndef TRAINER_OPTIMIZER_PRIMITIVE_ADAM_SLANG
#define TRAINER_OPTIMIZER_PRIMITIVE_ADAM_SLANG

#include "../common/fixed_point.slang"

// Generic Adam over the flat scalars a primitive exposes through IPrimitiveParameterStorage<N>;
// a new primitive only supplies a storage conformer, never touching the Adam math.

// `data[c]` is channel c's value, packed through ChannelCursor.
struct PrimitiveValues<let N : int>
{
    float data[N];
}

// Sequential field reader/writer: channel indices follow from the order fields are listed.
struct ChannelCursor<let N : int>
{
    uint offset;

    [mutating] void write1(inout float data[N], float v)
    {
        data[offset] = v;
        offset += 1u;
    }
    [mutating] void write2(inout float data[N], float2 v)
    {
        data[offset + 0u] = v.x;
        data[offset + 1u] = v.y;
        offset += 2u;
    }
    [mutating] void write3(inout float data[N], float3 v)
    {
        data[offset + 0u] = v.x;
        data[offset + 1u] = v.y;
        data[offset + 2u] = v.z;
        offset += 3u;
    }
    [mutating] void write4(inout float data[N], float4 v)
    {
        data[offset + 0u] = v.x;
        data[offset + 1u] = v.y;
        data[offset + 2u] = v.z;
        data[offset + 3u] = v.w;
        offset += 4u;
    }

    [mutating] float read1(inout float data[N])
    {
        float v = data[offset];
        offset += 1u;
        return v;
    }
    [mutating] float2 read2(inout float data[N])
    {
        float2 v = float2(data[offset + 0u], data[offset + 1u]);
        offset += 2u;
        return v;
    }
    [mutating] float3 read3(inout float data[N])
    {
        float3 v = float3(data[offset + 0u], data[offset + 1u], data[offset + 2u]);
        offset += 3u;
        return v;
    }
    [mutating] float4 read4(inout float data[N])
    {
        float4 v = float4(data[offset + 0u], data[offset + 1u],
                          data[offset + 2u], data[offset + 3u]);
        offset += 4u;
        return v;
    }
}

// Conformers own the concrete buffers and must preserve non-optimized fields across load/store.
interface IPrimitiveParameterStorage<let N : int>
{
    PrimitiveValues<N> loadValues(uint index);
    void storeValues(uint index, PrimitiveValues<N> values);
}

// Bias-correction denominators, precomputed on the host from the step count.
struct AdamConfig
{
    float bias_correction1;
    float bias_correction2;
}

// The host expands per-field learning rates into this per-channel vector.
struct ChannelLearningRates<let N : int>
{
    float rates[N];
}

// Channel-major planes indexed [channel * channel_stride + index]; gradients are int64
// fixed-point, atomically summed by the backward pass.
struct AdamPlanes<let N : int>
{
    RWStructuredBuffer<fixed_t> gradients;
    RWStructuredBuffer<float>   moment1;
    RWStructuredBuffer<float>   moment2;
    uint channel_stride;

    uint address(uint channel, uint index)
    {
        return channel * channel_stride + index;
    }
    float gradient(uint channel, uint index, float inv_pixels)
    {
        return fromFixed(gradients[address(channel, index)]) * inv_pixels;
    }
}

static const float ADAM_BETA1   = 0.9;
static const float ADAM_BETA2   = 0.999;
static const float ADAM_EPSILON = 1e-8;

// Adam step for one primitive that also clears its gradients; a NaN/Inf in any channel skips it.
struct AdamPass
{
    static void step<let N : int, P : IPrimitiveParameterStorage<N>>(
        P storage, AdamPlanes<N> planes, ChannelLearningRates<N> lrs,
        uint index, AdamConfig cfg, float inv_pixels)
    {
        [unroll]
        for (uint c = 0u; c < uint(N); ++c)
        {
            float g = planes.gradient(c, index, inv_pixels);
            if (isnan(g) || isinf(g)) return;
        }

        PrimitiveValues<N> values = storage.loadValues(index);

        [unroll]
        for (uint c = 0u; c < uint(N); ++c)
        {
            uint addr = planes.address(c, index);
            float g = fromFixed(planes.gradients[addr]) * inv_pixels;

            float m = planes.moment1[addr];
            float v = planes.moment2[addr];
            m = ADAM_BETA1 * m + (1.0 - ADAM_BETA1) * g;
            v = ADAM_BETA2 * v + (1.0 - ADAM_BETA2) * g * g;

            float m_hat = m / cfg.bias_correction1;
            float v_hat = v / cfg.bias_correction2;
            values.data[c] = values.data[c]
                           - lrs.rates[c] * m_hat / (sqrt(v_hat) + ADAM_EPSILON);

            planes.moment1[addr] = m;
            planes.moment2[addr] = v;
            planes.gradients[addr] = fixed_t(0);
        }

        storage.storeValues(index, values);
    }
}

#endif // TRAINER_OPTIMIZER_PRIMITIVE_ADAM_SLANG
