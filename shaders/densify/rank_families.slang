// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

#include "../common/fixed_point.slang"

// The first 256 counters hold requested slots. The next 256 flag counter overflow.
static const uint BINS = 256u;
static const uint CUTOFF = 2u * BINS, REMAINING = CUTOFF + 1u;

// Eight buckets per power of two keep ranking inexpensive at a densification tick.
uint priorityBin(float score)
{
    return uint(clamp(int(asuint(score) >> 20u) - (127 - 16) * 8, 0, int(BINS) - 1));
}

bool claimCutoffSlots(RWStructuredBuffer<uint> histogram, uint cost)
{
    uint available = histogram[REMAINING];
    while (available >= cost)
    {
        uint observed;
        InterlockedCompareExchange(histogram[REMAINING], available, available - cost, observed);
        if (observed == available) return true;
        available = observed;
    }
    return false;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void rankFamilies(uint3 tid : SV_DispatchThreadID, StructuredBuffer<uint> pre_count,
    RWStructuredBuffer<uint> histogram, RWStructuredBuffer<uint> action,
    RWStructuredBuffer<uint> factor_x, RWStructuredBuffer<uint> factor_y, RWStructuredBuffer<uint> factor_z,
    StructuredBuffer<uint> eta_x, StructuredBuffer<uint> eta_y, StructuredBuffer<uint> eta_z,
    StructuredBuffer<uint> views, ReadGradBuffer gradient,
    uniform float gradient_scale, uniform uint capacity, uniform uint apply)
{
    uint i = tid.x;
    if (i >= min(capacity, pre_count[0])) return;
    uint act = action[i];
    if (act != 1u && act != 2u) return;
    uint3 factors = act == 2u ? uint3(2u, 1u, 1u) : uint3(factor_x[i], factor_y[i], factor_z[i]);
    uint3 eta = uint3(eta_x[i], eta_y[i], eta_z[i]);
    float evidence = fromGrad(gradient[i]) / max(float(views[i]), 1.0) / gradient_scale;
    uint3 granted = uint3(1u);
    uint size = 1u;
    for (uint level = 0u; level < 3u; ++level)
    {
        // Each upgrade adds one complete axis, strongest structure signal first.
        uint axis = 3u;
        for (uint a = 0u; a < 3u; ++a)
            if (factors[a] > 1u && (axis == 3u || eta[a] > eta[min(axis, 2u)])) axis = a;
        if (axis == 3u || factors[min(axis, 2u)] > capacity / size) break;
        uint next_size = size * factors[axis];
        uint cost = next_size - size;
        uint bin = priorityBin(evidence / float(next_size));
        if (apply == 0u)
        {
            uint old;
            InterlockedAdd(histogram[bin], cost, old);
            if (old > 0xffffffffu - cost) InterlockedOr(histogram[BINS + bin], 1u);
        }
        else
        {
            uint cutoff = histogram[CUTOFF];
            if (bin < cutoff) break;
            if (bin == cutoff)
                if (!claimCutoffSlots(histogram, cost)) break;
            granted[axis] = factors[axis];
        }
        size = next_size;
        factors[axis] = 1u;
    }
    if (apply != 0u)
    {
        if (size == 1u) action[i] = 0u;
        if (act == 1u)
        {
            factor_x[i] = granted.x;
            factor_y[i] = granted.y;
            factor_z[i] = granted.z;
        }
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void chooseCutoff(RWStructuredBuffer<uint> histogram, StructuredBuffer<uint> pre_count, uniform uint capacity)
{
    uint slots = capacity - min(capacity, pre_count[0]);
    uint bin = BINS - 1u;
    while (bin > 0u && histogram[BINS + bin] == 0u && histogram[bin] <= slots)
    {
        slots -= histogram[bin];
        --bin;
    }
    histogram[CUTOFF] = bin;
    histogram[REMAINING] = slots;
}
