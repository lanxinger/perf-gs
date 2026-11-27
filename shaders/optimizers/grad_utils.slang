// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Utility kernels for int64 gradient-accumulator buffers + Adam moment state.
#include "../common/types.slang"

[shader("compute")]
[numthreads(256, 1, 1)]
void clearGradBuffer(
    uint3 dtid: SV_DispatchThreadID,
    GradBuffer grad_buffer,
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;
    grad_buffer[i] = GradScalar(0);
}

// `count` is the total uint length; a zero uint unpacks to two 0.0 fp16 halves.
[shader("compute")]
[numthreads(256, 1, 1)]
void clearUintBuffer(
    uint3 dtid: SV_DispatchThreadID,
    RWStructuredBuffer<uint> buffer,
    uniform uint count)
{
    uint i = dtid.x;
    if (i >= count) return;
    buffer[i] = 0u;
}

// Zero-init cast, so new AdamState fields need no edit here.
[shader("compute")]
[numthreads(256, 1, 1)]
void clearAdamState(
    uint3 dtid: SV_DispatchThreadID,
    RWStructuredBuffer<AdamState> adam_state,
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;
    adam_state[i] = (AdamState)0;
}
