// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

#ifndef TRAINER_FIXED_POINT_SLANG
#define TRAINER_FIXED_POINT_SLANG

// Fixed-point atomic gradient accumulation (int64, no overflow).
static const float FIXED_SCALE = 65536.0;
static const float FIXED_INV_SCALE = 1.0 / FIXED_SCALE;

int64_t toFixed(float value) { return int64_t(value * FIXED_SCALE); }
float fromFixed(int64_t value) { return float(value) * FIXED_INV_SCALE; }

// Metal Shading Language does not expose 64-bit integer atomic_fetch_add, so emulate exact
// two's-complement 64-bit signed InterlockedAdd over the same little-endian int64_t buffer
// using a 32-bit unsigned add + carry propagation into the high word.
#ifdef TRAINER_METAL_BACKEND
typedef RWByteAddressBuffer AtomicI64Buffer;

void atomicAddI64(RWByteAddressBuffer buf, uint index, int64_t delta)
{
    if (delta == int64_t(0)) return;
    uint lo_add = uint(uint64_t(delta) & 0xFFFFFFFFuL);
    uint hi_add = uint(uint64_t(delta) >> 32);
    uint old_lo;
    buf.InterlockedAdd(index * 8u, lo_add, old_lo);
    uint carry = (old_lo > (0xFFFFFFFFu - lo_add)) ? 1u : 0u;
    uint hi_delta = hi_add + carry;
    if (hi_delta != 0u)
        buf.InterlockedAdd(index * 8u + 4u, hi_delta);
}

// Apple Silicon Metal 3.0+ natively supports hardware atomic_fetch_add on float.
// Using fp32 gradient accumulators halves gradient memory/bandwidth and avoids
// emulated two-word 64-bit integer atomics.
typedef float GradScalar;
typedef RWStructuredBuffer<float> GradBuffer;
typedef StructuredBuffer<float> ReadGradBuffer;
typedef RWStructuredBuffer<float> AtomicGradBuffer;

GradScalar toGrad(float value) { return value; }
float fromGrad(GradScalar value) { return value; }

void atomicAddGrad(AtomicGradBuffer buf, uint index, float delta)
{
    if (delta != 0.0)
        InterlockedAdd(buf[index], delta);
}
#else
typedef RWStructuredBuffer<int64_t> AtomicI64Buffer;

void atomicAddI64(RWStructuredBuffer<int64_t> buf, uint index, int64_t delta)
{
    InterlockedAdd(buf[index], delta);
}

typedef int64_t GradScalar;
typedef RWStructuredBuffer<int64_t> GradBuffer;
typedef StructuredBuffer<int64_t> ReadGradBuffer;
typedef RWStructuredBuffer<int64_t> AtomicGradBuffer;

GradScalar toGrad(float value) { return toFixed(value); }
float fromGrad(GradScalar value) { return fromFixed(value); }

void atomicAddGrad(AtomicGradBuffer buf, uint index, float delta)
{
    InterlockedAdd(buf[index], toFixed(delta));
}
#endif

// Reflected by host allocation to size gradient buffers (4 B on Metal, 8 B on Vulkan/D3D12).
struct GradAccum
{
    GradScalar value;
};

#endif // TRAINER_FIXED_POINT_SLANG
