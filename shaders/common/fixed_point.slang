// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

#ifndef TRAINER_FIXED_POINT_SLANG
#define TRAINER_FIXED_POINT_SLANG

// Fixed-point atomic gradient accumulation. Backends without 64-bit integer
// buffer atomics (e.g. MoltenVK) compile with FIXED_POINT_INT32 using 32-bit
// atomics; other backends retain the full-range 64-bit accumulator.
// Scale 4096 keeps 1/4096 gradient resolution while allowing signed sums up to
// about 524k before overflow. It completed the 3,000-step M2 Truck run; larger
// or higher-gradient scenes have not been stress-tested for accumulator range.
#ifdef FIXED_POINT_INT32
typealias fixed_t = int;
static const float FIXED_SCALE = 4096.0;
// Removal scores sum squared RGB energy, not linear gradients, so use an
// independent scale with a larger 32-bit headroom. Keep the 65,536x scale ratio
// to the 64-bit path; this is not the gradient FIXED_SCALE.
static const float REMOVAL_COST_SCALE = 16384.0;
#else
typealias fixed_t = int64_t;
static const float FIXED_SCALE = 65536.0;
static const float REMOVAL_COST_SCALE = 1073741824.0;
#endif
static const float FIXED_INV_SCALE = 1.0 / FIXED_SCALE;
// Per-axis structure scores (eta_max) use uint InterlockedMax on all backends, independent of fixed_t width.
static const float ETA_FIXED_SCALE = 65536.0;

fixed_t toFixed(float value) { return fixed_t(value * FIXED_SCALE); }
float fromFixed(fixed_t value) { return float(value) * FIXED_INV_SCALE; }

// Apple Silicon Metal 3.0+ natively supports hardware atomic_fetch_add on float.
// Using fp32 gradient accumulators halves gradient memory/bandwidth and avoids
// emulated two-word 64-bit integer atomics. Explicit FIXED_POINT_INT32 overrides
// this (e.g. unit tests exercising the 32-bit fixed-point path on Metal).
#if defined(TRAINER_METAL_BACKEND) && !defined(FIXED_POINT_INT32)
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
typedef fixed_t GradScalar;
typedef RWStructuredBuffer<fixed_t> GradBuffer;
typedef StructuredBuffer<fixed_t> ReadGradBuffer;
typedef RWStructuredBuffer<fixed_t> AtomicGradBuffer;

GradScalar toGrad(float value) { return toFixed(value); }
float fromGrad(GradScalar value) { return fromFixed(value); }

void atomicAddGrad(AtomicGradBuffer buf, uint index, float delta)
{
    InterlockedAdd(buf[index], toFixed(delta));
}
#endif

// Metal Shading Language does not expose 64-bit integer atomic_fetch_add, so emulate exact
// two's-complement 64-bit signed InterlockedAdd over the same little-endian int64_t buffer
// using a 32-bit unsigned add + carry propagation into the high word.
#if defined(TRAINER_METAL_BACKEND) && !defined(FIXED_POINT_INT32)
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
#else
typedef RWStructuredBuffer<fixed_t> AtomicI64Buffer;

void atomicAddI64(RWStructuredBuffer<fixed_t> buf, uint index, fixed_t delta)
{
    InterlockedAdd(buf[index], delta);
}
#endif

// Reflected by host allocation to size gradient buffers (4 B float on Metal, 4 B int on INT32, 8 B int64 on Vulkan/D3D12).
struct GradAccum
{
    GradScalar value;
};

#endif // TRAINER_FIXED_POINT_SLANG
