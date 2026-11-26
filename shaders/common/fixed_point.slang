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
#else
typedef RWStructuredBuffer<int64_t> AtomicI64Buffer;

void atomicAddI64(RWStructuredBuffer<int64_t> buf, uint index, int64_t delta)
{
    InterlockedAdd(buf[index], delta);
}
#endif

#endif // TRAINER_FIXED_POINT_SLANG
