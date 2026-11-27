// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Execution constants shared by the kernels; mirrored by trainer.backend.compute.execution.

#ifndef TRAINER_EXECUTION_SLANG
#define TRAINER_EXECUTION_SLANG

static const uint TILE_WIDTH = 8;
static const uint TILE_HEIGHT = 8;
static const uint TILE_AREA = 64;
static const uint CLUSTER_SIZE = 128;
static const uint RADIX_DIGITS = 256;
static const uint RADIX_KEYS_PER_THREAD = 8;
static const uint RADIX_PASSES = 8;
static const uint RADIX_PART_SIZE = 2048;
static const uint RADIX_GLOBAL_HIST_SIZE = 2048;
static const uint SCAN_BLOCK_SIZE = 1024;
static const uint RASTER_BUCKET_SIZE = 32;

// Indirect linear dispatches tile workgroups over X and Y to stay within the 65535-group X limit.
static const uint LINEAR_DISPATCH_WIDTH = 65535u;

uint linearThreadIndex256(uint3 thread)
{
    uint row = thread.y * (LINEAR_DISPATCH_WIDTH * 256u);
    return row + min(thread.x, 0xffffffffu - row);  // Ignore padded lanes beyond uint range.
}

// Slang's Metal emitter drops the [[buffer(0)]] constant-buffer qualifier from EntryPointParams
// whenever a compute kernel combines uniform parameters with synthesized [[threads_per_simdgroup]]
// or [[thread_index_in_simdgroup]] builtins. Apple Silicon GPUs always execute 32-wide SIMD-groups
// partitioned linearly over the threadgroup, so 1D workgroups can derive both without the builtins.
#ifdef TRAINER_METAL_BACKEND
uint trainerWaveLaneCount() { return 32u; }
uint trainerWaveLaneIndex(uint group_thread_linear) { return group_thread_linear & 31u; }
#else
uint trainerWaveLaneCount() { return WaveGetLaneCount(); }
uint trainerWaveLaneIndex(uint group_thread_linear) { return WaveGetLaneIndex(); }
#endif

#endif // TRAINER_EXECUTION_SLANG
