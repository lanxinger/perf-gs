// Modfications to the software are Copyright (c) 2026 Electronic Arts Inc, provided under the original license.
// Original sorting code: Copyright (c) 2024 Thomas Smith (MIT).

// Reduce-then-scan device radix sort from GPUSorting (b0nes164, MIT): no forward-progress requirement.
// Wave size >= 16, ascending, stable; sorts (keys_lo, keys_hi, values), passes 0-3 on keys_lo, 4-7 on
// keys_hi. With key_words = 1 keys_hi is neither read nor written.

#ifndef TRAINER_DEVICE_RADIX_SORT_SLANG
#define TRAINER_DEVICE_RADIX_SORT_SLANG

static const uint RADIX = 256u;
static const uint RADIX_MASK = 255u;
static const uint RADIX_LOG = 8u;

// Upsweep / Scan block sizes, as in GPUSorting.
static const uint US_DIM = 128u;
static const uint SCAN_DIM = 128u;

// Downsweep sizes are shared with the host; the 1D dispatch limit caps n at 65535 * PART_SIZE.
#include "common/execution.slang"
static const uint D_DIM = RADIX_DIGITS;
static const uint KEYS_PER_THREAD = RADIX_KEYS_PER_THREAD;
static const uint PART_SIZE = RADIX_PART_SIZE;

// max(wave histograms at 16 lanes = D_DIM / 16 * RADIX = 4096, PART_SIZE + RADIX = 2304).
static const uint D_TOTAL_SMEM = 4096u;

static const uint SENTINEL = 0xFFFFFFFFu;

// Upsweep's two histograms: threads 0..63 fill hist 0, 64..127 hist 1; merged before write-back.
groupshared uint g_us[RADIX * 2];

groupshared uint g_scan[SCAN_DIM];

// Downsweep: [0, waveHistsSize) wave histograms while ranking, then [0, PART_SIZE) staged keys
// and [PART_SIZE, PART_SIZE + RADIX) per-digit device offsets.
groupshared uint g_d[D_TOTAL_SMEM];

struct KeyStruct { uint k[KEYS_PER_THREAD]; };
struct OffsetStruct { uint o[KEYS_PER_THREAD]; };
struct DigitStruct { uint d[KEYS_PER_THREAD]; };

uint activeBlocksOf(uint n) { return (n + PART_SIZE - 1u) / PART_SIZE; }

// Host-encoded digit source: key_word 0 = keys_lo, 1 = keys_hi; radix_shift is 0/8/16/24.
uint loadDigitSourceWord(uint idx, uint key_word,
                         StructuredBuffer<uint> keys_lo,
                         StructuredBuffer<uint> keys_hi)
{
    return (key_word == 0u) ? keys_lo[idx] : keys_hi[idx];
}

uint extractDigit(uint word, uint radix_shift)
{
    return (word >> radix_shift) & RADIX_MASK;
}

uint getWaveIndex(uint gtid)
{
    return gtid / trainerWaveLaneCount();
}

uint waveHistsSize()
{
    return (D_DIM / trainerWaveLaneCount()) * RADIX;
}

// Clears b_global_hist once per sort; dispatch one thread per entry (2048).
[shader("compute")]
[numthreads(256, 1, 1)]
void initGlobalHist(
    uint3 dtid: SV_DispatchThreadID,
    RWStructuredBuffer<uint> b_global_hist)
{
    if (dtid.x < RADIX * 8u)
        b_global_hist[dtid.x] = 0u;
}

// Indirect args (active_blocks, 1, 1) for upsweep and downsweep; scan is a direct 256-group dispatch.
[shader("compute")]
[numthreads(1, 1, 1)]
void prepareDispatchArgs(
    RWStructuredBuffer<uint> n_buf,
    RWStructuredBuffer<uint> dispatch_args,
    RWStructuredBuffer<uint> sort_stats, // [0] peak requested pairs, [1] sorts that dropped pairs
    uniform uint max_blocks,     // defensive cap: never launch more groups than we have scratch for
    uniform uint max_sort_pairs) // capacity of the sort-pair buffers
{
    uint requested = n_buf[0];
    sort_stats[0] = max(sort_stats[0], requested);
    if (requested > max_sort_pairs)
        sort_stats[1] += 1u;

    // Clamp n_buf in place: bin_emit drops pairs past capacity, and every later reader
    // (sort, identifyTileRanges) would otherwise read past the buffer.
    uint n = min(requested, max_sort_pairs);
    n_buf[0] = n;

    uint active_blocks = (n + PART_SIZE - 1u) / PART_SIZE;
    // Launch at least one group (zero-group dispatches may error); it early-outs on active_blocks.
    uint g = (active_blocks == 0u) ? 1u : active_blocks;
    // Clamp: if n is somehow corrupted, don't queue billions of workgroups.
    g = min(g, max_blocks);
    dispatch_args[0] = g;
    dispatch_args[1] = 1u;
    dispatch_args[2] = 1u;
}

// Per-partition digit counts -> b_pass_hist[digit * max_blocks + partition]; their exclusive
// digit scan is atomically added to b_global_hist[pass_idx * RADIX + digit].
[shader("compute")]
[numthreads(US_DIM, 1, 1)]
void upsweep(
    uint3 gid: SV_GroupID,
    uint3 gtid: SV_GroupThreadID,
    StructuredBuffer<uint> keys_lo,
    StructuredBuffer<uint> keys_hi,
    RWStructuredBuffer<uint> b_pass_hist,
    RWStructuredBuffer<uint> b_global_hist,
    StructuredBuffer<uint> n_buf,
    uniform uint radix_shift,
    uniform uint key_word,
    uniform uint pass_idx,
    uniform uint max_blocks)
{
    uint n = n_buf[0];
    uint active_blocks = activeBlocksOf(n);
    uint g = gid.x;
    uint t = gtid.x;

    // Inactive partitions zero their b_pass_hist column so Scan's fixed-stride scan stays exact.
    if (g >= active_blocks)
    {
        for (uint i = t; i < RADIX; i += US_DIM)
            b_pass_hist[i * max_blocks + g] = 0u;
        return;
    }

    for (uint i = t; i < RADIX * 2u; i += US_DIM)
        g_us[i] = 0u;
    GroupMemoryBarrierWithGroupSync();

    // Two histograms reduce shared-atomic contention.
    uint partition_start = g * PART_SIZE;
    uint partition_end = (g == active_blocks - 1u) ? n : partition_start + PART_SIZE;
    uint hist_offset = (t / 64u) * RADIX;
    for (uint i = partition_start + t; i < partition_end; i += US_DIM)
    {
        uint word = loadDigitSourceWord(i, key_word, keys_lo, keys_hi);
        InterlockedAdd(g_us[extractDigit(word, radix_shift) + hist_offset], 1u);
    }
    GroupMemoryBarrierWithGroupSync();

    // Merge the histograms, store this partition's counts, then start the digit scan per wave.
    for (uint i = t; i < RADIX; i += US_DIM)
    {
        g_us[i] += g_us[i + RADIX];
        b_pass_hist[i * max_blocks + g] = g_us[i];
        g_us[i] += WavePrefixSum(g_us[i]);  // inclusive within the wave
    }

    // GlobalHistExclusiveScanWGE16: wave scans -> 256-wide exclusive scan, added to this pass's row.
    GroupMemoryBarrierWithGroupSync();
    uint lane_count = trainerWaveLaneCount();
    uint lane_idx = trainerWaveLaneIndex(t);
    if (t < (RADIX / lane_count))
    {
        g_us[(t + 1u) * lane_count - 1u] +=
            WavePrefixSum(g_us[(t + 1u) * lane_count - 1u]);
    }
    GroupMemoryBarrierWithGroupSync();

    uint global_hist_offset = pass_idx * RADIX;
    uint lane_mask = lane_count - 1u;
    uint circular_lane_shift = (lane_idx + 1u) & lane_mask;
    for (uint i = t; i < RADIX; i += US_DIM)
    {
        uint index = circular_lane_shift + (i & ~lane_mask);
        uint add = (lane_idx != lane_mask ? g_us[i] : 0u)
                 + (i >= lane_count ? WaveReadLaneAt(g_us[i - 1u], 0u) : 0u);
        InterlockedAdd(b_global_hist[index + global_hist_offset], add);
    }
}

// Exclusive scan of each digit's active b_pass_hist entries, one group per digit (256 groups),
// written back one lane shifted (GPUSorting's circular lane shift).
[shader("compute")]
[numthreads(SCAN_DIM, 1, 1)]
void scan(
    uint3 gid: SV_GroupID,
    uint3 gtid: SV_GroupThreadID,
    RWStructuredBuffer<uint> b_pass_hist,
    StructuredBuffer<uint> n_buf,
    uniform uint max_blocks)
{
    uint n = n_buf[0];
    uint active_blocks = activeBlocksOf(n);
    uint digit = gid.x;
    uint t = gtid.x;

    uint device_offset = digit * max_blocks;
    uint lane_count = trainerWaveLaneCount();
    uint lane_idx = trainerWaveLaneIndex(t);
    uint lane_mask = lane_count - 1u;
    uint circular_lane_shift = (lane_idx + 1u) & lane_mask;

    uint parts_end = (active_blocks / SCAN_DIM) * SCAN_DIM;
    uint reduction = 0u;

    // Full chunks.
    for (uint i = t; i < parts_end; i += SCAN_DIM)
    {
        g_scan[t] = b_pass_hist[i + device_offset];
        g_scan[t] += WavePrefixSum(g_scan[t]);
        GroupMemoryBarrierWithGroupSync();

        if (t < SCAN_DIM / lane_count)
        {
            g_scan[(t + 1u) * lane_count - 1u] +=
                WavePrefixSum(g_scan[(t + 1u) * lane_count - 1u]);
        }
        GroupMemoryBarrierWithGroupSync();

        b_pass_hist[circular_lane_shift + (i & ~lane_mask) + device_offset] =
            (lane_idx != lane_mask ? g_scan[t] : 0u)
            + (t >= lane_count ? WaveReadLaneAt(g_scan[t - 1u], 0u) : 0u)
            + reduction;

        reduction += g_scan[SCAN_DIM - 1u];
        GroupMemoryBarrierWithGroupSync();
    }

    // Partial chunk: the last (< SCAN_DIM) active blocks.
    uint i = t + parts_end;
    if (i < active_blocks)
        g_scan[t] = b_pass_hist[device_offset + i];
    else
        g_scan[t] = 0u;
    g_scan[t] += WavePrefixSum(g_scan[t]);
    GroupMemoryBarrierWithGroupSync();

    if (t < SCAN_DIM / lane_count)
    {
        g_scan[(t + 1u) * lane_count - 1u] +=
            WavePrefixSum(g_scan[(t + 1u) * lane_count - 1u]);
    }
    GroupMemoryBarrierWithGroupSync();

    uint index = circular_lane_shift + (i & ~lane_mask);
    if (index < active_blocks)
    {
        b_pass_hist[index + device_offset] =
            (lane_idx != lane_mask ? g_scan[t] : 0u)
            + (t >= lane_count ? g_scan[(t & ~lane_mask) - 1u] : 0u)
            + reduction;
    }
}

// Downsweep: rank the digit-source word by wave multisplit, then scatter all three record
// words with the same offsets.

// Narrows waveFlags (one bit per lane) to the lanes sharing this key's digit, one ballot per bit.
void warpLevelMultiSplitWGE16(uint key, uint radix_shift, uint wave_parts,
                              inout uint4 waveFlags)
{
    [unroll]
    for (uint k = 0u; k < RADIX_LOG; ++k)
    {
        bool t_bit = ((key >> (k + radix_shift)) & 1u) != 0u;
        uint4 ballot = WaveActiveBallot(t_bit);
        uint mask_all = t_bit ? 0u : 0xFFFFFFFFu;
        for (uint wp = 0u; wp < wave_parts; ++wp)
            waveFlags[wp] &= mask_all ^ ballot[wp];
    }
}

uint waveFlagsWGE16()
{
    uint lane_count = trainerWaveLaneCount();
    return ((lane_count & 31u) != 0u) ? ((1u << lane_count) - 1u) : 0xFFFFFFFFu;
}

void countPeerBits(uint gtid, inout uint peerBits, inout uint totalBits,
                   uint4 waveFlags, uint wave_parts)
{
    uint lane_idx = trainerWaveLaneIndex(gtid);
    for (uint wp = 0u; wp < wave_parts; ++wp)
    {
        if (lane_idx >= wp * 32u)
        {
            uint ltMask = (lane_idx >= (wp + 1u) * 32u)
                ? 0xFFFFFFFFu
                : ((1u << (lane_idx & 31u)) - 1u);
            peerBits += countbits(waveFlags[wp] & ltMask);
        }
        totalBits += countbits(waveFlags[wp]);
    }
}

uint findLowestRankPeer(uint4 waveFlags, uint wave_parts)
{
    uint lowest = 0u;
    for (uint wp = 0u; wp < wave_parts; ++wp)
    {
        uint fbl = firstbitlow(waveFlags[wp]);
        if (fbl == 0xFFFFFFFFu)
            lowest += 32u;
        else
            return lowest + fbl;
    }
    return 0u;  // unreachable
}

// Intra-wave rank of each key among its digit peers; the lowest peer bumps the wave histogram
// for all of them. updateOffsetsWGE16 later adds the cross-wave offsets.
OffsetStruct rankKeysWGE16(uint gtid, KeyStruct keys, uint radix_shift)
{
    OffsetStruct offsets;
    uint wave_parts = (trainerWaveLaneCount() + 31u) / 32u;
    [unroll]
    for (uint i = 0u; i < KEYS_PER_THREAD; ++i)
    {
        uint4 waveFlags;
        waveFlags.x = waveFlagsWGE16();
        waveFlags.y = 0u; waveFlags.z = 0u; waveFlags.w = 0u;
        // If wave has more than 32 lanes (e.g. AMD RDNA 64) also fill next parts.
        if (wave_parts > 1u) waveFlags.y = 0xFFFFFFFFu;
        if (wave_parts > 2u) waveFlags.z = 0xFFFFFFFFu;
        if (wave_parts > 3u) waveFlags.w = 0xFFFFFFFFu;

        warpLevelMultiSplitWGE16(keys.k[i], radix_shift, wave_parts, waveFlags);

        uint index = extractDigit(keys.k[i], radix_shift) + getWaveIndex(gtid) * RADIX;
        uint lowest = findLowestRankPeer(waveFlags, wave_parts);

        uint peerBits = 0u;
        uint totalBits = 0u;
        countPeerBits(gtid, peerBits, totalBits, waveFlags, wave_parts);

        uint preInc = 0u;
        if (peerBits == 0u)
            InterlockedAdd(g_d[index], totalBits, preInc);
        offsets.o[i] = WaveReadLaneAt(preInc, lowest) + peerBits;
    }
    return offsets;
}

// Per digit, exclusive scan across the wave histograms in g_d; returns the digit's total.
uint waveHistInclusiveScanCircularShiftWGE16(uint gtid)
{
    uint histReduction = g_d[gtid];
    for (uint i = gtid + RADIX; i < waveHistsSize(); i += RADIX)
    {
        histReduction += g_d[i];
        g_d[i] = histReduction - g_d[i];  // exclusive scan across waves at fixed digit
    }
    return histReduction;
}

// Exclusive scan of the per-digit totals across all 256 digits (circular lane shift).
void waveHistReductionExclusiveScanWGE16(uint gtid, uint histReduction)
{
    uint lane_count = trainerWaveLaneCount();
    uint lane_idx = trainerWaveLaneIndex(gtid);
    if (gtid < RADIX)
    {
        uint lane_mask = lane_count - 1u;
        g_d[((lane_idx + 1u) & lane_mask) + (gtid & ~lane_mask)] = histReduction;
    }
    GroupMemoryBarrierWithGroupSync();

    if (gtid < RADIX / lane_count)
    {
        g_d[gtid * lane_count] = WavePrefixSum(g_d[gtid * lane_count]);
    }
    GroupMemoryBarrierWithGroupSync();

    if (gtid < RADIX && lane_idx != 0u)
        g_d[gtid] += WaveReadLaneAt(g_d[gtid - 1u], 1u);
}

// Turns intra-wave ranks into positions in the [0, PART_SIZE) staging area.
void updateOffsetsWGE16(uint gtid, inout OffsetStruct offsets, KeyStruct keys,
                        uint radix_shift)
{
    if (gtid >= trainerWaveLaneCount())
    {
        uint tw = getWaveIndex(gtid) * RADIX;
        [unroll]
        for (uint i = 0u; i < KEYS_PER_THREAD; ++i)
        {
            uint d = extractDigit(keys.k[i], radix_shift);
            offsets.o[i] += g_d[d + tw] + g_d[d];
        }
    }
    else
    {
        [unroll]
        for (uint i = 0u; i < KEYS_PER_THREAD; ++i)
            offsets.o[i] += g_d[extractDigit(keys.k[i], radix_shift)];
    }
}

void scatterKeysShared(OffsetStruct offsets, KeyStruct keys)
{
    [unroll]
    for (uint i = 0u; i < KEYS_PER_THREAD; ++i)
        g_d[offsets.o[i]] = keys.k[i];
}

uint readWord(uint idx, uint key_word,
              StructuredBuffer<uint> keys_lo, StructuredBuffer<uint> keys_hi)
{
    return (key_word == 0u) ? keys_lo[idx] : keys_hi[idx];
}

KeyStruct loadKeysWGE16Sel(uint gtid, uint part_index, uint key_word,
                           StructuredBuffer<uint> keys_lo,
                           StructuredBuffer<uint> keys_hi)
{
    KeyStruct keys;
    uint lane_count = trainerWaveLaneCount();
    uint base = trainerWaveLaneIndex(gtid) + getWaveIndex(gtid) * KEYS_PER_THREAD * lane_count
              + part_index * PART_SIZE;
    [unroll]
    for (uint i = 0u; i < KEYS_PER_THREAD; ++i)
    {
        keys.k[i] = readWord(base + i * lane_count, key_word, keys_lo, keys_hi);
    }
    return keys;
}

KeyStruct loadKeysPartialWGE16Sel(uint gtid, uint part_index, uint key_word,
                                  StructuredBuffer<uint> keys_lo,
                                  StructuredBuffer<uint> keys_hi,
                                  uint n_bound)
{
    KeyStruct keys;
    uint lane_count = trainerWaveLaneCount();
    uint base = trainerWaveLaneIndex(gtid) + getWaveIndex(gtid) * KEYS_PER_THREAD * lane_count
              + part_index * PART_SIZE;
    [unroll]
    for (uint i = 0u; i < KEYS_PER_THREAD; ++i)
    {
        uint idx = base + i * lane_count;
        keys.k[i] = (idx < n_bound) ? readWord(idx, key_word, keys_lo, keys_hi) : SENTINEL;
    }
    return keys;
}

// Single-buffer variants for the value payload.
KeyStruct loadKeysWGE16(uint gtid, uint part_index, StructuredBuffer<uint> buf)
{
    KeyStruct keys;
    uint lane_count = trainerWaveLaneCount();
    uint base = trainerWaveLaneIndex(gtid) + getWaveIndex(gtid) * KEYS_PER_THREAD * lane_count
              + part_index * PART_SIZE;
    [unroll]
    for (uint i = 0u; i < KEYS_PER_THREAD; ++i)
        keys.k[i] = buf[base + i * lane_count];
    return keys;
}

KeyStruct loadKeysPartialWGE16(uint gtid, uint part_index,
                               StructuredBuffer<uint> buf, uint n_bound)
{
    KeyStruct keys;
    uint lane_count = trainerWaveLaneCount();
    uint base = trainerWaveLaneIndex(gtid) + getWaveIndex(gtid) * KEYS_PER_THREAD * lane_count
              + part_index * PART_SIZE;
    [unroll]
    for (uint i = 0u; i < KEYS_PER_THREAD; ++i)
    {
        uint idx = base + i * lane_count;
        keys.k[i] = (idx < n_bound) ? buf[idx] : SENTINEL;
    }
    return keys;
}

// Load device scatter offsets (per digit) into g_d[PART_SIZE + digit].
void loadThreadBlockReductions(uint gtid, uint part_index,
                                RWStructuredBuffer<uint> b_pass_hist,
                                RWStructuredBuffer<uint> b_global_hist,
                                uint pass_idx, uint max_blocks,
                                uint exclusiveHistReduction)
{
    if (gtid < RADIX)
    {
        g_d[gtid + PART_SIZE] =
            b_global_hist[gtid + pass_idx * RADIX]
            + b_pass_hist[gtid * max_blocks + part_index]
            - exclusiveHistReduction;
    }
}

// Scatters the staged digit-source words and caches their digits for the payload scatters.
void scatterKeyPhaseOne(uint gtid, uint radix_shift,
                        RWStructuredBuffer<uint> key_out,
                        inout DigitStruct digits)
{
    [unroll]
    for (uint i = 0u, t = gtid; i < KEYS_PER_THREAD; ++i, t += D_DIM)
    {
        uint word = g_d[t];
        uint d = extractDigit(word, radix_shift);
        digits.d[i] = d;
        uint dev_off = g_d[d + PART_SIZE] + t;
        key_out[dev_off] = word;
    }
}

void scatterKeyPhasePartialOne(uint gtid, uint radix_shift, uint final_size,
                               RWStructuredBuffer<uint> key_out,
                               inout DigitStruct digits)
{
    [unroll]
    for (uint i = 0u, t = gtid; i < KEYS_PER_THREAD; ++i, t += D_DIM)
    {
        if (t < final_size)
        {
            uint word = g_d[t];
            uint d = extractDigit(word, radix_shift);
            digits.d[i] = d;
            uint dev_off = g_d[d + PART_SIZE] + t;
            key_out[dev_off] = word;
        }
    }
}

// key_word-selected wrappers.
void scatterKeyPhase(uint gtid, uint radix_shift, uint key_word,
                     RWStructuredBuffer<uint> keys_lo_out,
                     RWStructuredBuffer<uint> keys_hi_out,
                     inout DigitStruct digits)
{
    if (key_word == 0u)
        scatterKeyPhaseOne(gtid, radix_shift, keys_lo_out, digits);
    else
        scatterKeyPhaseOne(gtid, radix_shift, keys_hi_out, digits);
}

void scatterKeyPhasePartial(uint gtid, uint radix_shift, uint key_word,
                            uint final_size,
                            RWStructuredBuffer<uint> keys_lo_out,
                            RWStructuredBuffer<uint> keys_hi_out,
                            inout DigitStruct digits)
{
    if (key_word == 0u)
        scatterKeyPhasePartialOne(gtid, radix_shift, final_size, keys_lo_out, digits);
    else
        scatterKeyPhasePartialOne(gtid, radix_shift, final_size, keys_hi_out, digits);
}

void scatterPayloadShared(OffsetStruct offsets, KeyStruct payloads)
{
    [unroll]
    for (uint i = 0u; i < KEYS_PER_THREAD; ++i)
        g_d[offsets.o[i]] = payloads.k[i];
}

void scatterPayloadDevice(uint gtid, DigitStruct digits,
                          RWStructuredBuffer<uint> out_buf)
{
    [unroll]
    for (uint i = 0u, t = gtid; i < KEYS_PER_THREAD; ++i, t += D_DIM)
    {
        uint dev_off = g_d[digits.d[i] + PART_SIZE] + t;
        out_buf[dev_off] = g_d[t];
    }
}

void scatterPayloadDevicePartial(uint gtid, uint final_size, DigitStruct digits,
                                 RWStructuredBuffer<uint> out_buf)
{
    [unroll]
    for (uint i = 0u, t = gtid; i < KEYS_PER_THREAD; ++i, t += D_DIM)
    {
        if (t < final_size)
        {
            uint dev_off = g_d[digits.d[i] + PART_SIZE] + t;
            out_buf[dev_off] = g_d[t];
        }
    }
}

// DXC needs a statically resolvable target buffer, so branch on key_word around the whole scatter.
void scatterPayloadDeviceSel(uint gtid, DigitStruct digits, uint key_word,
                             RWStructuredBuffer<uint> keys_lo_out,
                             RWStructuredBuffer<uint> keys_hi_out)
{
    if (key_word == 0u)
        scatterPayloadDevice(gtid, digits, keys_hi_out);   // other = hi
    else
        scatterPayloadDevice(gtid, digits, keys_lo_out);   // other = lo
}

void scatterPayloadDevicePartialSel(uint gtid, uint final_size, DigitStruct digits,
                                    uint key_word,
                                    RWStructuredBuffer<uint> keys_lo_out,
                                    RWStructuredBuffer<uint> keys_hi_out)
{
    if (key_word == 0u)
        scatterPayloadDevicePartial(gtid, final_size, digits, keys_hi_out);
    else
        scatterPayloadDevicePartial(gtid, final_size, digits, keys_lo_out);
}

[shader("compute")]
[numthreads(D_DIM, 1, 1)]
void downsweep(
    uint3 gid: SV_GroupID,
    uint3 gtid: SV_GroupThreadID,
    StructuredBuffer<uint> keys_lo_in,
    StructuredBuffer<uint> keys_hi_in,
    StructuredBuffer<uint> values_in,
    RWStructuredBuffer<uint> keys_lo_out,
    RWStructuredBuffer<uint> keys_hi_out,
    RWStructuredBuffer<uint> values_out,
    RWStructuredBuffer<uint> b_pass_hist,
    RWStructuredBuffer<uint> b_global_hist,
    StructuredBuffer<uint> n_buf,
    uniform uint radix_shift,
    uniform uint key_word,
    uniform uint key_words,
    uniform uint pass_idx,
    uniform uint max_blocks)
{
    uint n = n_buf[0];
    uint active_blocks = activeBlocksOf(n);
    uint g = gid.x;
    uint t = gtid.x;

    if (g >= active_blocks) return;

    KeyStruct keys;
    OffsetStruct offsets;
    DigitStruct digits;

    // 1. Clear wave histograms in shared memory.
    uint hist_end = waveHistsSize();
    for (uint i = t; i < hist_end; i += D_DIM)
        g_d[i] = 0u;

    // 2. Load digit-source-word keys. Last active partition may be partial.
    bool last = (g == active_blocks - 1u);
    if (!last)
        keys = loadKeysWGE16Sel(t, g, key_word, keys_lo_in, keys_hi_in);
    else
        keys = loadKeysPartialWGE16Sel(t, g, key_word, keys_lo_in, keys_hi_in, n);

    GroupMemoryBarrierWithGroupSync();

    // 3. Rank within the block using wave multi-split.
    offsets = rankKeysWGE16(t, keys, radix_shift);
    GroupMemoryBarrierWithGroupSync();

    // 4. Scan the wave histograms per digit, then the digit totals.
    uint histReduction;
    if (t < RADIX)
    {
        histReduction = waveHistInclusiveScanCircularShiftWGE16(t);
        histReduction += WavePrefixSum(histReduction);
    }
    GroupMemoryBarrierWithGroupSync();

    waveHistReductionExclusiveScanWGE16(t, histReduction);
    GroupMemoryBarrierWithGroupSync();

    // 5. Intra-wave ranks -> staging positions.
    updateOffsetsWGE16(t, offsets, keys, radix_shift);

    uint exclusiveHistReduction = 0u;
    if (t < RADIX)
        exclusiveHistReduction = g_d[t];
    GroupMemoryBarrierWithGroupSync();

    // 6. Stage the digit-source-word into g_d[0..PART_SIZE).
    scatterKeysShared(offsets, keys);

    // 7. Load per-digit device scatter offsets into g_d[PART_SIZE + digit].
    loadThreadBlockReductions(t, g, b_pass_hist, b_global_hist,
                              pass_idx, max_blocks, exclusiveHistReduction);
    GroupMemoryBarrierWithGroupSync();

    // 8. Scatter the digit-source word and cache its digits.
    if (!last)
    {
        scatterKeyPhase(t, radix_shift, key_word,
                        keys_lo_out, keys_hi_out, digits);
    }
    else
    {
        uint final_size = n - g * PART_SIZE;
        scatterKeyPhasePartial(t, radix_shift, key_word, final_size,
                               keys_lo_out, keys_hi_out, digits);
    }
    GroupMemoryBarrierWithGroupSync();

    // 9. Scatter the other key word (64-bit keys only).
    if (key_words > 1u)
    {
        KeyStruct otherKey;
        uint other_key_word = 1u - key_word;
        if (!last)
            otherKey = loadKeysWGE16Sel(t, g, other_key_word, keys_lo_in, keys_hi_in);
        else
            otherKey = loadKeysPartialWGE16Sel(t, g, other_key_word, keys_lo_in, keys_hi_in, n);
        scatterPayloadShared(offsets, otherKey);
        GroupMemoryBarrierWithGroupSync();

        if (!last)
            scatterPayloadDeviceSel(t, digits, key_word, keys_lo_out, keys_hi_out);
        else
        {
            uint final_size = n - g * PART_SIZE;
            scatterPayloadDevicePartialSel(t, final_size, digits, key_word, keys_lo_out, keys_hi_out);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // 10. Scatter the values.
    KeyStruct valuePayload;
    if (!last)
        valuePayload = loadKeysWGE16(t, g, values_in);
    else
        valuePayload = loadKeysPartialWGE16(t, g, values_in, n);
    scatterPayloadShared(offsets, valuePayload);
    GroupMemoryBarrierWithGroupSync();

    if (!last)
        scatterPayloadDevice(t, digits, values_out);
    else
    {
        uint final_size = n - g * PART_SIZE;
        scatterPayloadDevicePartial(t, final_size, digits, values_out);
    }
}

#endif // TRAINER_DEVICE_RADIX_SORT_SLANG
