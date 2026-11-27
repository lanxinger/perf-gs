# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Reusable GPU algorithms: exclusive prefix scan and LSD radix sort."""

from __future__ import annotations

import slangpy as spy

from trainer.backend.compute.execution import (
    RADIX_GLOBAL_HIST_SIZE,
    RADIX_PART_SIZE,
    SCAN_BLOCK_SIZE,
)
from trainer.backend.compute.gpu import CommandRecorder
from trainer.backend.compute.kernels import KernelLibrary
from trainer.backend.memory.buffers import ScanResources, SortResources

_RADIX_SCAN_DIM = 128


class PrefixScan:
    """Three-level exclusive scan of up to ``SCAN_BLOCK_SIZE**3`` uints."""

    KERNELS = {
        "scan_block": ("common/scan.slang", "scanBlock"),
        "scan_block_sums": ("common/scan.slang", "scanBlockSums"),
        "add_block_offsets": ("common/scan.slang", "addBlockOffsets"),
    }

    def __init__(self, commands: CommandRecorder, kernels: KernelLibrary,
                 resources: ScanResources) -> None:
        self._commands = commands
        self._k = kernels.load(self.KERNELS)
        self.resources = resources

    def scan(self, n: int, in_buf: spy.NDBuffer, out_buf: spy.NDBuffer | None = None,
             total_buf: spy.NDBuffer | None = None) -> None:
        """Scan ``in_buf`` into ``out_buf`` (default: the work buffer) and write the total."""
        if out_buf is None:
            out_buf = self.resources.work
        if out_buf is None:
            raise ValueError("scan out_buf is required when no work buffer is allocated")
        if total_buf is None:
            total_buf = self.resources.total
        block_sums = self.resources.block_sums
        dispatch = self._commands.dispatch

        num_blocks = max(1, (n + SCAN_BLOCK_SIZE - 1) // SCAN_BLOCK_SIZE)
        dispatch(self._k.scan_block, "scan_block",
                 thread_count=[num_blocks * SCAN_BLOCK_SIZE, 1, 1],
                 in_buf=in_buf.storage, out_buf=out_buf.storage,
                 block_sums=block_sums.storage, n=n)
        if num_blocks <= SCAN_BLOCK_SIZE:
            dispatch(self._k.scan_block_sums, "scan_block_sums",
                     thread_count=[SCAN_BLOCK_SIZE, 1, 1],
                     block_sums=block_sums.storage, total_out=total_buf.storage,
                     num_blocks=num_blocks)
        else:
            block_sums_l2 = self.resources.block_sums_l2
            num_l2_blocks = max(1, (num_blocks + SCAN_BLOCK_SIZE - 1) // SCAN_BLOCK_SIZE)
            dispatch(self._k.scan_block, "scan_block",
                     thread_count=[num_l2_blocks * SCAN_BLOCK_SIZE, 1, 1],
                     in_buf=block_sums.storage, out_buf=block_sums.storage,
                     block_sums=block_sums_l2.storage, n=num_blocks)
            dispatch(self._k.scan_block_sums, "scan_block_sums",
                     thread_count=[SCAN_BLOCK_SIZE, 1, 1],
                     block_sums=block_sums_l2.storage, total_out=total_buf.storage,
                     num_blocks=num_l2_blocks)
            dispatch(self._k.add_block_offsets, "add_block_offsets",
                     thread_count=[num_l2_blocks * SCAN_BLOCK_SIZE, 1, 1],
                     out_buf=block_sums.storage, block_sums=block_sums_l2.storage,
                     n=num_blocks)
        dispatch(self._k.add_block_offsets, "add_block_offsets",
                 thread_count=[num_blocks * SCAN_BLOCK_SIZE, 1, 1],
                 out_buf=out_buf.storage, block_sums=block_sums.storage, n=n)


class RadixSort:
    """GPU-counted LSD radix sort of (64-bit key, uint value) pairs.

    Port of GPUSorting's DeviceRadixSort (MIT); upsweep/downsweep launch only live partitions.
    """

    KERNELS = {
        "init_global_hist": ("sort/device_radix_sort.slang", "initGlobalHist"),
        "prepare_dispatch": ("sort/device_radix_sort.slang", "prepareDispatchArgs"),
        "upsweep": ("sort/device_radix_sort.slang", "upsweep"),
        "scan": ("sort/device_radix_sort.slang", "scan"),
        "downsweep": ("sort/device_radix_sort.slang", "downsweep"),
    }

    def __init__(self, commands: CommandRecorder, kernels: KernelLibrary,
                 resources: SortResources) -> None:
        self._commands = commands
        self._k = kernels.load(self.KERNELS)
        self.resources = resources

    def prepare_dispatch(self, count: spy.NDBuffer,
                         known_count: int | None = None) -> None:
        """Compute indirect dispatch args (and zero global histogram) ahead of :meth:`sort`.

        Calling this before an earlier indirect dispatch lets Metal prefetch ``dispatch_args``
        on the same flush, eliminating an extra GPU sync. When ``known_count`` is supplied on
        Metal, ``dispatch_args`` is seeded directly without any GPU readback.
        """
        r = self.resources
        commands = self._commands
        max_blocks = (r.capacity + RADIX_PART_SIZE - 1) // RADIX_PART_SIZE
        commands.dispatch(self._k.prepare_dispatch, "radix_prepare_dispatch",
            thread_count=[1, 1, 1],
            n_buf=count.storage,
            dispatch_args=r.dispatch_args.storage,
            sort_stats=r.stats.storage,
            max_blocks=max_blocks,
            max_sort_pairs=r.capacity)
        commands.dispatch(self._k.init_global_hist, "radix_init_global_hist",
            thread_count=[RADIX_GLOBAL_HIST_SIZE, 1, 1],
            b_global_hist=r.global_hist.storage)
        if commands.batch is not None:
            if known_count is not None:
                n = min(max(int(known_count), 0), r.capacity)
                active_blocks = (n + RADIX_PART_SIZE - 1) // RADIX_PART_SIZE
                g = min(max(active_blocks, 1), max_blocks)
                commands.batch.seed_indirect(r.dispatch_args, (g, 1, 1))
            else:
                commands.batch.prefetch_indirect(r.dispatch_args)

    def sort(self, count: spy.NDBuffer, key_bits: int = 64, *,
             args_prepared: bool = False, known_count: int | None = None
             ) -> tuple[spy.NDBuffer, spy.NDBuffer, spy.NDBuffer]:
        """Sort the staged pairs on their low ``key_bits``; return the sorted buffers.

        ``count[0]`` holds the pair count and is clamped in place to capacity. Keys of at
        most 32 bits live in ``keys_lo`` alone; ``keys_hi`` is then left untouched.
        """
        r = self.resources
        commands = self._commands
        src = (r.keys_lo, r.keys_hi, r.values)
        dst = (r.keys_lo_alt, r.keys_hi_alt, r.values_alt)
        max_blocks = (r.capacity + RADIX_PART_SIZE - 1) // RADIX_PART_SIZE
        key_words = 1 if key_bits <= 32 else 2

        if not args_prepared:
            self.prepare_dispatch(count, known_count=known_count)

        for pass_idx in range((key_bits + 7) // 8):
            radix_shift = (pass_idx % 4) * 8
            key_word = 0 if pass_idx < 4 else 1
            with commands.marker_group(f"pass_{pass_idx}"):
                commands.dispatch_indirect(self._k.upsweep, "radix_upsweep",
                    r.dispatch_args,
                    keys_lo=src[0].storage, keys_hi=src[1].storage,
                    b_pass_hist=r.pass_hist.storage,
                    b_global_hist=r.global_hist.storage,
                    n_buf=count.storage,
                    radix_shift=radix_shift, key_word=key_word,
                    pass_idx=pass_idx, max_blocks=max_blocks)
                commands.dispatch(self._k.scan, "radix_scan",
                    thread_count=[256 * _RADIX_SCAN_DIM, 1, 1],
                    b_pass_hist=r.pass_hist.storage,
                    n_buf=count.storage,
                    max_blocks=max_blocks)
                commands.dispatch_indirect(self._k.downsweep, "radix_downsweep",
                    r.dispatch_args,
                    keys_lo_in=src[0].storage, keys_hi_in=src[1].storage,
                    values_in=src[2].storage,
                    keys_lo_out=dst[0].storage, keys_hi_out=dst[1].storage,
                    values_out=dst[2].storage,
                    b_pass_hist=r.pass_hist.storage,
                    b_global_hist=r.global_hist.storage,
                    n_buf=count.storage,
                    radix_shift=radix_shift, key_word=key_word, key_words=key_words,
                    pass_idx=pass_idx, max_blocks=max_blocks)
            src, dst = dst, src
        return src
