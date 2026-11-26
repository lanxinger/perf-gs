# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Per-marker GPU-timestamp durations of one recorded step; a timer is installed only
while profiling, so normal training records no timestamps."""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import slangpy as spy

# One train_step records well under this many marker boundaries, even fully instrumented.
DEFAULT_POOL_SIZE = 128


@dataclass
class StepTiming:
    """``markers_ms``: each label's summed duration (ms) over the step, in first-seen order."""

    markers_ms: dict[str, float] = field(default_factory=dict)

    def get(self, label: str, default: float = 0.0) -> float:
        return self.markers_ms.get(label, default)


class StepTimer:
    """Owns a timestamp query pool and the per-step interval bookkeeping.

    Per step: :meth:`begin_step`, :meth:`open`/:meth:`close` at each marker boundary while
    recording, then :meth:`read` once the submit has completed.
    """

    def __init__(self, device: spy.Device, pool_size: int = DEFAULT_POOL_SIZE) -> None:
        self._pool = device.create_query_pool(spy.QueryType.timestamp, pool_size)
        self._pool_size = pool_size
        self._next_index = 0
        # (label, start_index, end_index) for each entered marker scope.
        self._intervals: list[tuple[str, int, int]] = []
        # slang-rhi's Metal backend leaves QueryType.timestamp queries as no-ops (zeros);
        # bracket each marker with a submit+wait on sampled steps so --stage-timing-every works.
        info = getattr(device, "info", None)
        self._sync_timing = getattr(info, "type", None) == spy.DeviceType.metal
        self._host_intervals: list[tuple[str, float, float]] = []

    @property
    def pool(self) -> spy.QueryPool:
        return self._pool

    @property
    def needs_sync_brackets(self) -> bool:
        """True on Metal where GPU timestamp queries are unavailable and markers must be flushed."""
        return self._sync_timing

    def begin_step(self) -> None:
        """Reset for a fresh step; call host-side between two GPU-completed submits."""
        if not self._sync_timing:
            self._pool.reset()
        self._next_index = 0
        self._intervals = []
        self._host_intervals = []

    def _claim(self) -> int:
        idx = self._next_index
        self._next_index += 1
        return idx

    def open(self, label: str, encoder: spy.CommandEncoder) -> int:
        """Record a start timestamp; returns the interval id, or -1 (no-op) if the pool is full."""
        if self._sync_timing:
            now = time.perf_counter()
            self._host_intervals.append((label, now, now))
            return len(self._host_intervals) - 1
        if self._next_index + 1 >= self._pool_size:
            return -1
        start = self._claim()
        end = self._claim()
        encoder.write_timestamp(self._pool, start)
        self._intervals.append((label, start, end))
        return len(self._intervals) - 1

    def close(self, interval_id: int, encoder: spy.CommandEncoder) -> None:
        if interval_id < 0:
            return
        if self._sync_timing:
            label, start_s, _ = self._host_intervals[interval_id]
            self._host_intervals[interval_id] = (label, start_s, time.perf_counter())
            return
        _, _, end = self._intervals[interval_id]
        encoder.write_timestamp(self._pool, end)

    def read(self) -> StepTiming:
        """Per-marker summed durations in ms; call only after the timestamps' submit completed.

        Timestamp results are already seconds (the RHI applies the timestamp period).
        """
        if self._sync_timing:
            markers: dict[str, float] = {}
            for label, start_s, end_s in self._host_intervals:
                dur = max(0.0, (end_s - start_s) * 1e3)
                markers[label] = markers.get(label, 0.0) + dur
            return StepTiming(markers_ms=markers)
        n = self._next_index
        if n == 0:
            return StepTiming()
        ts = self._pool.get_timestamp_results(0, n)
        markers = {}
        for label, start, end in self._intervals:
            dur = max(0.0, (ts[end] - ts[start]) * 1e3)
            markers[label] = markers.get(label, 0.0) + dur
        return StepTiming(markers_ms=markers)
