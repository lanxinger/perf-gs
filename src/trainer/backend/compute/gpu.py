# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""GPU runtime (queue + submission) and command recording: each recorded batch
becomes one command buffer submitted under the shared queue lock."""

from __future__ import annotations

import contextlib
import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np
import slangpy as spy


@lru_cache(maxsize=None)
def marker_color(name: str) -> tuple[float, float, float]:
    """Run-stable RGB for a debug-marker name (blake2s: builtin hash() is salted per process)."""
    digest = hashlib.blake2s(name.encode("utf-8"), digest_size=3).digest()
    # Bias into [0.35, 1.0] so colors stay legible (never near-black).
    return tuple(0.35 + (b / 255.0) * 0.65 for b in digest)  # type: ignore[return-value]


@dataclass(frozen=True)
class SubmitHandle:
    """A submitted command buffer; wait on it to bound queue depth without a full device idle."""
    submit_id: int


def _is_metal_runtime(runtime: GpuRuntime | None) -> bool:
    device = getattr(runtime, "device", None)
    info = getattr(device, "info", None)
    return getattr(info, "type", None) == spy.DeviceType.metal


def _read_indirect_groups(args_buf: spy.NDBuffer, offset: int) -> tuple[int, int, int]:
    raw = np.frombuffer(args_buf.storage.to_numpy().tobytes(), dtype=np.uint32)
    base = offset // 4
    return int(raw[base]), int(raw[base + 1]), int(raw[base + 2])


class CommandBatch:
    """One command buffer's worth of recorded GPU work (wraps a ``spy.CommandEncoder``)."""

    def __init__(self, encoder: spy.CommandEncoder, label: str = "",
                 runtime: GpuRuntime | None = None) -> None:
        self._encoder = encoder
        self._label = label
        self._runtime = runtime
        self._emulate_indirect = _is_metal_runtime(runtime)
        self._marker_stack: list[str] = []
        # Metal indirect-dispatch cache: (id(storage), offset) -> (gx, gy, gz).
        # Entries are invalidated whenever a subsequent dispatch binds the same storage buffer.
        self._indirect_cache: dict[tuple[int, int], tuple[int, int, int]] = {}
        self._indirect_cached_buffers: set[int] = set()
        # Indirect buffers already written in the current encoder that can be piggybacked
        # onto the next flush+wait (e.g. adam_dispatch_args prepared at train_step start).
        self._prefetch_indirect: dict[tuple[int, int], spy.NDBuffer] = {}

    @property
    def encoder(self) -> spy.CommandEncoder:
        return self._encoder

    def _invalidate_indirect_for_kwargs(self, kwargs: dict) -> None:
        if not self._indirect_cached_buffers:
            return
        for val in kwargs.values():
            vid = id(val)
            if vid in self._indirect_cached_buffers:
                self._indirect_cached_buffers.discard(vid)
                for key in [k for k in self._indirect_cache if k[0] == vid]:
                    del self._indirect_cache[key]

    def prefetch_indirect(self, args_buf: spy.NDBuffer, offset: int = 0) -> None:
        """Mark ``args_buf`` as already written in this encoder so the next Metal flush caches it."""
        if self._emulate_indirect:
            self._prefetch_indirect[(id(args_buf.storage), offset)] = args_buf

    def seed_indirect(self, args_buf: spy.NDBuffer, groups: tuple[int, int, int],
                      offset: int = 0) -> None:
        """Seed the Metal indirect-dispatch cache with host-known group counts (zero GPU sync)."""
        if self._emulate_indirect:
            buf_id = id(args_buf.storage)
            self._indirect_cache[(buf_id, offset)] = groups
            self._indirect_cached_buffers.add(buf_id)
            self._prefetch_indirect.pop((buf_id, offset), None)

    def dispatch(self, kernel, marker: str, **kwargs) -> None:
        if self._emulate_indirect:
            self._invalidate_indirect_for_kwargs(kwargs)
        self._encoder.insert_debug_marker(marker, spy.math.float3(*marker_color(marker)))
        kernel.dispatch(command_encoder=self._encoder, **kwargs)

    def dispatch_groups(self, kernel, marker: str, groups: tuple[int, int, int], **kwargs) -> None:
        """Record a direct compute dispatch with explicit threadgroup counts."""
        if self._emulate_indirect:
            self._invalidate_indirect_for_kwargs(kwargs)
        if groups[0] <= 0 or groups[1] <= 0 or groups[2] <= 0:
            return
        self._encoder.insert_debug_marker(marker, spy.math.float3(*marker_color(marker)))
        pass_enc = self._encoder.begin_compute_pass()
        try:
            shader_obj = pass_enc.bind_pipeline(kernel.pipeline)
            cursor = spy.ShaderCursor(shader_obj).find_entry_point(0)
            cursor.write(kwargs)
            pass_enc.dispatch_compute(list(groups))
        finally:
            pass_enc.end()

    def dispatch_indirect(self, kernel, marker: str, args_buf: spy.NDBuffer,
                          offset: int = 0, **kwargs) -> None:
        """Record an indirect dispatch; groups_x/y/z are uint32s at ``offset`` in ``args_buf``.

        No barrier before the fetch: sgl's resource tracking covers the ``args_buf`` write.
        """
        if self._emulate_indirect and self._runtime is not None:
            buf_id = id(args_buf.storage)
            key = (buf_id, offset)
            groups = self._indirect_cache.get(key)
            if groups is None:
                handle = self.flush_and_reopen(self._runtime)
                self._runtime.wait_for_submit(handle)
                groups = _read_indirect_groups(args_buf, offset)
                self._indirect_cache[key] = groups
                self._indirect_cached_buffers.add(buf_id)
                if self._prefetch_indirect:
                    for pkey, pbuf in list(self._prefetch_indirect.items()):
                        if pkey != key:
                            self._indirect_cache[pkey] = _read_indirect_groups(pbuf, pkey[1])
                            self._indirect_cached_buffers.add(pkey[0])
                    self._prefetch_indirect.clear()
            self.dispatch_groups(kernel, marker, groups, **kwargs)
            return
        self._encoder.insert_debug_marker(marker, spy.math.float3(*marker_color(marker)))
        pass_enc = self._encoder.begin_compute_pass()
        try:
            shader_obj = pass_enc.bind_pipeline(kernel.pipeline)
            cursor = spy.ShaderCursor(shader_obj).find_entry_point(0)
            cursor.write(kwargs)
            pass_enc.dispatch_compute_indirect(spy.BufferOffsetPair(args_buf.storage, offset))
        finally:
            pass_enc.end()

    def flush_and_reopen(self, runtime: GpuRuntime) -> SubmitHandle:
        """Submit work recorded so far and reopen a fresh encoder, preserving debug groups."""
        for _ in self._marker_stack:
            self._encoder.pop_debug_group()
        handle = runtime.submit(self)
        self._encoder = runtime.device.create_command_encoder()
        for label in self._marker_stack:
            self._encoder.push_debug_group(label, spy.math.float3(*marker_color(label)))
        return handle

    @contextlib.contextmanager
    def marker_group(self, label: str):
        self._marker_stack.append(label)
        self._encoder.push_debug_group(label, spy.math.float3(*marker_color(label)))
        try:
            yield
        finally:
            self._encoder.pop_debug_group()
            self._marker_stack.pop()

    def finish(self):
        return self._encoder.finish()


class GpuRuntime:
    """Owns the device and hands out CommandBatches; submissions serialize under ``gpu_lock``.

    The lock (shared with the viewer; a no-op by default) covers submits and ``wait_for_idle``,
    not single-submission fence waits or the viewer's ``present``.
    """

    def __init__(self, device: spy.Device,
                 gpu_lock: "contextlib.AbstractContextManager | None" = None) -> None:
        self.device = device
        self.gpu_lock = gpu_lock if gpu_lock is not None else contextlib.nullcontext()

    def command_batch(self, label: str) -> CommandBatch:
        return CommandBatch(self.device.create_command_encoder(), label, runtime=self)

    def submit(self, batch: CommandBatch,
               after_submit: Callable[[], None] | None = None) -> SubmitHandle:
        """Submit under the queue lock; ``after_submit`` (viewer present, serialized by sgl)
        runs after unlocking, so a stalled present cannot hold the queue against the trainer.
        """
        cmd = batch.finish()
        with self.gpu_lock:
            submit_id = self.device.submit_command_buffer(cmd)
        if after_submit is not None:
            after_submit()
        return SubmitHandle(submit_id)

    def wait_for_submit(self, handle: SubmitHandle | int) -> None:
        """Wait for one submission's fence; not under ``gpu_lock``, which would serialize
        the viewer and the trainer for the whole GPU wait.
        """
        submit_id = handle.submit_id if isinstance(handle, SubmitHandle) else handle
        self.device.wait_for_submit(submit_id)

    def wait_for_idle(self) -> None:
        """Queue-wide idle wait; a queue operation, so it takes the lock."""
        with self.gpu_lock:
            self.device.wait_for_idle()

    def queue_scope(self):
        """Lock scope for compound queue-adjacent operations (reload/reconfigure)."""
        return self.gpu_lock


class RecordingTransaction:
    """A recorded batch plus host-state commits that run once its submit succeeds."""

    def __init__(self, batch: CommandBatch) -> None:
        self.batch = batch
        self._commits: list[Callable[[], None]] = []

    def after_submit(self, commit: Callable[[], None]) -> None:
        self._commits.append(commit)

    def commit(self) -> None:
        for commit in self._commits:
            commit()


class CommandRecorder:
    """Records GPU work into the active batch, or submits each dispatch on its own.

    Shared by every GPU component, so a training step is one command buffer; ``timer``
    (a ``StepTimer``) is set only while profiling and timestamps every marker group.
    """

    def __init__(self, runtime: GpuRuntime) -> None:
        self.runtime = runtime
        self.batch: CommandBatch | None = None
        self.timer = None
        self.last_submit_id = 0
        # slang-rhi's Metal backend does not implement IComputePassEncoder::dispatchComputeIndirect yet.
        self._emulate_indirect = _is_metal_runtime(runtime)

    def dispatch(self, kernel, marker: str, **kwargs) -> None:
        if self.batch is not None:
            self.batch.dispatch(kernel, marker, **kwargs)
            return
        transient = self.runtime.command_batch(marker)
        transient.dispatch(kernel, marker, **kwargs)
        self.last_submit_id = self.runtime.submit(transient).submit_id

    def dispatch_indirect(self, kernel, marker: str, args_buf: spy.NDBuffer,
                          offset: int = 0, **kwargs) -> None:
        if self.batch is not None:
            self.batch.dispatch_indirect(kernel, marker, args_buf, offset, **kwargs)
            return
        if self._emulate_indirect:
            self.runtime.wait_for_idle()
            groups = _read_indirect_groups(args_buf, offset)
            transient = self.runtime.command_batch(marker)
            transient.dispatch_groups(kernel, marker, groups, **kwargs)
            self.last_submit_id = self.runtime.submit(transient).submit_id
            return
        transient = self.runtime.command_batch(marker)
        transient.dispatch_indirect(kernel, marker, args_buf, offset, **kwargs)
        self.last_submit_id = self.runtime.submit(transient).submit_id

    @contextlib.contextmanager
    def marker_group(self, label: str):
        if self.batch is None:
            yield
            return
        batch, timer = self.batch, self.timer
        interval = timer.open(label, batch.encoder) if timer is not None else -1
        with batch.marker_group(label):
            try:
                yield
            finally:
                if timer is not None:
                    timer.close(interval, batch.encoder)

    @contextlib.contextmanager
    def recording(self, label: str):
        """Record into a fresh batch and submit it once, only if the body succeeds."""
        batch = self.runtime.command_batch(label)
        transaction = RecordingTransaction(batch)
        self.batch = batch
        try:
            yield transaction
            self.last_submit_id = self.runtime.submit(batch).submit_id
            transaction.commit()
        finally:
            self.batch = None

    @contextlib.contextmanager
    def attached(self, batch: CommandBatch):
        """Record into a batch the caller owns and submits (the viewer frame)."""
        previous, self.batch = self.batch, batch
        try:
            yield
        finally:
            self.batch = previous

    def upload_texture(self, texture: spy.Texture, pixels) -> None:
        if self.batch is not None:
            self.batch.encoder.upload_texture_data(texture, 0, 0, pixels)
        else:
            texture.copy_from_numpy(pixels)
