# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""What the viewer copies out of the trainer, and what it tells its rasterizer.

The snapshot is the trainer's only per-iteration cost for having a viewer open, so
it copies the ACTIVE splat range rather than the buffers' ``--max-splats``
allocation. That makes the published count safety-critical: a count ahead of the
copied range would have the viewer's rasterizer walk uninitialized tail memory.

Device-free — the encoder, the buffers and both pipelines are fakes, so what is
under test is the sizing arithmetic and the count discipline, not the GPU.
"""
from __future__ import annotations

import contextlib
from types import SimpleNamespace

import numpy as np

from trainer.frontend.viewer.session import Session
from trainer.frontend.viewer.viewer import Viewer, _copy_splats

GAUSSIAN_STRIDE = 56        # float3 mean + float3 log_scale + float4 quat + 2 floats
CAPACITY = 10_000_000


class _Buffer:
    def __init__(self, size: int) -> None:
        self.size = size


class _Encoder:
    def __init__(self) -> None:
        self.copies: list[tuple[_Buffer, int]] = []
        self.uploads: list[tuple[_Buffer, np.ndarray]] = []

    def copy_buffer(self, dst, dst_offset, src, src_offset, size):
        assert (dst_offset, src_offset) == (0, 0)
        self.copies.append((dst, size))

    def upload_buffer_data(self, buffer, offset, data):
        assert offset == 0
        self.uploads.append((buffer, np.asarray(data)))


# --------------------------------------------------------------------------
# _copy_splats: bytes, not elements
# --------------------------------------------------------------------------

def test_only_the_active_range_is_copied() -> None:
    encoder = _Encoder()
    src, dst = _Buffer(CAPACITY * GAUSSIAN_STRIDE), _Buffer(CAPACITY * GAUSSIAN_STRIDE)
    _copy_splats(encoder, src, dst, count=150_000, capacity=CAPACITY)
    assert encoder.copies == [(dst, 150_000 * GAUSSIAN_STRIDE)]
    # ... which is what the fix is for: a snapshot at 1.5% occupancy used to move
    # the whole 534 MiB allocation.
    assert encoder.copies[0][1] < src.size // 50


def test_the_stride_comes_from_each_buffers_own_size() -> None:
    """filter_3D is 4 B/splat and features_rest is half-packed — one helper, no
    layout knowledge in it."""
    encoder = _Encoder()
    for stride in (4, 56, 92):
        src, dst = _Buffer(CAPACITY * stride), _Buffer(CAPACITY * stride)
        _copy_splats(encoder, src, dst, count=1_000, capacity=CAPACITY)
        assert encoder.copies[-1] == (dst, 1_000 * stride)


def test_a_smaller_destination_is_never_overrun() -> None:
    encoder = _Encoder()
    src = _Buffer(CAPACITY * GAUSSIAN_STRIDE)
    dst = _Buffer(1_000 * GAUSSIAN_STRIDE)          # viewer configured smaller
    _copy_splats(encoder, src, dst, count=500_000, capacity=CAPACITY)
    assert encoder.copies == [(dst, dst.size)]


def test_an_empty_model_copies_nothing() -> None:
    encoder = _Encoder()
    _copy_splats(encoder, _Buffer(1024), _Buffer(1024), count=0, capacity=CAPACITY)
    assert encoder.copies == []


# --------------------------------------------------------------------------
# update_gaussians: the published count is the copied count
# --------------------------------------------------------------------------

class _Runtime:
    def __init__(self, encoder: _Encoder) -> None:
        self.encoder = encoder
        self.submitted = 0

    def command_batch(self, _label):
        return SimpleNamespace(encoder=self.encoder)

    def submit(self, _batch):
        self.submitted += 1

    def queue_scope(self):
        return contextlib.nullcontext()

    def wait_for_idle(self):
        pass


def _buffers(*, capacity: int = CAPACITY, sh: bool = False):
    """A stand-in for GaussianBuffers: the per-splat buffers + the counter."""
    return SimpleNamespace(
        max_splats=capacity,
        sh_enabled=sh,
        primitive=SimpleNamespace(
            values=SimpleNamespace(storage=_Buffer(capacity * GAUSSIAN_STRIDE)),
            filter_3d=SimpleNamespace(storage=_Buffer(capacity * 4)),
            active_count=SimpleNamespace(storage=_Buffer(4))),
        sh=SimpleNamespace(features_rest=SimpleNamespace(storage=_Buffer(capacity * 92))),
    )


def _snapshot(*, live: int, viewer_capacity: int = CAPACITY, sh: bool = False,
              mip: bool = False, active_sh_degree: int = 0, hardness: float = 1.0, vignetting: bool = False):
    """Run the real ``update_gaussians`` against fakes; return (encoder, viewer)."""
    encoder = _Encoder()
    pipeline = SimpleNamespace(
        buffers=_buffers(capacity=viewer_capacity, sh=sh),
        raster_config=SimpleNamespace(mip_enabled=mip),
        rasterizer=SimpleNamespace(natural_vignetting=False),
        set_active_sh_degree=lambda degree: None,
        set_extinction_hardness=lambda value: setattr(pipeline, "hardness", value),
        resize=lambda capacity, _pairs: setattr(
            pipeline, "buffers", _buffers(capacity=capacity, sh=sh)),
    )
    invalidations: list[int] = []
    viewer = SimpleNamespace(
        state=Session(),
        runtime=_Runtime(encoder),
        max_splats=viewer_capacity,
        _has_data=False,
        _pipeline=pipeline,
        _render_lock=contextlib.nullcontext(),
        _explicit_sort_pairs=None,
        _grow=lambda count, limit: Viewer._grow(viewer, count, limit),
        _copy_snapshot=lambda train, count: Viewer._copy_snapshot(viewer, train, count),
        live_splat_count=lambda pipeline: Viewer.live_splat_count(viewer, pipeline),
        # While training and parked on a test camera the render loop re-rasterizes
        # ONLY on this signal (see Viewer._scene_due), so a snapshot that failed to
        # raise it would leave the previous model on screen indefinitely.
        invalidate_scene=lambda: invalidations.append(1),
        invalidations=invalidations,
    )
    view_buffers = pipeline.buffers
    train = SimpleNamespace(rasterizer=SimpleNamespace(natural_vignetting=vignetting), buffers=_buffers(sh=sh), max_splats=CAPACITY,
                            primitives=SimpleNamespace(count=lambda: live),
                            active_sh_degree=active_sh_degree, extinction_hardness=hardness)
    Viewer.update_gaussians(viewer, train)
    return encoder, viewer, view_buffers


def test_the_count_the_viewer_is_given_is_the_count_that_was_copied() -> None:
    """NOT the trainer's counter: the readback can be a densify behind by the time
    the batch is recorded, and a larger count would render uninitialized memory."""
    encoder, viewer, view_buffers = _snapshot(live=150_000)
    assert encoder.uploads == [(view_buffers.primitive.active_count.storage,
                                np.array([150_000], dtype=np.uint32))]
    copied_bytes = dict(encoder.copies)[view_buffers.primitive.values.storage]
    assert copied_bytes == 150_000 * GAUSSIAN_STRIDE
    assert viewer._has_data is True


def test_a_snapshot_publishes_the_live_splat_count() -> None:
    """Densify keeps the count on the GPU, so the panel learns it from snapshots."""
    _encoder, viewer, _buffers_ = _snapshot(live=150_000)
    assert viewer.state.snapshot().stats.gaussians == 150_000


def test_the_counter_travels_in_the_same_batch_as_the_data() -> None:
    """One queue, so a submitted batch is atomic against viewer frames — the count
    and the splats it describes must not be two submissions."""
    encoder, viewer, _ = _snapshot(live=1_000)
    assert viewer.runtime.submitted == 1
    assert len(encoder.uploads) == 1 and encoder.copies


def test_a_count_beyond_the_viewers_capacity_grows_the_viewer() -> None:
    encoder, viewer, _ = _snapshot(live=5_000_000, viewer_capacity=1_000_000)
    assert viewer.max_splats == 5_000_000
    assert encoder.uploads[0][1].tolist() == [5_000_000]
    assert dict(encoder.copies)[viewer._pipeline.buffers.primitive.values.storage] == \
        5_000_000 * GAUSSIAN_STRIDE


def test_sh_and_mip_buffers_follow_the_same_count() -> None:
    encoder, _, view_buffers = _snapshot(live=2_000, sh=True, mip=True)
    sizes = dict(encoder.copies)
    assert sizes[view_buffers.primitive.values.storage] == 2_000 * GAUSSIAN_STRIDE
    assert sizes[view_buffers.sh.features_rest.storage] == 2_000 * 92
    assert sizes[view_buffers.primitive.filter_3d.storage] == 2_000 * 4


def test_showing_raw_points_clears_only_the_live_mip_filter():
    _, viewer, buffers = _snapshot(live=6, viewer_capacity=8, mip=True, vignetting=True)
    footprints = np.arange(1, 9, dtype=np.float32)
    def upload(values):
        footprints[:len(values)] = values
    buffers.primitive.filter_3d.storage.copy_from_numpy = upload
    viewer._pipeline.set_gaussians = lambda **kwargs: None
    count = 3
    points = {key: np.ones((count, lanes), dtype=np.float32)
              for key, lanes in (("means", 3), ("log_scales", 3), ("rot_quats", 4), ("color_logits", 3))}
    points["opacities"] = np.ones(count, dtype=np.float32)
    assert Viewer.preview_points(viewer, points) == count
    np.testing.assert_array_equal(footprints, [0, 0, 0, 4, 5, 6, 7, 8])
    assert viewer._pipeline.rasterizer.natural_vignetting is False


def test_nothing_is_copied_before_the_first_splat_exists() -> None:
    encoder, viewer, _ = _snapshot(live=0)
    assert encoder.copies == []
    assert encoder.uploads[0][1].tolist() == [0]
    assert viewer._has_data is False          # nothing to rasterize yet


def test_a_snapshot_invalidates_the_rendered_scene() -> None:
    """The scene budget (Viewer._scene_due) skips the rasterize while training
    unless something says the model changed. A snapshot IS that something — and
    parked on a test camera it is the only trigger there is, so forgetting it
    would freeze the picture for the whole run."""
    _, viewer, _ = _snapshot(live=1_000)
    assert viewer.invalidations == [1]


def test_snapshot_carries_nondefault_extinction_hardness() -> None:
    _, viewer, _ = _snapshot(live=1_000, hardness=2.5)
    assert viewer._pipeline.hardness == 2.5


def test_snapshot_restores_lens_response_after_a_viewer_grow():
    _, viewer, _ = _snapshot(live=15, viewer_capacity=10, vignetting=True)
    assert viewer._pipeline.rasterizer.natural_vignetting


def test_nongrowing_snapshot_holds_render_lock_and_updates_cached_count_after_submit() -> None:
    """Non-growing updates must hold ``_render_lock`` and publish ``cached_count`` only
    after submitting the GPU upload batch, so ``render_frame()`` never reads a new
    ``cached_count`` against an older GPU ``active_count``."""
    events: list[str] = []

    class _TrackingLock:
        def __enter__(self):
            events.append("lock_enter")
        def __exit__(self, *_exc):
            events.append("lock_exit")

    encoder = _Encoder()
    view_buffers = _buffers(capacity=100)
    view_buffers.primitive.cached_count = 50

    class _TrackingRuntime(_Runtime):
        def submit(self, batch):
            assert view_buffers.primitive.cached_count == 50, (
                "cached_count must not advance before the GPU upload batch is submitted")
            events.append("submit")
            super().submit(batch)

    pipeline = SimpleNamespace(
        buffers=view_buffers,
        raster_config=SimpleNamespace(mip_enabled=False),
        rasterizer=SimpleNamespace(natural_vignetting=False),
        set_active_sh_degree=lambda _d: None,
        set_extinction_hardness=lambda _v: None,
    )
    viewer = SimpleNamespace(
        state=Session(),
        runtime=_TrackingRuntime(encoder),
        max_splats=100,
        _has_data=False,
        _pipeline=pipeline,
        _render_lock=_TrackingLock(),
        _explicit_sort_pairs=None,
        _grow=lambda count, limit: Viewer._grow(viewer, count, limit),
        _copy_snapshot=lambda train, count: Viewer._copy_snapshot(viewer, train, count),
        live_splat_count=lambda p: Viewer.live_splat_count(viewer, p),
        invalidate_scene=lambda: events.append("invalidate"),
    )
    train = SimpleNamespace(
        rasterizer=SimpleNamespace(natural_vignetting=False),
        buffers=_buffers(capacity=100),
        max_splats=100,
        primitives=SimpleNamespace(count=lambda: 30),
        active_sh_degree=0,
        extinction_hardness=1.0,
    )
    Viewer.update_gaussians(viewer, train)
    assert view_buffers.primitive.cached_count == 30
    assert events == ["lock_enter", "submit", "invalidate", "lock_exit"]
