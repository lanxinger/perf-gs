# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Bound SH permutation scratch without changing any FP32 moment bits."""
from __future__ import annotations

import contextlib
from types import SimpleNamespace

import numpy as np
import pytest

from trainer.backend.compute.device import create_device
from trainer.config.config import DensifyConfig, StructureConfig
from trainer.backend.memory import buffers
from trainer.backend.densify.morton import MortonReorder
from trainer.backend.densify.simple_densifier import SimpleDensifier
from trainer.backend.memory.arena import ArenaCommitter
from trainer.backend.render.pipeline import TrainingPipeline
from trainer.common.sh import pack_features_rest, sh_rest_coeffs
from trainer.config.config_types import DensificationMode
from trainer.config.config import AppearanceConfig

GAUSSIAN_FLOAT_COUNT = 14
ADAM_STATE_FLOAT_COUNT = 28


def _types():
    def layout(stride):
        return SimpleNamespace(buffer_layout=SimpleNamespace(
            reflection=SimpleNamespace(stride=stride)))

    return SimpleNamespace(gaussian=layout(GAUSSIAN_FLOAT_COUNT * 4),
                           adam=layout(ADAM_STATE_FLOAT_COUNT * 4),
                           sh_moments=layout(8),
                           projected=layout(28), blend=layout(40),
                           gaussian_layout=None)


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
@pytest.mark.parametrize("capacity", [73, 10_000_000])
def test_scratch_bound_is_independent_of_sh_moment_width(degree, capacity) -> None:
    rest_floats = sh_rest_coeffs(degree) * 3
    assert buffers.scratch_resource_bytes(_types(), capacity, rest_floats) == capacity * 112


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_allocator_shrinks_scratch_not_persistent_moments(monkeypatch, degree) -> None:
    """Run the production allocator with metadata-only buffers, not GB of VRAM."""
    allocations = {}
    types = _types()

    class CountingAllocator:
        def __init__(self, *_args):
            pass

        def typed_buffer(self, label, dtype, *shape):
            stride = 8 if dtype == "int64_t" or dtype == "uint2" else 4
            if not isinstance(dtype, str):
                stride = dtype.buffer_layout.reflection.stride
            size = int(np.prod(shape)) * stride
            allocations[label] = size
            return SimpleNamespace(storage=SimpleNamespace(size=size),
                                   copy_from_numpy=lambda _values: None)

        def uint_buffer(self, label, *shape):
            return self.typed_buffer(label, "uint", *shape)

        def fixed_buffer(self, label, *shape):
            return self.typed_buffer(label, "int64_t", *shape)

        def indirect_buffer(self, label, records=1):
            return self.uint_buffer(label, 3 * records)

        def tex(self, *_args):
            return None

    monkeypatch.setattr(buffers, "_reflect_types", lambda _module: types)
    monkeypatch.setattr(buffers, "_Alloc", CountingAllocator)
    capacity = 10_000_000
    rest_floats = sh_rest_coeffs(degree) * 3
    allocated = buffers.allocate(None, None, capacity, 16, 16,
                                  max_sort_pairs=1, sh_rest_floats=rest_floats)

    assert allocated.densification.scratch_arena.capacity_bytes == 1_120_000_000
    assert allocations["densification.scratch_arena"] == 1_120_000_000
    if degree:
        assert allocations["sh.adam_rest"] == capacity * rest_floats * 8
        assert allocations["sh.features_rest"] == capacity * ((rest_floats + 1) // 2) * 4
        assert allocated.sh.rest_floats == rest_floats
    else:
        assert allocated.sh is None


def test_plane_batches_commit_before_reusing_scratch() -> None:
    """Each plane scatter must commit before another batch reuses its arena."""
    events = []
    capacity, planes = 73, 90
    arena = SimpleNamespace(capacity_bytes=capacity * 112, storage="arena",
                            copy_dispatch_args=SimpleNamespace(storage="args"))
    moments = SimpleNamespace(storage=SimpleNamespace(size=capacity * planes * 4))
    live_count = SimpleNamespace(storage="live")

    class Commands:
        batch = "active"

        def dispatch(self, kernel, _marker, **kwargs):
            if kernel == "scatter":
                events.append(("scatter", kwargs["first_plane"], kwargs["plane_count"]))
                assert kwargs["adam_rest_in"] is moments.storage
                assert kwargs["adam_rest_out"] == "arena"

        def dispatch_indirect(self, _kernel, _marker, _args, offset=0, **kwargs):
            assert kwargs["count_buf"] == "live" and kwargs["soa"] == 1
            events.append(("commit", kwargs["destination_word_offset"] // capacity,
                           kwargs["record_words"]))

        @contextlib.contextmanager
        def marker_group(self, _label):
            yield

    kernels = SimpleNamespace(load=lambda specs: SimpleNamespace(**{name: name for name in specs}))
    ArenaCommitter(Commands(), kernels, arena, capacity).reorder_planes(
        moments, "scatter", "reorder", planes, live_count)
    assert events == [(operation, first, count)
                      for first, count in [(0, 28), (28, 28), (56, 28), (84, 6)]
                      for operation in ("scatter", "commit")]


@pytest.fixture(params=[(backend, degree) for backend in ("vulkan", "d3d12", "metal")
                        for degree in (1, 2, 3)], scope="module")
def sh_pipeline(request):
    backend, degree = request.param
    try:
        device = create_device(backend=backend)
    except Exception as error:
        pytest.skip(f"backend unavailable: {error}")
    pipeline = TrainingPipeline(densify_mode=DensificationMode.STRUCTURE_AWARE, device=device, width=16, height=16,
                                max_splats=73, max_sort_pairs=1024,
                                appearance_config=AppearanceConfig(sh_degree=degree))
    yield pipeline
    device.close()


@pytest.mark.parametrize("kind", ["compaction", "morton"])
@pytest.mark.parametrize("live", [0, 1, 17, 73])
def test_batched_planes_preserve_all_bits_and_dead_tails(sh_pipeline, kind, live) -> None:
    pipeline = sh_pipeline
    b, capacity = pipeline.buffers, pipeline.max_splats
    planes = b.sh.rest_floats * 2
    rng = np.random.default_rng(1701)
    original = rng.integers(0, 2**32, (planes, capacity), dtype=np.uint32)
    # Include representations that must never be canonicalized as floating values.
    original[:, 0] = np.resize(np.array([0, 0x80000000, 0x7F800000,
                                        0x7FC01234, 1, 0xDEADBEEF], np.uint32), planes)
    b.sh.adam_rest.copy_from_numpy(original.ravel())
    b.scan.total.copy_from_numpy(np.array([live], np.uint32))
    if kind == "compaction":
        selected = np.sort(rng.choice(capacity, live, replace=False))
        mask = np.zeros(capacity, np.uint32)
        mask[selected] = 1
        scan = (np.cumsum(mask, dtype=np.uint32) - mask).astype(np.uint32)
        b.densification.valid_mask.copy_from_numpy(mask)
        b.scan.work.copy_from_numpy(scan)
        kernel = pipeline.kernels.kernel(*SimpleDensifier.KERNELS["scatter_compact_adam_rest"])
        permutation = dict(valid_mask=b.densification.valid_mask.storage, scan=b.scan.work.storage)
    else:
        selected = rng.permutation(live)
        indices = np.zeros(b.max_sort_pairs, np.uint32)
        indices[:live] = selected
        b.sort.values.copy_from_numpy(indices)
        kernel = pipeline.kernels.kernel(*MortonReorder.KERNELS["scatter_adam_rest"])
        permutation = dict(sorted_values=b.sort.values.storage, count_buf=b.scan.total.storage)
    expected = original.copy()
    expected[:, :live] = original[:, selected]

    arena = ArenaCommitter(pipeline.commands, pipeline.kernels,
                           b.densification.scratch_arena, capacity)
    with pipeline.commands.recording("test_batched_planes"):
        arena.reorder_planes(b.sh.adam_rest, kernel, kind, planes, b.scan.total, **permutation)
    pipeline.runtime.wait_for_idle()

    adam_stride = b.adam.state.storage.size // capacity
    assert b.densification.scratch_arena.capacity_bytes == capacity * adam_stride
    np.testing.assert_array_equal(b.sh.adam_rest.to_numpy().view(np.uint32), expected.ravel())


@pytest.mark.parametrize("kind", ["simple", "structure", "morton"])
def test_production_reorders_keep_every_coefficient_and_moment(sh_pipeline, kind) -> None:
    pipeline = sh_pipeline
    b, capacity = pipeline.buffers, pipeline.max_splats
    n = 37
    rng = np.random.default_rng(803)
    means = rng.uniform(-1, 1, (n, 3)).astype(np.float32)
    opacity = np.full(n, 0.7, np.float32)
    opacity[::5] = 0.001
    pipeline.load_snapshot(
        means=means, log_scales=np.log(np.full((n, 3), 0.02, np.float32)),
        rot_quats=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
        opacities=opacity, color_logits=np.repeat(np.arange(n, dtype=np.float32)[:, None], 3, axis=1))
    rest = rng.standard_normal((capacity, b.sh.rest_floats)).astype(np.float32)
    features = pack_features_rest(rest)
    moments = rng.integers(0, 2**32, (b.sh.rest_floats * 2, capacity), dtype=np.uint32)
    b.sh.features_rest.copy_from_numpy(features)
    b.sh.adam_rest.copy_from_numpy(moments.ravel())
    if kind == "simple":
        pipeline.simple_densifier.run(DensifyConfig(), allow_scale_prune=False)
    elif kind == "structure":
        pipeline.structure_densifier.densify(StructureConfig(opacity_cap=0))
    else:
        pipeline.morton.reorder(np.full(3, -2, np.float32), np.full(3, 2, np.float32))
    pipeline.runtime.wait_for_idle()
    count = int(b.primitive.active_count.to_numpy()[0])
    expected_count = n if kind == "morton" else int(np.count_nonzero(opacity >= 0.005))
    assert count == expected_count
    records = np.frombuffer(b.primitive.values.to_numpy().tobytes(), dtype=b.primitive.layout.dtype)
    source_slots = records["sh0"][:count, 0].astype(np.intp)
    assert np.unique(source_slots).size == count
    if kind != "morton":
        np.testing.assert_array_equal(source_slots, np.flatnonzero(opacity >= 0.005))
    expected_moments = moments.copy()
    expected_moments[:, :count] = moments[:, source_slots]
    np.testing.assert_array_equal(b.sh.adam_rest.to_numpy().view(np.uint32), expected_moments.ravel())
    expected_features = features.reshape(capacity, b.sh.feat_uints).copy()
    expected_features[:count] = expected_features[source_slots].copy()
    np.testing.assert_array_equal(b.sh.features_rest.to_numpy().view(np.uint32), expected_features.ravel())
