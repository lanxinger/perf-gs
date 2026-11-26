# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""GPU-counted arena commits preserve live bits and leave dead slots alone."""

from __future__ import annotations

import numpy as np
import pytest
import slangpy as spy

from trainer.backend.compute.device import SHADERS_DIR, create_device
from trainer.backend.compute.gpu import CommandBatch, GpuRuntime


@pytest.fixture(params=["vulkan", "d3d12", "metal"], scope="module")
def copy_gpu(request):
    try:
        device = create_device(backend=request.param)
    except Exception as error:
        pytest.skip(f"backend unavailable: {error}")
    path = str(SHADERS_DIR / "densify/copy_active.slang")
    prepare = device.create_compute_kernel(device.load_program(path, ["prepareActiveCopyArgs"]))
    copy = device.create_compute_kernel(device.load_program(path, ["copyActiveRecords"]))
    yield device, GpuRuntime(device), prepare, copy
    device.close()


def _uint_buffer(device, values: np.ndarray, *, indirect: bool = False):
    usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
    if indirect:
        usage |= spy.BufferUsage.indirect_argument
    buffer = spy.NDBuffer(device=device, dtype="uint", shape=values.shape, usage=usage)
    buffer.copy_from_numpy(values)
    return buffer


@pytest.mark.parametrize("soa", [False, True])
@pytest.mark.parametrize("record_words", [14, 28, 23, 90])
@pytest.mark.parametrize("count", [0, 1, 17, 73, 99])
def test_copy_live_records_bit_exact(copy_gpu, soa, record_words, count) -> None:
    device, runtime, prepare, copy = copy_gpu
    capacity = 73
    source = np.random.default_rng(123).integers(
        0, 2**32, capacity * record_words, dtype=np.uint32)
    untouched = np.full(source.shape, 0xDEADBEEF, np.uint32)
    src = _uint_buffer(device, source)
    dst = _uint_buffer(device, untouched)
    count_buf = _uint_buffer(device, np.array([count], np.uint32))
    args = _uint_buffer(device, np.zeros(3, np.uint32), indirect=True)
    batch: CommandBatch = runtime.command_batch("test_active_copy")
    batch.dispatch(prepare, "prepare", thread_count=[1, 1, 1],
                   count_buf=count_buf.storage, dispatch_args=args.storage,
                   record_words=record_words, max_splats=capacity)
    batch.dispatch_indirect(copy, "copy", args, source=src.storage,
                            destination=dst.storage, count_buf=count_buf.storage,
                            record_words=record_words, soa=int(soa),
                            destination_word_offset=0, max_splats=capacity)
    runtime.wait_for_submit(runtime.submit(batch))
    expected = untouched.copy()
    live = min(count, capacity)
    if soa:
        expected.reshape(record_words, capacity)[:, :live] = \
            source.reshape(record_words, capacity)[:, :live]
    else:
        expected[:live * record_words] = source[:live * record_words]
    np.testing.assert_array_equal(dst.to_numpy().view(np.uint32), expected)


def test_copy_dispatch_exceeds_one_dimension(copy_gpu) -> None:
    device, runtime, prepare, _ = copy_gpu
    count = 1_000_000
    words = 90
    count_buf = _uint_buffer(device, np.array([count], np.uint32))
    args = _uint_buffer(device, np.zeros(3, np.uint32), indirect=True)
    batch = runtime.command_batch("test_copy_args")
    batch.dispatch(prepare, "prepare", thread_count=[1, 1, 1],
                   count_buf=count_buf.storage, dispatch_args=args.storage,
                   record_words=words, max_splats=count)
    runtime.wait_for_submit(runtime.submit(batch))
    groups = (count * words + 255) // 256
    np.testing.assert_array_equal(args.to_numpy(),
                                  np.array([65535, (groups + 65534) // 65535, 1], np.uint32))


@pytest.mark.parametrize("first_plane,planes", [(0, 28), (28, 28), (56, 28), (84, 6)])
@pytest.mark.parametrize("count", [0, 1, 17, 73, 99])
def test_copy_soa_batch_at_destination_offset(copy_gpu, first_plane, planes, count) -> None:
    device, runtime, prepare, copy = copy_gpu
    capacity, total_planes = 73, 90
    source = np.random.default_rng(7).integers(
        0, 2**32, capacity * planes, dtype=np.uint32)
    original = np.full(capacity * total_planes, 0xDEADBEEF, np.uint32)
    src, dst = _uint_buffer(device, source), _uint_buffer(device, original)
    count_buf = _uint_buffer(device, np.array([count], np.uint32))
    args = _uint_buffer(device, np.zeros(3, np.uint32), indirect=True)
    batch = runtime.command_batch("test_offset_copy")
    batch.dispatch(prepare, "prepare", thread_count=[1, 1, 1],
                   count_buf=count_buf.storage, dispatch_args=args.storage,
                   record_words=planes, max_splats=capacity)
    batch.dispatch_indirect(copy, "copy", args, source=src.storage,
                            destination=dst.storage, count_buf=count_buf.storage,
                            record_words=planes, soa=1,
                            destination_word_offset=first_plane * capacity,
                            max_splats=capacity)
    runtime.wait_for_submit(runtime.submit(batch))
    expected = original.reshape(total_planes, capacity).copy()
    live = min(count, capacity)
    expected[first_plane:first_plane + planes, :live] = source.reshape(planes, capacity)[:, :live]
    np.testing.assert_array_equal(dst.to_numpy().view(np.uint32), expected.ravel())