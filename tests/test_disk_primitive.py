# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Second-primitive architectural proof: disk forward/backward/generic Adam.

The disk uses the production ``blendForward<IPrimitiveSource>`` loop,
``IBlendablePrimitive`` reverse-mode autodiff, and
``AdamPass<IPrimitiveParameterStorage<7>>``. Its shaders import no Gaussian
module/type and its host resources are plain Disk2D + channel-major optimizer
planes, not GaussianBuffers or AdamState.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pytest
import slangpy as spy

from trainer.backend.compute.device import SHADERS_DIR, create_device
from trainer.backend.memory.layout import StructLayout

BACKENDS = ["vulkan", "d3d12", "metal"]
DISK_SHADER = SHADERS_DIR / "primitives" / "disk" / "vertical.slang"
CHANNELS = 7
FIXED_SCALE = 65536.0


def _device(backend: str):
    try:
        return create_device(backend=backend)
    except Exception as exc:
        pytest.skip(f"backend {backend!r} unavailable: {exc}")


def _kernel(device, entry: str):
    return device.create_compute_kernel(
        device.load_program(str(DISK_SHADER), [entry]))


def _disk_dtype(device):
    module = spy.Module.load_from_file(device, str(DISK_SHADER))
    return module.layout.find_type_by_name("Disk2D")


def _disk_coverage(row: np.ndarray, pixel: np.ndarray) -> float:
    radius = np.exp(row[2])
    distance = np.linalg.norm(pixel - row[:2])
    t = np.clip(1.0 - distance / radius, 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    return float((1.0 / (1.0 + np.exp(-row[3]))) * smooth)


def _objective(row, pixel, coverage_seed, color_seed) -> float:
    return _disk_coverage(row, pixel) * coverage_seed + float(np.dot(row[4:7], color_seed))


def _finite_difference(row, pixel, coverage_seed, color_seed, h=1e-3):
    gradient = np.zeros(CHANNELS, dtype=np.float64)
    for channel in range(CHANNELS):
        plus = row.astype(np.float64).copy()
        minus = row.astype(np.float64).copy()
        plus[channel] += h
        minus[channel] -= h
        gradient[channel] = (
            _objective(plus, pixel, coverage_seed, color_seed)
            - _objective(minus, pixel, coverage_seed, color_seed)
        ) / (2.0 * h)
    return gradient


def _buffer(device, dtype, shape, data=None):
    buffer = spy.NDBuffer(device=device, dtype=dtype, shape=shape)
    if data is not None:
        buffer.copy_from_numpy(np.ascontiguousarray(data))
    return buffer


def _primitive_buffer(device, rows):
    disk_type = _disk_dtype(device)
    layout = StructLayout.of(disk_type)
    packed = np.zeros(len(rows), dtype=layout.dtype)
    packed["center"] = rows[:, 0:2]
    packed["log_radius"] = rows[:, 2]
    packed["opacity"] = rows[:, 3]
    packed["color_value"] = rows[:, 4:7]
    buffer = _buffer(device, disk_type, (len(rows),))
    buffer.copy_from_numpy(np.ascontiguousarray(packed.view(np.uint8)))
    return buffer


def _read_primitives(device, buffer, count: int) -> np.ndarray:
    layout = StructLayout.of(_disk_dtype(device))
    rec = np.frombuffer(buffer.to_numpy().tobytes(), dtype=layout.dtype)[:count]
    return np.column_stack([rec["center"], rec["log_radius"][:, None],
                            rec["opacity"][:, None], rec["color_value"]])


def _float3_buffer(device, values: np.ndarray):
    buffer = _buffer(device, "float3", (len(values),))
    stride = buffer.storage.size // (len(values) * 4)
    if stride == 4:
        padded = np.zeros((len(values), 4), dtype=np.float32)
        padded[:, :3] = values
        buffer.copy_from_numpy(np.ascontiguousarray(padded))
    else:
        buffer.copy_from_numpy(np.ascontiguousarray(values.astype(np.float32)))
    return buffer


def test_disk_has_no_gaussian_dependency() -> None:
    disk_dir = Path(SHADERS_DIR) / "primitives" / "disk"
    text = "\n".join(path.read_text(encoding="utf-8") for path in disk_dir.glob("*.slang"))
    code = re.sub(r"//.*", "", text)
    assert "Gaussian" not in code
    assert "common/types.slang" not in code
    assert "primitives/gaussian" not in code


@pytest.mark.parametrize("backend", BACKENDS)
def test_disk_uses_shared_forward_blend(backend: str) -> None:
    device = _device(backend)
    rows = np.array([
        [8.0, 8.0, np.log(8.0), 0.5, 0.8, 0.2, 0.1],
        [11.0, 8.0, np.log(6.0), -0.2, 0.1, 0.7, 0.3],
        [8.0, 12.0, np.log(5.0), 0.9, 0.2, 0.1, 0.9],
    ], dtype=np.float32)
    pixels = np.array([[8.5, 8.5], [10.0, 9.0], [7.0, 11.0]], dtype=np.float32)
    primitive_count = len(rows)
    pixel_count = len(pixels)

    primitives = _primitive_buffer(device, rows)
    sorted_ids = _buffer(device, "uint", (primitive_count,), np.arange(primitive_count, dtype=np.uint32))
    pixel_buffer = _buffer(device, "float2", (pixel_count,), pixels)
    checkpoints = _buffer(device, "uint2", (256,))
    outputs = _buffer(device, "float4", (pixel_count,))
    processed = _buffer(device, "uint", (pixel_count,))

    _kernel(device, "diskForward").dispatch(
        thread_count=[pixel_count, 1, 1],
        primitives=primitives.storage,
        sorted_primitive_ids=sorted_ids.storage,
        pixels=pixel_buffer.storage,
        checkpoints=checkpoints.storage,
        outputs=outputs.storage,
        processed_out=processed.storage,
        primitive_count=primitive_count,
        pixel_count=pixel_count,
        extinction_hardness=1.0,
    )
    device.wait_for_idle()
    got = np.frombuffer(outputs.to_numpy().tobytes(), dtype=np.float32).reshape(-1, 4)[:pixel_count]

    expected = []
    for pixel in pixels:
        color = np.zeros(3, dtype=np.float64)
        transmittance = 1.0
        for row in rows:
            alpha = min(_disk_coverage(row, pixel), 0.99)
            if alpha < 1.0 / 255.0:
                continue
            color += row[4:7] * alpha * transmittance
            transmittance *= 1.0 - alpha
            if transmittance < 1e-4:
                break
        expected.append(np.r_[color, 1.0 - transmittance])
    np.testing.assert_allclose(got, np.asarray(expected), atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_disk_generic_backward_matches_finite_difference(backend: str) -> None:
    device = _device(backend)
    rows = np.array([
        [3.0, 4.0, np.log(5.0), 0.3, 0.2, 0.4, 0.6],
        [8.0, 2.0, np.log(4.0), -0.7, 0.7, 0.1, 0.3],
    ], dtype=np.float32)
    pixels = np.array([[4.2, 4.7], [7.0, 3.1]], dtype=np.float32)
    coverage_seeds = np.array([0.7, -0.35], dtype=np.float32)
    color_seeds = np.array([[0.1, -0.2, 0.4], [-0.3, 0.5, 0.2]], dtype=np.float32)

    primitives = _primitive_buffer(device, rows)
    pixel_buffer = _buffer(device, "float2", (len(rows),), pixels)
    coverage_buffer = _buffer(device, "float", (len(rows),), coverage_seeds)
    color_buffer = _float3_buffer(device, color_seeds)
    gradients = _buffer(device, "float", (len(rows) * CHANNELS,))

    _kernel(device, "diskBackward").dispatch(
        thread_count=[len(rows), 1, 1],
        primitives=primitives.storage,
        pixels=pixel_buffer.storage,
        coverage_seeds=coverage_buffer.storage,
        color_seeds=color_buffer.storage,
        gradients=gradients.storage,
        count=len(rows),
    )
    device.wait_for_idle()
    got = np.frombuffer(gradients.to_numpy().tobytes(), dtype=np.float32).reshape(-1, CHANNELS)[:len(rows)]
    expected = np.stack([
        _finite_difference(rows[i], pixels[i], coverage_seeds[i], color_seeds[i])
        for i in range(len(rows))
    ])
    np.testing.assert_allclose(got, expected, atol=3e-3, rtol=3e-2)


@pytest.mark.parametrize("backend", BACKENDS)
def test_disk_generic_adam_matches_numpy(backend: str) -> None:
    device = _device(backend)
    rows = np.array([
        [3.0, 4.0, np.log(5.0), 0.3, 0.2, 0.4, 0.6],
        [8.0, 2.0, np.log(4.0), -0.7, 0.7, 0.1, 0.3],
    ], dtype=np.float32)
    count = len(rows)
    stride = 8
    primitive_buffer = _primitive_buffer(device, rows)
    count_buffer = _buffer(device, "uint", (1,), np.array([count], dtype=np.uint32))

    raw_gradients = np.zeros((CHANNELS, stride), dtype=np.int64)
    for channel in range(CHANNELS):
        raw_gradients[channel, :count] = np.round(
            (0.2 + channel * 0.1 + np.arange(count) * 0.05) * FIXED_SCALE).astype(np.int64)
    gradient_buffer = _buffer(device, "int64_t", (CHANNELS * stride,), raw_gradients.reshape(-1))
    moment1 = _buffer(device, "float", (CHANNELS * stride,), np.zeros(CHANNELS * stride, np.float32))
    moment2 = _buffer(device, "float", (CHANNELS * stride,), np.zeros(CHANNELS * stride, np.float32))
    rates = np.array([1e-3, 1e-3, 3e-3, 2e-2, 5e-3, 5e-3, 5e-3], dtype=np.float32)

    _kernel(device, "diskAdam").dispatch(
        thread_count=[count, 1, 1],
        primitives=primitive_buffer.storage,
        gradients=gradient_buffer.storage,
        moment1=moment1.storage,
        moment2=moment2.storage,
        primitive_count=count_buffer.storage,
        learning_rates={"rates": rates.tolist()},
        cfg={"bias_correction1": 0.1, "bias_correction2": 0.001},
        inverse_pixels=1.0,
        channel_stride=stride,
    )
    device.wait_for_idle()

    got = _read_primitives(device, primitive_buffer, count)
    gradients = raw_gradients[:, :count].T.astype(np.float64) / FIXED_SCALE
    expected = rows.astype(np.float64).copy()
    m = 0.1 * gradients
    v = 0.001 * gradients * gradients
    expected -= rates * (m / 0.1) / (np.sqrt(v / 0.001) + 1e-8)
    np.testing.assert_allclose(got, expected, atol=2e-5, rtol=2e-5)

    cleared = np.frombuffer(gradient_buffer.to_numpy().tobytes(), dtype=np.int64).reshape(CHANNELS, stride)
    assert np.all(cleared[:, :count] == 0)
