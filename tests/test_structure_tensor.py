# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Structure-tensor field: analytic properties of the reference, and GPU parity.

For a sinusoidal grating the gradient is purely along the grating's axis and its
magnitude grows with frequency, so the field must be dominated by that axis and
its top eigenvalue must grow with frequency. A flat image gives ~0.
"""

from __future__ import annotations

import numpy as np
import pytest

from structure_tensor_reference import EPS, principal_eigenvalue, structure_tensor_field
from trainer.backend.compute.device import create_device
from trainer.backend.densify.structure_tensor import StructureTensor

BACKENDS = ["vulkan", "d3d12", "metal"]


def _grating(h: int, w: int, omega: float, axis: str) -> np.ndarray:
    """RGB sinusoidal grating of ``omega`` cycles/pixel along ``axis``."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    coord = xx if axis == "x" else yy
    value = 0.5 + 0.5 * np.sin(2.0 * np.pi * omega * coord)
    return np.stack([value, value, value], axis=-1).astype(np.float32)


def _device(backend: str):
    try:
        return create_device(backend=backend)
    except Exception as exc:
        pytest.skip(f"backend {backend!r} unavailable: {exc}")


def _assert_field_close(reference: np.ndarray, gpu: np.ndarray) -> None:
    """float32 GPU vs float64 scipy; the field spans orders of magnitude."""
    assert gpu.shape == reference.shape
    assert gpu.dtype == np.float32
    scale = float(np.abs(reference).max()) + 1e-8
    max_abs = float(np.abs(gpu - reference).max())
    assert max_abs < 5e-3 * scale, (
        f"max abs diff {max_abs:.3e} exceeds {5e-3 * scale:.3e} (scale {scale:.3e})")
    np.testing.assert_allclose(gpu, reference, rtol=2e-2, atol=1e-3 * scale)


# -- reference properties ---------------------------------------------------------

def test_flat_image_has_negligible_tensor() -> None:
    field = structure_tensor_field(np.full((64, 64, 3), 0.5, dtype=np.float32))
    assert np.all(np.abs(field) < 1e-3), f"flat image tensor not ~0: max {np.abs(field).max()}"


@pytest.mark.parametrize("axis,component", [("x", 0), ("y", 2)])
def test_grating_axis_dominates(axis: str, component: int) -> None:
    core = structure_tensor_field(_grating(96, 96, omega=0.1, axis=axis))[24:72, 24:72]
    dominant = np.median(core[..., component])
    other = np.median(core[..., 2 - component])
    assert dominant > 10.0 * max(other, 1e-6), f"dominant {dominant:.3e} other {other:.3e}"


def test_principal_eigenvalue_grows_with_frequency() -> None:
    lambdas = [float(np.median(principal_eigenvalue(structure_tensor_field(
        _grating(128, 128, omega=omega, axis="x")))[32:96, 32:96]))
        for omega in (0.04, 0.08, 0.16)]
    assert lambdas[0] < lambdas[1] < lambdas[2], f"lambda1 not increasing: {lambdas}"


def test_output_shape_and_dtype() -> None:
    image = _grating(48, 72, omega=0.1, axis="x")
    field = structure_tensor_field(image)
    assert field.shape == (48, 72, 3) and field.dtype == np.float32
    rgba = np.concatenate([image, np.ones((48, 72, 1), np.float32)], axis=-1)
    assert structure_tensor_field(rgba).shape == (48, 72, 3)


def _stripe_wavelength(wavelength: float, size: int = 256, num_levels: int = 6) -> float:
    row = 0.5 + 0.5 * np.sin(2.0 * np.pi * np.arange(size) / wavelength)
    image = np.stack([np.tile(row, (size, 1))] * 3, axis=-1).astype(np.float32)
    lam1 = principal_eigenvalue(structure_tensor_field(image, num_levels=num_levels))
    core = lam1[size // 4:3 * size // 4, size // 4:3 * size // 4]
    return 1.0 / (float(np.sqrt(core.mean())) + EPS)


def test_wavelength_monotone_and_stable() -> None:
    """The recovered wavelength is a stable fraction (~0.25x, the SADGS convention)
    of the true one and increases with it; absolute recovery is not the contract."""
    wavelengths = [8.0, 12.0, 16.0, 20.0]
    recovered = [_stripe_wavelength(wl) for wl in wavelengths]
    assert all(a < b for a, b in zip(recovered, recovered[1:])), recovered
    ratios = np.array([r / wl for r, wl in zip(recovered, wavelengths)])
    assert ratios.max() / ratios.min() < 2.0, f"unstable proportionality: {ratios}"


# -- GPU parity ---------------------------------------------------------------------

@pytest.mark.parametrize("backend", BACKENDS)
def test_gpu_matches_reference_on_random_texture(backend: str) -> None:
    image = np.random.default_rng(0).random((40, 48, 3), dtype=np.float32)
    gpu = StructureTensor(_device(backend), 48, 40).compute(image)
    _assert_field_close(structure_tensor_field(image), gpu)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("axis", ["x", "y"])
def test_gpu_matches_reference_on_grating(backend: str, axis: str) -> None:
    image = _grating(64, 64, omega=0.1, axis=axis)
    reference = structure_tensor_field(image)
    gpu = StructureTensor(_device(backend), 64, 64).compute(image)
    _assert_field_close(reference, gpu)
    lam_ref = principal_eigenvalue(reference)[16:48, 16:48]
    lam_gpu = principal_eigenvalue(gpu)[16:48, 16:48]
    np.testing.assert_allclose(lam_gpu, lam_ref, rtol=3e-2,
                               atol=1e-3 * (float(np.abs(lam_ref).max()) + 1e-8))


@pytest.mark.parametrize("backend", BACKENDS)
def test_one_instance_serves_successive_views(backend: str) -> None:
    builder = StructureTensor(_device(backend), 40, 32)
    for omega in (0.05, 0.15):
        image = _grating(32, 40, omega=omega, axis="x")
        _assert_field_close(structure_tensor_field(image), builder.compute(image))
