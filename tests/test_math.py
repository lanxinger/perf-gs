# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Unit tests for Slang math utilities via SlangPy's functional API.

Check sigmoid, quaternion rotation, covariance and Gaussian evaluation on both backends.
"""

from __future__ import annotations

import numpy as np
import slangpy as spy


def test_sigmoid(math_module: spy.Module) -> None:
    xs = np.array([-10.0, -1.0, 0.0, 1.0, 10.0], dtype=np.float32)
    expected = 1.0 / (1.0 + np.exp(-xs))

    tensor = spy.Tensor.from_numpy(math_module.device, xs)
    result = math_module.sigmoidScalar(tensor).to_numpy()
    np.testing.assert_allclose(result, expected, atol=1e-6)


def _read_mat3(tensor: spy.Tensor) -> np.ndarray:
    """Read a batch of float3x3 values, respecting MSL's 16-byte float3 row stride on Metal."""
    raw = np.frombuffer(tensor.storage.to_numpy().tobytes(), dtype=np.float32)
    row_stride = raw.size // (tensor.shape[0] * 3)
    return raw.reshape(tensor.shape[0], 3, row_stride)[..., :3]


def test_quat_identity(math_module: spy.Module) -> None:
    q = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)  # (w, x, y, z) identity
    tensor = spy.Tensor.from_numpy(math_module.device, q)
    result = _read_mat3(math_module.rotMatFromQuat(tensor))[0]
    np.testing.assert_allclose(result, np.eye(3, dtype=np.float32), atol=1e-6)


def test_quat_90deg_z(math_module: spy.Module) -> None:
    # 90° rotation about +Z: q = (cos(45°), 0, 0, sin(45°)) in (w, x, y, z)
    half = np.sqrt(0.5).astype(np.float32)
    q = np.array([[half, 0.0, 0.0, half]], dtype=np.float32)
    expected = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
    ], dtype=np.float32)
    tensor = spy.Tensor.from_numpy(math_module.device, q)
    result = _read_mat3(math_module.rotMatFromQuat(tensor))[0]
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_cov_identity(math_module: spy.Module) -> None:
    scale = spy.Tensor.from_numpy(math_module.device, np.array([[1.0, 1.0, 1.0]], dtype=np.float32))
    quat = spy.Tensor.from_numpy(math_module.device, np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
    result = _read_mat3(math_module.covFromScaleQuat(scale, quat))[0]
    np.testing.assert_allclose(result, np.eye(3, dtype=np.float32), atol=1e-6)


def test_cov_anisotropic(math_module: spy.Module) -> None:
    scale = spy.Tensor.from_numpy(math_module.device, np.array([[2.0, 1.0, 1.0]], dtype=np.float32))
    quat = spy.Tensor.from_numpy(math_module.device, np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
    expected = np.diag([4.0, 1.0, 1.0]).astype(np.float32)
    result = _read_mat3(math_module.covFromScaleQuat(scale, quat))[0]
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_gaussian2d_at_mean(math_module: spy.Module) -> None:
    """alpha at the mean of an identity-covariance gaussian is ``opacity``."""
    pixel = spy.Tensor.from_numpy(math_module.device, np.array([[0.0, 0.0]], dtype=np.float32))
    mean = spy.Tensor.from_numpy(math_module.device, np.array([[0.0, 0.0]], dtype=np.float32))
    cov = spy.Tensor.from_numpy(math_module.device, np.array([[1.0, 0.0, 1.0]], dtype=np.float32))
    opacity = spy.Tensor.from_numpy(math_module.device, np.array([1.0], dtype=np.float32))
    result = math_module.evalGaussian2D(pixel, mean, cov, opacity).to_numpy()[0]
    np.testing.assert_allclose(result, 1.0, atol=1e-5)


def test_gaussian2d_one_sigma(math_module: spy.Module) -> None:
    """alpha 1 unit out along x of identity-cov gaussian = ``opacity * exp(-0.5)``."""
    pixel = spy.Tensor.from_numpy(math_module.device, np.array([[1.0, 0.0]], dtype=np.float32))
    mean = spy.Tensor.from_numpy(math_module.device, np.array([[0.0, 0.0]], dtype=np.float32))
    cov = spy.Tensor.from_numpy(math_module.device, np.array([[1.0, 0.0, 1.0]], dtype=np.float32))
    opacity = spy.Tensor.from_numpy(math_module.device, np.array([1.0], dtype=np.float32))
    result = math_module.evalGaussian2D(pixel, mean, cov, opacity).to_numpy()[0]
    np.testing.assert_allclose(result, np.exp(-0.5), atol=1e-5)
