# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Opaque-core splat kernels (``--splat-kernel``, ``--splat-profile-power``).

The density kernels read a splat's footprint as an optical depth, so alpha
saturates over a core instead of only at the peak; a profile power above 1
flattens that core and shortens its rim. Checks the closed forms, that the
binning footprint still ends exactly at the 1/255 blend floor, the backward's
gradients against finite differences, and that opaque cores actually form.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import slangpy as spy
from _pipeline_testkit import TestPipeline
from test_gradients import _scene, _target, finite_difference_mismatches

from trainer.backend.compute.device import create_device, create_slang_session
from trainer.config.config import LossConfig, ShaderSpecialization
from trainer.config.config_types import SplatKernel
from trainer.data.dataset import simple_camera
from trainer.frontend.factory import shader_defines

BACKENDS = ["vulkan", "d3d12", "metal"]
# (kernel, profile power): the default, each density kernel, and a flat-top profile
# with and without optical depth.
KERNELS = [(SplatKernel.GAUSSIAN, 1.0), (SplatKernel.DENSITY, 1.0),
           (SplatKernel.LOG_DENSITY, 1.0), (SplatKernel.GAUSSIAN, 2.0),
           (SplatKernel.DENSITY, 2.0), (SplatKernel.OPAQUE_CORE, 3.0)]
KERNEL_IDS = [f"{kernel.value}-p{power:g}" for kernel, power in KERNELS]
BLEND_FLOOR = 1.0 / 255.0


def _device(backend: str, kernel: SplatKernel, power: float = 1.0, onset: float = 0.0):
    defines = shader_defines(ShaderSpecialization(splat_kernel=kernel,
                                                  splat_profile_power=power,
                                                  splat_profile_onset=onset))
    try:
        return create_device(backend=backend, defines=defines)
    except Exception as exc:  # pragma: no cover - backend availability
        pytest.skip(f"backend {backend!r} unavailable: {exc}")


def _math(backend: str, kernel: SplatKernel, power: float, onset: float = 0.0) -> spy.Module:
    return spy.Module.load_from_file(_device(backend, kernel, power, onset),
                                     str(Path(__file__).with_name("shaders") / "math_tests.slang"))


def _call(module: spy.Module, name: str, *arrays) -> np.ndarray:
    tensors = [spy.Tensor.from_numpy(module.device, np.asarray(a, np.float32))
               for a in arrays]
    return getattr(module, name)(*tensors).to_numpy()


def _expected_alpha(kernel: SplatKernel, power: float, opacity: np.ndarray,
                    radius: np.ndarray, onset: float = 0.0) -> np.ndarray:
    if kernel is SplatKernel.OPAQUE_CORE:
        ramp = opacity / (opacity + (1.0 - opacity) * np.exp(onset))  # sigmoid(l - onset)
        power = 1.0 + (power - 1.0) * ramp
    falloff = np.exp(-(0.5 * radius ** 2) ** power)
    if kernel is SplatKernel.DENSITY:
        return 1.0 - (1.0 - opacity) ** falloff
    if kernel in (SplatKernel.LOG_DENSITY, SplatKernel.OPAQUE_CORE):
        return 1.0 - np.exp(-opacity / (1.0 - opacity) * falloff)
    return opacity * falloff


def _camera(size: int):
    return simple_camera(width=size, height=size, eye=(0.0, 0.0, 0.0),
                         target=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0),
                         fovy_degrees=60.0)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(("kernel", "power"), KERNELS, ids=KERNEL_IDS)
def test_alpha_matches_the_closed_form(backend: str, kernel: SplatKernel,
                                       power: float) -> None:
    module = _math(backend, kernel, power)
    opacity, radius = np.meshgrid([0.05, 0.5, 0.9, 0.999], [0.0, 0.5, 1.0, 2.0, 3.0])
    opacity, radius = opacity.ravel(), radius.ravel()
    n = opacity.size
    pixels = np.stack([radius, np.zeros(n)], axis=1)
    alpha = _call(module, "evalGaussian2D", pixels, np.zeros((n, 2)),
                  np.tile([1.0, 0.0, 1.0], (n, 1)), opacity)

    np.testing.assert_allclose(alpha, _expected_alpha(kernel, power, opacity, radius),
                               rtol=1e-4, atol=1e-6)
    if kernel not in (SplatKernel.LOG_DENSITY, SplatKernel.OPAQUE_CORE):
        center = radius == 0.0
        np.testing.assert_allclose(alpha[center], opacity[center], rtol=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(("kernel", "power"), KERNELS, ids=KERNEL_IDS)
def test_footprint_ends_at_the_blend_floor(backend: str, kernel: SplatKernel,
                                           power: float) -> None:
    """SnugBox's right edge, on the major axis of an axis-aligned splat, is the
    point where the kernel's alpha falls to exactly 1/255 — for every kernel,
    including near-opaque splats whose footprint reaches far past 3 sigma."""
    module = _math(backend, kernel, power)
    opacity = np.array([0.02, 0.5, 0.99, 0.99999])
    n = opacity.size
    cov = np.tile([4.0, 0.0, 1.0], (n, 1))
    box = _call(module, "snugBoxTest", np.zeros((n, 2)), cov, opacity)
    edge = np.stack([box[:, 2], np.zeros(n)], axis=1)
    alpha = _call(module, "evalGaussian2D", edge, np.zeros((n, 2)), cov, opacity)
    np.testing.assert_allclose(alpha, BLEND_FLOOR, rtol=1e-3)


@pytest.mark.parametrize("backend", BACKENDS)
def test_onset_delays_opaque_core_sharpening(backend: str) -> None:
    module = _math(backend, SplatKernel.OPAQUE_CORE, 3.0, onset=2.0)
    opacity, radius = np.meshgrid([0.05, 0.5, 0.9, 0.999], [0.0, 0.5, 1.0, 2.0, 3.0])
    opacity, radius = opacity.ravel(), radius.ravel()
    n = opacity.size
    alpha = _call(module, "evalGaussian2D", np.stack([radius, np.zeros(n)], axis=1),
                  np.zeros((n, 2)), np.tile([1.0, 0.0, 1.0], (n, 1)), opacity)
    np.testing.assert_allclose(
        alpha, _expected_alpha(SplatKernel.OPAQUE_CORE, 3.0, opacity, radius, onset=2.0),
        rtol=1e-4, atol=1e-6)

    floor_opacity = np.array([0.02, 0.5, 0.99, 0.99999])
    cov = np.tile([4.0, 0.0, 1.0], (floor_opacity.size, 1))
    box = _call(module, "snugBoxTest", np.zeros((floor_opacity.size, 2)), cov, floor_opacity)
    edge = np.stack([box[:, 2], np.zeros(floor_opacity.size)], axis=1)
    np.testing.assert_allclose(
        _call(module, "evalGaussian2D", edge, np.zeros((floor_opacity.size, 2)), cov,
              floor_opacity),
        BLEND_FLOOR, rtol=1e-3)


def _large_splat_scene() -> dict:
    """test_gradients' scene with every splat 4x larger, so each one's 1/255
    contour lies outside the 16x16 frame. The blend floor is a step in alpha;
    with small splats a finite difference can straddle it on some pixel and
    read a jump instead of a slope, which is geometry luck, not a gradient bug.
    The smooth large splats also take a 10x larger step, which keeps the loss's
    float32 rounding out of the finite difference."""
    scene = _scene()
    scene["log_scales"] = scene["log_scales"] + np.log(4.0).astype(np.float32)
    return scene


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(("kernel", "power"), KERNELS, ids=KERNEL_IDS)
def test_gradients_finite_difference(backend: str, kernel: SplatKernel,
                                     power: float) -> None:
    pipeline = TestPipeline(device=_device(backend, kernel, power), width=16, height=16,
                            max_splats=16,
                            loss_config=LossConfig(ssim_weight=0.0, alpha_weight=0.0))
    pipeline.targets.set_image(_target(16, 16))
    mismatches = finite_difference_mismatches(pipeline, _camera(16), _large_splat_scene(),
                                              eps=1e-2)
    assert not mismatches, "\n".join(mismatches)


def _opaque_stack(device, **pipeline_kwargs) -> np.ndarray:
    """A render of ten nearly opaque splats stacked along the view axis."""
    count = 10
    pipeline = TestPipeline(device=device, width=16, height=16, max_splats=16,
                            **pipeline_kwargs)
    pipeline.set_primitive_from_numpy(
        means=np.stack([np.zeros(count), np.zeros(count),
                        np.linspace(2.0, 3.0, count)], axis=1).astype(np.float32),
        log_scales=np.log(np.full((count, 3), 0.4, np.float32)),
        rot_quats=np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (count, 1)),
        opacities=np.full(count, 0.99, np.float32),
        color_logits=np.zeros((count, 3), np.float32))
    pipeline.render(_camera(16))
    return pipeline.read_output_image()


def _early_out_fraction(backend: str, kernel: SplatKernel) -> float:
    """Share of pixels the opaque stack turns opaque (transmittance under the
    forward's 1e-4 early-out)."""
    image = _opaque_stack(_device(backend, kernel))
    return float(np.mean(image[..., 3] > 1.0 - 1e-4))


@pytest.mark.parametrize("backend", BACKENDS)
def test_density_grows_opaque_cores(backend: str) -> None:
    gaussian = _early_out_fraction(backend, SplatKernel.GAUSSIAN)
    density = _early_out_fraction(backend, SplatKernel.DENSITY)
    assert density > 1.5 * gaussian > 0.0, (density, gaussian)


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_run_session_compiles_like_a_device_with_its_defines(backend: str) -> None:
    """A hub run picks its kernel in its own Slang session on the hub's device."""
    spec = ShaderSpecialization(splat_kernel=SplatKernel.OPAQUE_CORE, splat_profile_power=3.0)
    hub_device = _device(backend, SplatKernel.GAUSSIAN)
    session = create_slang_session(hub_device, shader_defines(spec))

    in_session = _opaque_stack(hub_device, slang_session=session)
    np.testing.assert_array_equal(
        in_session, _opaque_stack(_device(backend, SplatKernel.OPAQUE_CORE, 3.0)))
    assert not np.array_equal(in_session, _opaque_stack(hub_device))
