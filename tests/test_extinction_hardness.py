# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Finite-difference validation + behavioral check of the HARDENED extinction
compositor.

Compiling with ``-DEXTINCTION_HARDNESS=k`` switches the blend + backward to
``transStep=(1-a)^k``, ``weight=1-(1-a)^k`` (k>1 = stronger occlusion culling;
Beer-Lambert with optical depth scaled by k). This test:

  1. Verifies the GENERALIZED backward's gradients against central finite
     differences (exactly like test_gradients, but with k compiled in) — this
     is the correctness proof for the rederived dC/dalpha telescoping.
  2. Confirms the forward is actually "harder": final transmittance
     ``T_hardened = T_base^k``, so the alpha channel (1-T) is >= the default's
     everywhere and strictly larger where splats accumulate.

on both backends. See shaders/core/interfaces.slang (ICompositor /
HardenedExtinction) and the ``#ifdef EXTINCTION_HARDNESS`` backward branch in
shaders/rasterize/tile_binned/backward_persplat.slang.
"""

from __future__ import annotations

import numpy as np
import pytest

from _pipeline_testkit import TestPipeline
from trainer.config.config import LossConfig
from trainer.data.dataset import simple_camera
from trainer.backend.compute.device import create_device

# Reuse the exact scene / target / parameter table the alpha-over finite-diff
# test uses, so this is a like-for-like check with only the compositor changed.
from test_gradients import _scene, _target, PARAM_TABLE

BACKENDS = ["vulkan", "d3d12", "metal"]
K = 2.0


def _camera(w: int, h: int):
    return simple_camera(
        width=w, height=h,
        eye=(0.0, 0.0, 0.0), target=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0),
        fovy_degrees=60.0,
    )


def _setup(backend: str, w: int, h: int, hardness: float):
    try:
        device = create_device(backend=backend)
    except Exception as exc:  # pragma: no cover - backend availability
        pytest.skip(f"backend {backend!r} unavailable: {exc}")
    pipeline = TestPipeline(
        device=device, width=w, height=h, max_splats=16,
        loss_config=LossConfig(ssim_weight=0.0, alpha_weight=0.0),
    )
    # Hardness is applied render-state now (the schedule owns it in production);
    # set it directly for the test's fixed-k forward/backward.
    pipeline.set_extinction_hardness(hardness)
    return pipeline


def _loss_for(pipeline: TestPipeline, scene: dict, camera) -> float:
    pipeline.set_primitive_from_numpy(**scene)
    pipeline.forward_and_loss(camera)
    return pipeline.read_loss()


@pytest.mark.parametrize("backend", BACKENDS)
def test_hardened_gradients_finite_difference(backend: str) -> None:
    W = H = 16
    pipeline = _setup(backend, W, H, K)
    camera = _camera(W, H)
    pipeline.targets.set_image(_target(H, W))
    scene = _scene()
    n = scene["means"].shape[0]

    pipeline.set_primitive_from_numpy(**scene)
    pipeline.forward_and_loss(camera)
    pipeline.optimizer.zero_grad()
    pipeline.run_backward()
    analytic = pipeline.read_gradients()

    eps = 1e-3
    mismatches: list[str] = []
    for grad_name, field_name, idx in PARAM_TABLE:
        col = analytic[grad_name]
        for gi in range(n):
            sp = {k: v.copy() for k, v in scene.items()}
            sm = {k: v.copy() for k, v in scene.items()}
            if grad_name == "opacity":
                p = float(scene[field_name][gi])
                p = min(max(p, 1e-5), 1.0 - 1e-5)
                bl = np.log(p / (1.0 - p))
                sp[field_name][gi] = 1.0 / (1.0 + np.exp(-(bl + eps)))
                sm[field_name][gi] = 1.0 / (1.0 + np.exp(-(bl - eps)))
            elif idx is None:
                sp[field_name][gi] += eps
                sm[field_name][gi] -= eps
            else:
                sp[field_name][gi, idx] += eps
                sm[field_name][gi, idx] -= eps

            numerical = (_loss_for(pipeline, sp, camera)
                         - _loss_for(pipeline, sm, camera)) / (2 * eps)
            a = float(col[gi])
            abs_diff = abs(numerical - a)
            denom = max(abs(numerical), abs(a), 1e-2)
            if abs_diff / denom > 0.10 and abs_diff > 2e-2:
                mismatches.append(
                    f"g{gi} {grad_name:14s}: analytic={a:+.4e} "
                    f"numerical={numerical:+.4e} rel_err={abs_diff / denom:.3f}")

    if mismatches:
        pytest.fail(f"{len(mismatches)} hardened-gradient mismatches:\n"
                    + "\n".join(mismatches))


@pytest.mark.parametrize("backend", BACKENDS)
def test_hardness_increases_extinction(backend: str) -> None:
    """k>1 must decay transmittance faster: since every splat's transStep is
    raised to k, the stacked T_hardened = (prod (1-a_i))^k = T_base^k, so the
    written alpha channel (1 - T) is >= the alpha-over default everywhere and
    strictly larger where coverage accumulates."""
    W = H = 32
    camera = _camera(W, H)
    scene = _scene()
    target = _target(H, W)

    base = _setup(backend, W, H, 1.0)
    base.targets.set_image(target)
    base.set_primitive_from_numpy(**scene)
    base.forward_and_loss(camera)
    base.device.wait_for_idle()
    a_base = base.buffers.raster.output_image.to_numpy()[..., 3].astype(np.float64)

    hard = _setup(backend, W, H, K)
    hard.targets.set_image(target)
    hard.set_primitive_from_numpy(**scene)
    hard.forward_and_loss(camera)
    hard.device.wait_for_idle()
    a_hard = hard.buffers.raster.output_image.to_numpy()[..., 3].astype(np.float64)

    assert np.all(a_hard >= a_base - 2e-3), \
        "hardened extinction reduced coverage somewhere (should be >= default)"
    covered = a_base > 0.05
    assert covered.sum() > 0, "test scene produced no coverage"
    assert a_hard[covered].mean() > a_base[covered].mean() + 1e-3, \
        "hardness did not increase mean coverage on covered pixels"

    # Closed form T_hardened = T_base^k on non-saturated covered pixels (away
    # from the early-break regime where both alphas round to ~1).
    mid = covered & (a_base < 0.95)
    if mid.sum() > 0:
        expected = 1.0 - np.power(1.0 - a_base[mid], K)
        np.testing.assert_allclose(a_hard[mid], expected, atol=1e-2)
