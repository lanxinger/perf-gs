# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Opacity/scale regularization gradient test.

Validates shaders/losses/regularization.slang, which had NO coverage before.
The reg implements 3DGS L_reg = w * mean_i(sigmoid(opacity_i)); the per-splat
opacity-logit gradient is therefore

    d L_reg / d logit_i = w_opacity * sigmoid(o_i) * (1 - sigmoid(o_i)) / n

where n is the LIVE gaussian count (the /n is the scale-stable normalization —
previously the kernel used inv_n=1.0, a SUM penalty). This test pins the
formula, sign, and the /n normalization so the mean-vs-sum convention can't
silently regress.
"""
from __future__ import annotations

import numpy as np
import pytest

from _pipeline_testkit import TestPipeline
from trainer.config.config import LossConfig, RegularizationConfig
from trainer.backend.compute.device import create_device


BACKENDS = ["vulkan", "d3d12", "metal"]


def _scene():
    # Opacities span low..high so sigmoid(o)*(1-sigmoid(o)) varies per splat.
    means = np.array([[0.0, 0.0, 3.0], [0.2, 0.1, 2.5], [-0.15, -0.1, 2.0]],
                     dtype=np.float32)
    log_scales = np.log(np.array([[0.3, 0.3, 0.3], [0.2, 0.2, 0.2],
                                  [0.25, 0.25, 0.25]], dtype=np.float32))
    rot_quats = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float32)
    opacities = np.array([0.05, 0.5, 0.9], dtype=np.float32)   # probabilities
    color_logits = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], dtype=np.float32)
    return dict(means=means, log_scales=log_scales, rot_quats=rot_quats,
                opacities=opacities, color_logits=color_logits)


@pytest.mark.parametrize("backend", BACKENDS)
def test_opacity_reg_gradient(backend: str):
    try:
        device = create_device(backend=backend)
    except Exception as exc:
        pytest.skip(f"backend {backend!r} unavailable: {exc}")

    W_OPACITY = 0.03
    p = TestPipeline(
        device=device, width=16, height=16, max_splats=16,
        regularization_config=RegularizationConfig(
            opacity_weight=W_OPACITY, scale_weight=0.0),
        loss_config=LossConfig(ssim_weight=0.0, alpha_weight=0.0),
    )
    scene = _scene()
    p.set_primitive_from_numpy(**scene)

    # Isolate the reg gradient: clear all grads, then run ONLY regularization.
    p.optimizer.zero_grad()
    p.regularize()
    device.wait_for_idle()

    grad_opacity = p.read_gradients()["opacity"]
    n = scene["opacities"].shape[0]

    # Expected: w * sigmoid(o)*(1-sigmoid(o)) / n. The GPU stores the pre-sigmoid
    # LOGIT; set_primitive_from_numpy took probabilities, so sigmoid(logit) == p.
    prob = scene["opacities"].astype(np.float64)
    expected = W_OPACITY * prob * (1.0 - prob) / n

    # Fixed-point (2^16) + float; generous abs floor for the tiny magnitudes.
    np.testing.assert_allclose(grad_opacity[:n], expected, rtol=1e-2, atol=1e-5)

    # Sign sanity: reg gradient is POSITIVE (pushes logit down -> opacity -> 0).
    assert np.all(grad_opacity[:n] > 0.0), "opacity reg gradient must be positive"


@pytest.mark.parametrize("backend", BACKENDS)
def test_opacity_reg_off_is_zero(backend: str):
    # With opacity_reg=0, _run_regularization early-outs -> no grad added.
    try:
        device = create_device(backend=backend)
    except Exception as exc:
        pytest.skip(f"backend {backend!r} unavailable: {exc}")
    p = TestPipeline(
        device=device, width=16, height=16, max_splats=16,
        regularization_config=RegularizationConfig(),
        loss_config=LossConfig(ssim_weight=0.0, alpha_weight=0.0),
    )
    scene = _scene()
    p.set_primitive_from_numpy(**scene)
    p.optimizer.zero_grad()
    p.regularize()
    device.wait_for_idle()
    n = scene["opacities"].shape[0]
    assert np.all(p.read_gradients()["opacity"][:n] == 0.0)
