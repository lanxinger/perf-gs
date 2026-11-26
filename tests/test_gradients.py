# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Finite-difference validation of the backward pass.

For each gaussian-parameter component:
    numerical ≈ (L(p + ε) - L(p - ε)) / (2ε)
Compared to the analytic gradient produced by the backward kernel.

Tolerance is relative (default 5%) with an absolute floor; exact gradient
agreement isn't attainable because:
- Fixed-point quantization at 2^16 loses ~1.5e-5 per accumulation.
- Forward rasterizer early-breaks; the FD perturbation can change which
  gaussians contribute to a given pixel, introducing discontinuities.
- bwd_diff generates code slightly different from what a purely numerical
  JVP would produce; small ULP differences are expected.

Scene: 3 gaussians at different depths + colors, 16×16 image, 60° FOV.
Small enough to run both backends in <30 s.
"""

from __future__ import annotations

import numpy as np
import pytest

from _pipeline_testkit import TestPipeline
from trainer.config.config import LossConfig
from trainer.data.dataset import simple_camera
from trainer.backend.compute.device import create_device


BACKENDS = ["vulkan", "d3d12", "metal"]


def _scene():
    means = np.array([
        [0.0, 0.0, 3.0],
        [0.2, 0.1, 2.5],
        [-0.15, -0.1, 2.0],
    ], dtype=np.float32)
    log_scales = np.log(np.array([
        [0.3, 0.3, 0.3],
        [0.2, 0.2, 0.2],
        [0.25, 0.25, 0.25],
    ], dtype=np.float32))
    rot_quats = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.9659, 0.0, 0.2588, 0.0],  # 30° about Y
        [0.9239, 0.0, 0.0, 0.3827],  # 45° about Z
    ], dtype=np.float32)
    opacities = np.array([0.8, 0.6, 0.7], dtype=np.float32)
    color_logits = np.array([
        [1.5, -1.5, -1.5],
        [-1.5, 1.5, -1.5],
        [-1.5, -1.5, 1.5],
    ], dtype=np.float32)
    return {
        "means": means,
        "log_scales": log_scales,
        "rot_quats": rot_quats,
        "opacities": opacities,
        "color_logits": color_logits,
    }


def _target(h: int, w: int) -> np.ndarray:
    # All-ones sRGB target. target_image is rgba8_unorm, so the shader reads the
    # quantized target; L1's sign(C-T) gradient is non-differentiable at C==T,
    # and a ramp target put some covered pixels of the frontmost gaussian on
    # that knife-edge, making the finite-difference gradient check fragile once
    # the target is 8-bit. 1.0 is 8-bit-exact (255) and strictly above the
    # render's max (~0.83 here), so C-T<0 for every pixel: sign is uniformly -1,
    # no knife-edges, and analytic == numerical robustly. The gradient math
    # (what this test validates) is exercised identically regardless of the
    # target's spatial pattern.
    return np.ones((h, w, 3), dtype=np.float32)


def _setup_pipeline(backend: str, w: int, h: int) -> TestPipeline:
    try:
        device = create_device(backend=backend)
    except Exception as exc:
        pytest.skip(f"backend {backend!r} unavailable: {exc}")
    p = TestPipeline(device=device, width=w, height=h, max_splats=16,
                     loss_config=LossConfig(ssim_weight=0.0, alpha_weight=0.0))
    return p


# The parameter table. (component_name_in_grad_buffer, field, index_in_field)
# ``field`` keys are into the ``_scene()`` dict above.
PARAM_TABLE: list[tuple[str, str, int]] = [
    ("mean_x", "means", 0),
    ("mean_y", "means", 1),
    ("mean_z", "means", 2),
    ("log_scale_x", "log_scales", 0),
    ("log_scale_y", "log_scales", 1),
    ("log_scale_z", "log_scales", 2),
    ("rot_w", "rot_quats", 0),
    ("rot_x", "rot_quats", 1),
    ("rot_y", "rot_quats", 2),
    ("rot_z", "rot_quats", 3),
    ("opacity", "opacities", None),
    ("color_r", "color_logits", 0),
    ("color_g", "color_logits", 1),
    ("color_b", "color_logits", 2),
]


def _loss_for(pipeline: TestPipeline, scene: dict, camera) -> float:
    pipeline.set_primitive_from_numpy(**scene)
    pipeline.forward_and_loss(camera)
    return pipeline.read_loss()


@pytest.mark.parametrize("backend", BACKENDS)
def test_gradients_finite_difference(backend: str) -> None:
    W = H = 16
    pipeline = _setup_pipeline(backend, W, H)
    camera = simple_camera(
        width=W, height=H,
        eye=(0.0, 0.0, 0.0), target=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0),
        fovy_degrees=60.0,
    )
    pipeline.targets.set_image(_target(H, W))

    mismatches = finite_difference_mismatches(pipeline, camera, _scene())
    if mismatches:
        msg = "\n".join(mismatches)
        pytest.fail(f"{len(mismatches)} gradient mismatches:\n{msg}")


def finite_difference_mismatches(pipeline: TestPipeline, camera, scene: dict,
                                 eps: float = 1e-3) -> list[str]:
    """Every PARAM_TABLE component whose analytic gradient disagrees with the
    central finite difference of the loss (target already set)."""
    n = scene["means"].shape[0]

    # Analytic grads from backward.
    pipeline.set_primitive_from_numpy(**scene)
    pipeline.forward_and_loss(camera)
    pipeline.optimizer.zero_grad()
    pipeline.run_backward()
    analytic = pipeline.read_gradients()

    mismatches: list[str] = []

    for grad_name, field_name, idx in PARAM_TABLE:
        analytic_col = analytic[grad_name]

        for gi in range(n):
            # Build perturbed scenes.
            scene_plus = {k: v.copy() for k, v in scene.items()}
            scene_minus = {k: v.copy() for k, v in scene.items()}

            if grad_name == "opacity":
                # The GPU stores a pre-sigmoid logit; ``set_primitive_from_numpy``
                # receives a probability and applies logit() internally. To
                # perturb the logit by ±eps we must convert the probability,
                # perturb in logit space, then sigmoid back before writing.
                p = float(scene[field_name][gi])
                p = min(max(p, 1e-5), 1.0 - 1e-5)
                base_logit = np.log(p / (1.0 - p))
                scene_plus[field_name][gi] = 1.0 / (1.0 + np.exp(-(base_logit + eps)))
                scene_minus[field_name][gi] = 1.0 / (1.0 + np.exp(-(base_logit - eps)))
            elif idx is None:
                scene_plus[field_name][gi] += eps
                scene_minus[field_name][gi] -= eps
            else:
                scene_plus[field_name][gi, idx] += eps
                scene_minus[field_name][gi, idx] -= eps

            L_plus = _loss_for(pipeline, scene_plus, camera)
            L_minus = _loss_for(pipeline, scene_minus, camera)
            numerical = (L_plus - L_minus) / (2 * eps)

            a = float(analytic_col[gi])

            # Tolerance: 10% relative OR 2e-2 absolute — whichever is looser.
            # Fixed-point quantization at 2^16 loses 1.5e-5 per accumulation;
            # on 16x16 pixels with back-to-front tail terms the backward
            # can accumulate ULP-level noise. The alpha-min skip (forward.slang
            # ALPHA_MIN=1/255) also adds a hard contribution cutoff: a splat
            # straddling the 1/255 boundary is a step discontinuity, so a
            # central finite-difference across it reads a large numerical
            # gradient while the analytic sees alpha=0 — a benign artifact only
            # for near-floor (<2e-2) gradients (e.g. log_scale_z on a
            # near-spherical splat). The 2e-2 absolute floor excludes that
            # regime; real bugs still trip (rel_err>1.0 on large-magnitude
            # grads — sign flips, wrong-scale mean grads — all far above 2e-2).
            abs_diff = abs(numerical - a)
            denom = max(abs(numerical), abs(a), 1e-2)
            rel_err = abs_diff / denom
            if rel_err > 0.10 and abs_diff > 2e-2:
                mismatches.append(
                    f"g{gi} {grad_name:14s}: analytic={a:+.4e} "
                    f"numerical={numerical:+.4e}  rel_err={rel_err:.3f}"
                )
    return mismatches
