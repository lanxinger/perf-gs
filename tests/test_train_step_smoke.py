# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""One real train_step records and submits exactly one batch, then updates parameters."""

from __future__ import annotations

import numpy as np
import pytest

from _pipeline_testkit import TestPipeline
from trainer.config.config import AdamLRs, LossConfig
from trainer.data.dataset import simple_camera
from trainer.backend.compute.device import create_device

BACKENDS = ["vulkan", "d3d12", "metal"]


def _setup(backend: str) -> TestPipeline:
    try:
        device = create_device(backend=backend)
    except Exception as exc:  # backend unavailable on host
        pytest.skip(f"backend {backend!r} unavailable: {exc}")
    p = TestPipeline(
        device=device, width=16, height=16, max_splats=16,
        loss_config=LossConfig(ssim_weight=0.0, alpha_weight=0.0),
    )
    means = np.array([[0.0, 0.0, 3.0], [0.2, 0.1, 2.5], [-0.15, -0.1, 2.0]],
                     dtype=np.float32)
    log_scales = np.log(np.full((3, 3), 0.3, dtype=np.float32))
    rot_quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (3, 1))
    opacities = np.array([0.8, 0.6, 0.7], dtype=np.float32)
    color_logits = np.array([[1.5, -1.5, -1.5], [-1.5, 1.5, -1.5], [-1.5, -1.5, 1.5]],
                            dtype=np.float32)
    p.set_primitive_from_numpy(means=means, log_scales=log_scales, rot_quats=rot_quats,
                               opacities=opacities, color_logits=color_logits)
    target = np.zeros((16, 16, 3), np.float32)
    target[..., 0] = 0.5   # non-matching target => non-zero gradients
    p.targets.set_image(target)
    return p


@pytest.mark.parametrize("backend", BACKENDS)
def test_train_step_records_one_batch_and_updates(backend: str) -> None:
    p = _setup(backend)
    cam = simple_camera(width=16, height=16, eye=(0.0, 0.0, 0.0),
                        target=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0),
                        fovy_degrees=60.0)
    # Seed the "scalar grads are zero at start-of-iter" invariant train_step assumes.
    p.optimizer.zero_grad()
    p.device.wait_for_idle()

    before = p.buffers.primitive.values.to_numpy().copy()
    prev_submit = p.last_submit_id

    p.train_step(cam, lrs=AdamLRs(mean=1e-2, scale=1e-2, rot=1e-3,
                                  opacity=1e-2, color=1e-2))
    p.device.wait_for_idle()

    # A single batch was recorded + submitted (submit id advanced) and cleared.
    assert p.last_submit_id != prev_submit
    assert p.commands.batch is None
    # Adam moved parameters (non-matching target => non-zero grads => update).
    after = p.buffers.primitive.values.to_numpy()
    assert not np.array_equal(before, after)


@pytest.mark.parametrize("backend", BACKENDS)
def test_construction_with_initial_state_primes_pipeline(backend: str) -> None:
    """``Pipeline(initial_state=...)`` uploads the snapshot and zeros the derived
    state (Adam, grads, tracking) in ``__init__``, so the pipeline is born ready
    to render + train with no separate priming call. Production path used by
    create_pipeline."""
    try:
        device = create_device(backend=backend)
    except Exception as exc:
        pytest.skip(f"backend {backend!r} unavailable: {exc}")
    state = dict(
        means=np.array([[0.0, 0.0, 3.0], [0.2, 0.1, 2.5], [-0.15, -0.1, 2.0]],
                       dtype=np.float32),
        log_scales=np.log(np.full((3, 3), 0.3, dtype=np.float32)),
        rot_quats=np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (3, 1)),
        opacities=np.array([0.8, 0.6, 0.7], dtype=np.float32),
        color_logits=np.array([[1.5, -1.5, -1.5], [-1.5, 1.5, -1.5],
                               [-1.5, -1.5, 1.5]], dtype=np.float32),
    )
    # TestPipeline is a drop-in Pipeline subclass; it adds the (camera, lrs)
    # train_step shim while still driving the real Pipeline.train_step. This
    # test targets construction-with-initial_state, not the raw ctor per se.
    p = TestPipeline(device=device, width=16, height=16, max_splats=16,
                     loss_config=LossConfig(ssim_weight=0.0, alpha_weight=0.0),
                     initial_state=state)
    cam = simple_camera(width=16, height=16, eye=(0.0, 0.0, 0.0),
                        target=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0),
                        fovy_degrees=60.0)

    # Born valid: the ctor-uploaded snapshot renders content and the optimizer
    # state is reset (step count 0) — no separate priming call was made.
    assert p.iteration == 0
    p.render(cam)
    rendered = p.read_output_image()[..., :3]
    assert np.isfinite(rendered).all()
    assert float(np.abs(rendered).sum()) > 0.0

    # And it trains end-to-end straight from construction (the ctor priming
    # seeded the scalar-grad-zero invariant train_step relies on).
    target = np.zeros((16, 16, 3), np.float32)
    target[..., 0] = 0.5
    p.targets.set_image(target)
    before = p.buffers.primitive.values.to_numpy().copy()
    p.train_step(cam, lrs=AdamLRs(mean=1e-2, scale=1e-2, rot=1e-3,
                                  opacity=1e-2, color=1e-2))
    p.device.wait_for_idle()
    assert not np.array_equal(before, p.buffers.primitive.values.to_numpy())


@pytest.mark.parametrize("backend", BACKENDS)
def test_two_train_steps_advance_submit(backend: str) -> None:
    p = _setup(backend)
    cam = simple_camera(width=16, height=16, eye=(0.0, 0.0, 0.0),
                        target=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0),
                        fovy_degrees=60.0)
    p.optimizer.zero_grad()
    p.train_step(cam, lrs=AdamLRs())
    first = p.last_submit_id
    p.train_step(cam, lrs=AdamLRs())
    p.device.wait_for_idle()
    assert p.last_submit_id != first
    assert p.commands.batch is None
