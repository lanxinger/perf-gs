# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""bf16 Adam moments (``--adam-moments bf16``): half the optimizer memory."""

from __future__ import annotations

import numpy as np
import pytest
import slangpy as spy
from _colmap_testkit import write_colmap_scene
from _pipeline_testkit import TestPipeline

from trainer.backend.compute.device import SHADERS_DIR, create_device
from trainer.backend.memory.buffers import GRAD_SCALAR_COMPONENTS
from trainer.backend.memory.layout import StructLayout
from trainer.backend.optimize.adam import AdamOptimizer
from trainer.backend.render.pipeline import TrainingPipeline
from trainer.config.config import AdamLRs, AppearanceConfig, LossConfig
from trainer.data.dataset import simple_camera
from trainer.frontend.app import TrainingApplication
from trainer.frontend.cli import build_parser

CAMERA = simple_camera(width=16, height=16, eye=(0.0, 0.0, 0.0),
                       target=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0), fovy_degrees=60.0)
BF16 = {"ADAM_MOMENTS_BF16": "1"}
# float32 AdamState offsets of each scalar's m and v, in the bf16 word order
# (GRAD_SCALAR_COMPONENTS); see test_optimizer.
M_IDX = [0, 1, 2, 6, 7, 8, 12, 13, 14, 15, 20, 22, 23, 24]
V_IDX = [3, 4, 5, 9, 10, 11, 16, 17, 18, 19, 21, 25, 26, 27]
BF16_ULP = 2.0 ** -7


@pytest.fixture(params=["vulkan", "d3d12", "metal"])
def backend(request):
    return request.param


def _device(backend: str, defines=None):
    try:
        return create_device(backend=backend, defines=defines)
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"backend unavailable: {error}")


def _splats(device, count: int = 3) -> TestPipeline:
    pipeline = TestPipeline(device=device, width=16, height=16, max_splats=16,
                            appearance_config=AppearanceConfig(sh_degree=1),
                            loss_config=LossConfig(ssim_weight=0.0, alpha_weight=0.0))
    pipeline.set_primitive_from_numpy(
        means=np.array([[0.0, 0.0, 3.0], [0.2, 0.1, 2.5], [-0.15, -0.1, 2.0]] * 6,
                       np.float32)[:count],
        log_scales=np.log(np.full((count, 3), 0.3, np.float32)),
        rot_quats=np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (count, 1)),
        opacities=np.full(count, 0.7, np.float32),
        color_logits=np.tile(np.array([1.5, -1.5, -1.5], np.float32), (count, 1)))
    pipeline.optimizer.zero_grad()
    return pipeline


def _words(buffer) -> np.ndarray:
    return np.frombuffer(buffer.to_numpy().tobytes(), np.uint32)


def _halves(words: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (words << 16).view(np.float32), (words & 0xFFFF0000).view(np.float32)


def test_bf16_moments_halve_memory_and_round_only_the_stored_moments(backend) -> None:
    target = np.zeros((16, 16, 3), np.float32)
    target[..., 0] = 0.5
    lrs = AdamLRs(mean=1e-2, scale=1e-2, rot=1e-3, opacity=1e-2, color=1e-2)
    runs = {}
    for name, defines in (("fp32", None), ("bf16", BF16)):
        device = _device(backend, defines)
        pipeline = _splats(device)
        pipeline.targets.select(pipeline.targets.pack(target))
        pipeline.train_step(CAMERA, lrs=lrs, active_sh_degree=1)
        device.wait_for_idle()
        b = pipeline.buffers
        runs[name] = [_words(b.primitive.values), _words(b.sh.features_rest),
                      _words(b.adam.state), _words(b.sh.adam_rest)]
        device.close()
    gaussians, features, adam, adam_rest = runs["fp32"]
    gaussians_b, features_b, adam_b, adam_rest_b = runs["bf16"]

    # The step itself runs in float32 from zero moments.
    np.testing.assert_array_equal(gaussians, gaussians_b)
    np.testing.assert_array_equal(features, features_b)
    assert adam_b.nbytes * 2 == adam.nbytes
    assert adam_rest_b.nbytes * 2 == adam_rest.nbytes

    moments = adam.view(np.float32).reshape(16, 28)
    m, v = _halves(adam_b.reshape(16, 14))
    np.testing.assert_allclose(m, moments[:, M_IDX], rtol=BF16_ULP, atol=0)
    np.testing.assert_allclose(v, moments[:, V_IDX], rtol=BF16_ULP, atol=0)
    rest = adam_rest.view(np.float32).reshape(-1, 2, 16)
    m, v = _halves(adam_rest_b.reshape(-1, 16))
    np.testing.assert_allclose(m, rest[:, 0], rtol=BF16_ULP, atol=0)
    np.testing.assert_allclose(v, rest[:, 1], rtol=BF16_ULP, atol=0)


def test_stochastic_rounding_keeps_the_second_moment_moving(backend) -> None:
    """Each step moves v by 0.1% (1 - beta2), under half a bf16 ulp: rounding
    to nearest would leave it at its start value forever."""
    device = _device(backend, BF16)
    pipeline = _splats(device, count=16)
    b = pipeline.buffers
    start = np.full(16 * 14, np.float32(0.25).view(np.uint32), np.uint32)  # m=0, v=0.25
    b.adam.state.storage.copy_from_numpy(start)
    from trainer.backend.memory.buffers import encode_grad_array
    gradient = encode_grad_array(b.gradients.params["mean_x"], np.ones(16, dtype=np.float64))
    steps = 300
    for step in range(1, steps + 1):
        for name in GRAD_SCALAR_COMPONENTS:
            b.gradients.params[name].copy_from_numpy(gradient)
        pipeline.commands.dispatch(pipeline.kernels.kernel(*AdamOptimizer.KERNELS["adam"]),
            "adam", thread_count=[16, 1, 1],
            gaussians=b.primitive.values.storage, adam_state=b.adam.state.storage,
            num_gaussians_counter=b.primitive.active_count.storage,
            **b.gradients.param_kwargs(), cluster_visible=b.raster.cluster_visible.storage,
            cfg={"lr_mean": 0.0, "lr_scale": 0.0, "lr_rot": 0.0, "lr_opacity": 0.0,
                 "lr_color": 0.0, "bias_correction1": 1.0, "bias_correction2": 1.0,
                 "step": step},
            pixels_per_step=1.0)
    device.wait_for_idle()
    _, v = _halves(_words(b.adam.state))
    device.close()
    exact = 1.0 - 0.75 * 0.999 ** steps
    assert abs(v.mean() / exact - 1.0) < 0.01


def test_bf16_training_run_with_sh_and_densification(backend, tmp_path, monkeypatch) -> None:
    _device(backend).close()
    monkeypatch.chdir(tmp_path)
    scene = write_colmap_scene(tmp_path / "scene", views=[
        (f"{i}.png", 16, 16, 20.0) for i in range(4)])
    args = build_parser().parse_args([
        "--scene", str(scene), "--backend", backend, "--adam-moments", "bf16",
        "--sh-degree", "1", "--no-viewer", "--iters", "4", "--max-splats", "64",
        "--width", "16", "--height", "16", "--eval-holdout-stride", "2",
        "--no-evaluation", "--no-write-ground-truths",
        "--preview-every", "0", "--densify-mode", "structure_aware", "--densify-from", "0",
        "--densify-to", "4", "--densify-every", "1", "--sad-accum-every", "1",
        "--sad-structure-every", "2",
    ])
    monkeypatch.setattr("trainer.frontend.app.available_bytes", lambda *_: None)
    monkeypatch.setattr("trainer.frontend.factory.TrainingPipeline",
                        lambda **kwargs: TrainingPipeline(max_sort_pairs=512, **kwargs))
    app = TrainingApplication(args)
    app.prepare_training_context()
    try:
        app.build_training_loop().launch_training_loop()
        app.pipeline.runtime.wait_for_idle()
        sh = app.pipeline.buffers.sh
        assert sh.moment_planes == sh.rest_floats
        assert np.isfinite(_words(app.pipeline.buffers.primitive.values).view(np.float32)).all()
    finally:
        app.runtime.artifacts.close_all()
        app.device.close()


def test_a_hub_run_turns_bf16_on_after_the_device_exists(backend, tmp_path, monkeypatch) -> None:
    """The hub creates its device before the panel is read; a run switched to bf16 there
    must still get the bf16 layout (from its own Slang session) and train."""
    _device(backend).close()
    monkeypatch.chdir(tmp_path)
    scene = write_colmap_scene(tmp_path / "scene", views=[
        (f"{i}.png", 16, 16, 20.0) for i in range(4)])
    args = build_parser().parse_args([
        "--scene", str(scene), "--backend", backend, "--adam-moments", "fp32", "--no-viewer", "--iters", "4",
        "--max-splats", "64", "--width", "16", "--height", "16",
        "--eval-holdout-stride", "2", "--no-evaluation", "--no-write-ground-truths",
        "--preview-every", "0",
    ])
    monkeypatch.setattr("trainer.frontend.app.available_bytes", lambda *_: None)
    monkeypatch.setattr("trainer.frontend.factory.TrainingPipeline",
                        lambda **kwargs: TrainingPipeline(max_sort_pairs=512, **kwargs))
    create_device_then = TrainingApplication._create_device

    def device_then_panel(app) -> None:
        create_device_then(app)
        app.args.adam_moments = "bf16"     # the panel's setup lands after the device exists

    monkeypatch.setattr(TrainingApplication, "_create_device", device_then_panel)
    app = TrainingApplication(args)
    app.prepare_training_context()
    try:
        assert app.pipeline.slang_session is not app.device.slang_session
        # bf16 packs each scalar's m and v into one word: 14 words a splat, not 28.
        assert _words(app.pipeline.buffers.adam.state).size == app.pipeline.max_splats * 14
        app.build_training_loop().launch_training_loop()
        app.pipeline.runtime.wait_for_idle()
        assert np.isfinite(_words(app.pipeline.buffers.primitive.values).view(np.float32)).all()
    finally:
        app.runtime.artifacts.close_all()
        app.device.close()
