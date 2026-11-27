# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Allocate and initialize only the resources a run actually consumes."""
from types import SimpleNamespace

import numpy as np
import pytest

from trainer.backend.compute.device import create_device
from trainer.backend.memory import buffers
from trainer.backend.render.pipeline import TrainingPipeline, _module_for_reflection
from trainer.config.config_types import DensificationMode
from trainer.frontend.app import TrainingApplication
from trainer.frontend.cli import build_parser
from _colmap_testkit import write_colmap_scene


@pytest.fixture(params=["vulkan", "d3d12", "metal"])
def device(request):
    try:
        device = create_device(backend=request.param)
    except Exception as error:
        pytest.skip(f"backend unavailable: {error}")
    yield device
    device.close()


def test_structure_allocation_is_optional_and_zeroed(device, monkeypatch):
    allocate = buffers.allocate

    def dirty_allocate(*args, **kwargs):
        result = allocate(*args, **kwargs)
        if result.eta is not None:
            for buffer in [result.eta.n_total, *result.eta.components.values()]:
                buffer.copy_from_numpy(np.full(7, 0xFFFFFFFF, np.uint32))
        return result

    # The estimator executes the same recipe with fake resources; only dirty
    # actual allocations so this tests initialization independent of the driver.
    monkeypatch.setattr(buffers, "allocate", dirty_allocate)
    for mode in DensificationMode:
        pipeline = TrainingPipeline(device, 8, 8, max_splats=7, max_sort_pairs=64,
                                    densify_mode=mode)
        pipeline.runtime.wait_for_idle()
        if mode == DensificationMode.STRUCTURE_AWARE:
            eta = pipeline.buffers.eta
            assert eta is not None
            for buffer in [eta.n_total, *eta.components.values()]:
                np.testing.assert_array_equal(buffer.to_numpy(), np.zeros(7, np.uint32))
        else:
            assert pipeline.buffers.eta is None
            assert not hasattr(pipeline.buffers, "split_factor_x")


def test_budget_accounts_for_optional_structure_resources(device):
    module = _module_for_reflection(device)
    sizes = dict(max_splats=7_000_000, width=16, height=16, max_sort_pairs=1024)
    simple = buffers.estimate_allocation_bytes(module, **sizes)
    structure = buffers.estimate_allocation_bytes(module, **sizes, structure_aware=True)
    assert structure - simple == 7_000_000 * 60 + 4


def test_only_training_camera_targets_are_uploaded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = build_parser().parse_args(["--scene", "unused", "--no-viewer"])
    app = TrainingApplication(args)
    app.scene_context = SimpleNamespace(train_indices=[1, 3], width=8, height=8)
    app.targets = SimpleNamespace(srgb=[0, 1, 2, 3])
    app.config = SimpleNamespace(densify_mode=DensificationMode.SIMPLE)
    uploads = []
    app.pipeline = SimpleNamespace(targets=SimpleNamespace(
        pack=lambda image: image,
        upload=lambda image: uploads.append(image) or f"texture-{image}"))
    app._prepare_textures()
    assert uploads == [1, 3]
    assert app.target_textures == {1: "texture-1", 3: "texture-3"}
    assert app.structure_textures == {}


@pytest.mark.parametrize("mode", list(DensificationMode))
def test_training_loop_runs_with_mode_specific_resources(device, mode, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    scene = write_colmap_scene(tmp_path / "scene", views=[
        (f"{i}.png", 16, 16, 20.0) for i in range(4)])
    args = build_parser().parse_args([
        "--scene", str(scene), "--no-viewer", "--iters", "3", "--max-splats", "64",
        "--width", "16", "--height", "16", "--eval-holdout-stride", "2",
        "--no-evaluation", "--no-write-ground-truths", "--preview-every", "0",
        "--densify-mode", mode.value, "--densify-from", "0", "--densify-to", "3",
        "--densify-every", "1", "--sad-accum-every", "1", "--sad-structure-every", "1",
    ])
    monkeypatch.setattr("trainer.frontend.app.create_training_device", lambda _args: device)
    # Keep the real factory and training loop, with a tiny sort-pair capacity.
    monkeypatch.setattr("trainer.frontend.factory.TrainingPipeline",
                        lambda **kwargs: TrainingPipeline(max_sort_pairs=512, **kwargs))
    app = TrainingApplication(args)
    app.prepare_training_context()
    try:
        assert set(app.target_textures) == {1, 3}
        assert set(app.structure_textures) == (
            {1, 3} if mode == DensificationMode.STRUCTURE_AWARE else set())
        result = app.build_training_loop().launch_training_loop()
        app.pipeline.runtime.wait_for_idle()
        assert result.iterations == 3
        assert app.pipeline.iteration == 3
        app.pipeline.render(app.scene_context.colmapScene.images[0].camera)
        assert np.isfinite(app.pipeline.read_output_image()).all()
    finally:
        app.runtime.artifacts.close_all()
