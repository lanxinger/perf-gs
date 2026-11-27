# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""VRAM budget planning, streamed targets, sort-pair accounting, viewer growth."""
from __future__ import annotations

import sys

import numpy as np
import pytest
import slangpy as spy
from _colmap_testkit import write_colmap_scene
from _pipeline_testkit import TestPipeline

from trainer.backend.compute.device import create_device
from trainer.backend.densify.structure_tensor import StructureTensor
from trainer.backend.memory import budget
from trainer.backend.platform.dxgi import query_video_memory
from trainer.backend.render.pipeline import TrainingPipeline, default_sort_pairs, estimate_pipeline_bytes
from trainer.config.config import AdamLRs, LossConfig
from trainer.data.dataset import simple_camera
from trainer.frontend.app import TrainingApplication
from trainer.frontend.cli import build_parser
from trainer.frontend.viewer.viewer import Viewer

GIB = budget.GIB
CAMERA = simple_camera(width=16, height=16, eye=(0.0, 0.0, 0.0),
                       target=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0), fovy_degrees=60.0)


@pytest.fixture(params=["vulkan", "d3d12", "metal"])
def device(request):
    try:
        device = create_device(backend=request.param)
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"backend unavailable: {error}")
    yield device
    device.close()


def _three_splats(device, **kwargs) -> TestPipeline:
    pipeline = TestPipeline(device=device, width=16, height=16, max_splats=16,
                            loss_config=LossConfig(ssim_weight=0.0, alpha_weight=0.0),
                            **kwargs)
    pipeline.set_primitive_from_numpy(
        means=np.array([[0.0, 0.0, 3.0], [0.2, 0.1, 2.5], [-0.15, -0.1, 2.0]], np.float32),
        log_scales=np.log(np.full((3, 3), 0.3, np.float32)),
        rot_quats=np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (3, 1)),
        opacities=np.array([0.8, 0.6, 0.7], np.float32),
        color_logits=np.array([[1.5, -1.5, -1.5], [-1.5, 1.5, -1.5], [-1.5, -1.5, 1.5]],
                              np.float32))
    pipeline.optimizer.zero_grad()
    return pipeline


def test_preset_sort_workspace() -> None:
    parser = build_parser()
    for preset, expected in (("ultrafast", 16_776_960), ("balanced", 16_776_960), ("quality", 24_000_000)):
        assert default_sort_pairs(parser.parse_args(["--preset", preset]).max_splats) == expected


def test_fit_capacity_is_the_largest_that_fits() -> None:
    def cost(capacity: int) -> int:
        return 1000 + 10 * capacity
    assert budget.fit_capacity(500, 10_000, cost) == 500
    assert budget.fit_capacity(5000, 10_000, cost) == 900
    assert budget.fit_capacity(5000, 999, cost) == 0


def test_available_bytes_reserves_headroom(monkeypatch) -> None:
    monkeypatch.setattr(budget, "query_video_memory", lambda name: (16 * GIB, GIB))
    assert budget.available_bytes(0, "gpu") == 16 * GIB - GIB - 16 * GIB // 10
    # An explicit budget replaces DXGI's but still subtracts what is in use.
    assert budget.available_bytes(8, "gpu") == 8 * GIB - GIB - 8 * GIB // 10
    monkeypatch.setattr(budget, "query_video_memory", lambda name: None)
    assert budget.available_bytes(0, "gpu") is None
    assert budget.available_bytes(4, "gpu") == 4 * GIB - GIB // 2


@pytest.mark.skipif(sys.platform not in ("win32", "darwin"), reason="DXGI/Metal probe is Windows/macOS-only")
def test_query_video_memory_reports_this_process() -> None:
    measured = query_video_memory("no such adapter")   # falls back to the largest GPU
    assert measured is not None
    assert measured[0] > 0 and measured[1] >= 0


def test_allocated_bytes_cover_the_planned_allocation(device) -> None:
    pipeline = _three_splats(device)
    owned = budget.allocated_bytes(pipeline.buffers)
    assert owned["raster.output_image"] >= 16 * 16 * 16
    planned = estimate_pipeline_bytes(pipeline.shared_data_module, max_splats=16,
                                      width=16, height=16, sh_degree=0,
                                      max_sort_pairs=pipeline.max_sort_pairs)
    assert sum(owned.values()) >= planned


def test_streamed_target_matches_resident_texture(device) -> None:
    target = np.zeros((16, 16, 3), np.float32)
    target[..., 0] = 0.5
    lrs = AdamLRs(mean=1e-2, scale=1e-2, rot=1e-3, opacity=1e-2, color=1e-2)
    results = []
    for streamed in (False, True):
        pipeline = _three_splats(device)
        packed = pipeline.targets.pack(target)
        pipeline.targets.select(
            packed if streamed else pipeline.targets.upload(packed))
        pipeline.train_step(CAMERA, lrs=lrs)
        device.wait_for_idle()
        results.append(pipeline.buffers.primitive.values.to_numpy().copy())
    if device.info.type == spy.DeviceType.metal:
        # Hardware atomic_float accumulation can differ by ~1 fp32 ULP across runs.
        np.testing.assert_allclose(
            results[0].view(np.float32), results[1].view(np.float32), rtol=1e-5, atol=1e-6)
    else:
        np.testing.assert_array_equal(results[0], results[1])


def test_streamed_structure_fields_match_resident_texture(device) -> None:
    target = np.random.default_rng(0).uniform(0.0, 1.0, (16, 16, 3)).astype(np.float32)
    builder = StructureTensor(device, 16, 16)
    results = []
    for mode in ("resident", "derived"):
        pipeline = _three_splats(device)
        packed = pipeline.targets.pack(target)
        target_texture = pipeline.targets.upload(packed)
        field = device.create_texture(
            type=spy.TextureType.texture_2d, format=spy.Format.rgba16_float,
            width=16, height=16,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access)
        batch = builder.runtime.command_batch("field")
        builder.record_into(batch.encoder, target_texture, field)
        builder.runtime.submit(batch)
        builder.runtime.wait_for_idle()
        structure = field if mode == "resident" else builder
        pipeline.targets.select(target_texture if mode == "resident" else packed)
        pipeline.train_step(CAMERA, structure_tensor=structure)
        device.wait_for_idle()
        eta = pipeline.buffers.eta
        results.append(np.stack([buffer.to_numpy() for buffer in eta.components.values()]))
    assert results[0].any()
    np.testing.assert_array_equal(results[0], results[1])


def test_sort_stats_count_demand_and_dropped_pairs(device) -> None:
    roomy = _three_splats(device, max_sort_pairs=1024)
    roomy.render(CAMERA)
    peak, dropped = roomy.buffers.sort.stats.to_numpy().view(np.uint32)
    assert peak > 2 and dropped == 0
    tight = _three_splats(device, max_sort_pairs=2)
    tight.render(CAMERA)
    assert list(tight.buffers.sort.stats.to_numpy().view(np.uint32)) == [peak, 1]


def test_viewer_grows_to_the_snapshot(device) -> None:
    train = _three_splats(device)
    viewer = Viewer(device, width=16, height=16, max_splats=1)
    viewer.update_gaussians(train)
    device.wait_for_idle()
    assert viewer.max_splats == 3
    shown = viewer._pipeline.buffers.primitive.active_count.to_numpy().view(np.uint32)
    assert shown[0] == 3
    viewer._pipeline.render(CAMERA)      # the resized buffers rasterize
    assert viewer._pipeline.read_output_image()[..., 3].max() > 0


def _train_scene(device, tmp_path, monkeypatch, residency: str) -> np.ndarray:
    scene = write_colmap_scene(tmp_path / residency, views=[
        (f"{i}.png", 16, 16, 20.0) for i in range(4)])
    args = build_parser().parse_args([
        "--scene", str(scene), "--no-viewer", "--iters", "4", "--max-splats", "64",
        "--width", "16", "--height", "16", "--eval-holdout-stride", "2",
        "--no-evaluation", "--no-write-ground-truths", "--preview-every", "0",
        "--densify-mode", "structure_aware", "--densify-from", "0", "--densify-to", "4",
        "--densify-every", "1", "--sad-accum-every", "1", "--sad-structure-every", "2",
        "--image-residency", residency, "--run-report", "--run-name", residency,
    ])
    monkeypatch.setattr("trainer.frontend.app.create_training_device", lambda _args: device)
    # Earlier tests' devices count against this process's DXGI budget.
    monkeypatch.setattr("trainer.frontend.app.available_bytes", lambda *_: None)
    monkeypatch.setattr("trainer.frontend.factory.TrainingPipeline",
                        lambda **kwargs: TrainingPipeline(max_sort_pairs=512, **kwargs))
    app = TrainingApplication(args)
    app.prepare_training_context()
    try:
        assert app._stream_targets == (residency == "stream")
        app.build_training_loop().launch_training_loop()
        app.pipeline.runtime.wait_for_idle()
        memory = app._report_memory()
        ram, cache = memory["host_ram"], memory["gpu_view_cache"]
        assert ram["decoded_images_bytes"] == 4 * 16 * 16 * 3 * 4
        assert ram["linear_cache_bytes"] == 0
        packed_bytes = 2 * 16 * 16 * 4
        if residency == "stream":
            assert ram["packed_training_images_bytes"] == packed_bytes
            assert cache == {"training_images_bytes": 0, "structure_fields_bytes": 0}
        else:
            assert ram["packed_training_images_bytes"] == 0
            assert cache["training_images_bytes"] >= packed_bytes
            assert cache["structure_fields_bytes"] >= 2 * packed_bytes
        return app.pipeline.buffers.primitive.values.to_numpy().copy()
    finally:
        app.runtime.artifacts.close_all()


def test_streamed_training_matches_resident(device, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    resident = _train_scene(device, tmp_path, monkeypatch, "gpu")
    streamed = _train_scene(device, tmp_path, monkeypatch, "stream")
    if device.info.type == spy.DeviceType.metal:
        np.testing.assert_allclose(
            resident.view(np.float32), streamed.view(np.float32), rtol=1e-5, atol=1e-5)
    else:
        np.testing.assert_array_equal(resident, streamed)
