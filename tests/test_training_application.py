# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Focused contracts for local training-application configuration objects."""

from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import numpy as np
from PIL import Image

from trainer.backend.compute.device import create_device
from trainer.backend.compute.fixed_point import (
    FIXED_POINT_INT32,
    FIXED_POINT_INT64,
    fixed_point_format,
    fixed_point_shader_defines,
    supports_int64_atomics,
)
from trainer.config.config import (
    AppearanceConfig,
    LossConfig,
    RasterConfig,
    RegularizationConfig,
    ShaderSpecialization,
)
from trainer.config.resolve import RunConfig, shader_specialization
from _colmap_testkit import write_colmap_scene

from trainer.frontend.evaluation.evaluation import render_all_cameras
from trainer.frontend import hub as hub_module
from trainer.frontend.hub import TrainingHub
from trainer.config.config_types import DensificationMode
from trainer.frontend.viewer.session import (
    MODE_HUB,
    DatasetState,
    ResultState,
    Session,
)
from trainer.data.targets import TargetDataset
from trainer.frontend.app import TrainingApplication, TrainingLoop
from trainer.frontend.cli import build_parser
from trainer.frontend.factory import (
    SlangSessions,
    create_pipeline,
    create_training_device,
    shader_defines,
)


def _parse(*flags: str):
    return build_parser().parse_args(["--scene", "unused", *flags])


def test_default_shader_specialization_uses_shader_default() -> None:
    # No stride override and hardness NOT specialized (default) => no defines.
    assert shader_defines(ShaderSpecialization()) is None


def test_strided_backward_specialization_emits_slang_uint_define() -> None:
    assert shader_defines(ShaderSpecialization(backward_pixel_stride=4)) == {
        "BACKWARD_PIXEL_STRIDE": "4u",
    }


def test_hardness_one_specialization_emits_define() -> None:
    assert shader_defines(ShaderSpecialization(hardness_is_one=True)) == {
        "COMPOSITOR_HARDNESS_ONE": "1"}


def test_specialization_compiles_hardness_away_only_when_target_le_one() -> None:
    on = shader_specialization(_parse("--adam-moments", "fp32", "--extinction-hardness", "1"))
    assert shader_defines(on) == {"COMPOSITOR_HARDNESS_ONE": "1"}
    off = shader_specialization(_parse("--adam-moments", "fp32", "--extinction-hardness", "2"))
    assert shader_defines(off) is None


def test_a_run_compiles_in_the_session_of_its_own_specialization() -> None:
    """The hub's device predates the panel, so a run with other compile-time settings gets its
    own Slang session, once per specialization; the device's session serves its own defines."""
    launch = shader_specialization(_parse("--hub"))
    device = create_device(defines=shader_defines(launch))
    sessions = SlangSessions(device)
    assert sessions.session_for(launch) is device.slang_session

    opaque = shader_specialization(_parse("--hub", "--splat-kernel", "opaque_core"))
    session = sessions.session_for(opaque)
    assert session is not device.slang_session
    backend_defines = {
        k: v for k, v in device.slang_session.desc.compiler_options.defines.items()
        if k == "TRAINER_METAL_BACKEND"
    }
    assert session.desc.compiler_options.defines == {
        **backend_defines,
        **fixed_point_shader_defines(device),
        **(shader_defines(opaque) or {}),
    }
    assert sessions.session_for(opaque) is session

    # Disabling macros present on the initial device (hardness=1, bf16) must drop them
    # rather than silently inheriting obsolete specializations from own.defines.
    plain = shader_specialization(
        _parse("--hub", "--extinction-hardness", "2", "--adam-moments", "fp32"))
    plain_session = sessions.session_for(plain)
    assert dict(plain_session.desc.compiler_options.defines) == {
        **backend_defines,
        **fixed_point_shader_defines(device),
    }
    device.close()


def test_fixed_point_format_follows_device_atomic_int64_capability(monkeypatch) -> None:
    import slangpy as spy
    from trainer.backend.compute import device as device_module

    def fake_device(api_name: str, features: set, defines: dict[str, str] | None = None):
        return SimpleNamespace(
            info=SimpleNamespace(api_name=api_name),
            has_feature=lambda feature: feature in features,
            slang_session=SimpleNamespace(desc=SimpleNamespace(
                compiler_options=SimpleNamespace(defines=dict(defines or {})))),
            close=lambda: closed.append(True),
        )

    closed: list[bool] = []
    vk_64 = fake_device("Vulkan", {spy.Feature.atomic_int64})
    vk_32 = fake_device("Vulkan", set())
    dx_64 = fake_device("D3D12", {spy.Feature.sm_6_6, spy.Feature.int64})
    dx_32 = fake_device("D3D12", {spy.Feature.int64})
    metal = fake_device("Metal", set(), {"TRAINER_METAL_BACKEND": "1"})

    assert supports_int64_atomics(vk_64) and fixed_point_format(vk_64) is FIXED_POINT_INT64
    assert not supports_int64_atomics(vk_32) and fixed_point_format(vk_32) is FIXED_POINT_INT32
    assert supports_int64_atomics(dx_64) and fixed_point_format(dx_64) is FIXED_POINT_INT64
    assert not supports_int64_atomics(dx_32) and fixed_point_format(dx_32) is FIXED_POINT_INT32
    # Native Metal emulates exact 64-bit atomics for blend_weight/removal_cost and uses fp32 grads.
    assert fixed_point_format(metal) is FIXED_POINT_INT64
    assert fixed_point_shader_defines(metal) == {}

    opens: list[dict[str, str] | None] = []

    def fake_open(_backend, _debug, _sdi, defines):
        opens.append(defines)
        return fake_device("Vulkan", set(), defines)

    monkeypatch.setattr(device_module, "_open_device", fake_open)
    device = device_module.create_device(backend="vulkan", defines={"ADAM_MOMENTS_BF16": "1"})
    assert opens == [
        {"ADAM_MOMENTS_BF16": "1"},
        {"FIXED_POINT_INT32": "1", "ADAM_MOMENTS_BF16": "1"},
    ]
    assert closed == [True]
    assert fixed_point_shader_defines(device) == {"FIXED_POINT_INT32": "1"}


def test_int32_device_preserves_fixed_point_define_across_slang_sessions() -> None:
    launch = shader_specialization(_parse("--hub"))
    device = create_device(defines={"FIXED_POINT_INT32": "1", **(shader_defines(launch) or {})})
    try:
        assert fixed_point_format(device) is FIXED_POINT_INT32
        backend_defines = {
            k: v for k, v in device.slang_session.desc.compiler_options.defines.items()
            if k == "TRAINER_METAL_BACKEND"
        }
        sessions = SlangSessions(device)
        assert sessions.session_for(launch) is device.slang_session

        opaque = shader_specialization(_parse("--hub", "--splat-kernel", "opaque_core"))
        session = sessions.session_for(opaque)
        assert session is not device.slang_session
        assert session.desc.compiler_options.defines == {
            **backend_defines,
            "FIXED_POINT_INT32": "1",
            **(shader_defines(opaque) or {}),
        }
        assert fixed_point_format(session) is FIXED_POINT_INT32
    finally:
        device.close()


def test_position_lr_horizon_follows_iters_unless_overridden() -> None:
    """``--position-lr-max-steps`` 0 (default) follows ``--iters``."""
    def horizon(*flags):
        return RunConfig.from_args(_parse("--iters", "7000", *flags), 1.0).position_lr_max_steps

    assert horizon() == 7000
    assert horizon("--position-lr-max-steps", "-1") == 7000
    assert horizon("--position-lr-max-steps", "1500") == 1500


def test_device_factory_passes_resolved_specialization(monkeypatch) -> None:
    captured = {}
    sentinel = object()

    def fake_create_device(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr("trainer.frontend.factory.create_device", fake_create_device)
    # Hardness > 1 keeps the compositor's runtime path. Balanced uses bf16 moments.
    args = _parse("--backward-pixel-stride", "2", "--extinction-hardness", "2")

    assert create_training_device(args) is sentinel
    assert captured == {
        "backend": "metal" if __import__("sys").platform == "darwin" else "vulkan",
        # --nsight-debug-names defaults to off unless --capture-external.
        "enable_debug": False,
        "shader_debug_info": False,
        "defines": {"BACKWARD_PIXEL_STRIDE": "2u", "ADAM_MOMENTS_BF16": "1"},
    }


def test_pipeline_factory_separates_configuration_ownership(monkeypatch) -> None:
    captured = {}
    sentinel = object()

    def fake_pipeline(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr("trainer.frontend.factory.TrainingPipeline", fake_pipeline)
    args = _parse(
        "--max-splats", "123", "--z-near-cull", "0.25",
        "--no-cluster-cull", "--mip-splatting", "--mip-kernel-size", "0.1",
        "--sh-degree", "0", "--sh-band-interval", "1000", "--extinction-hardness", "2",
        "--w-alpha", "0.4", "--opacity-reg", "0.03", "--scale-reg", "0.07",
        "--densify-mode", "structure_aware", "--sad-lambda-l2", "1.5",
    )
    scene = Namespace(width=64, height=32)

    result = create_pipeline(device="device", gpu_lock="lock", scene=scene,
                             max_splats=123, config=RunConfig.from_args(args, 1.0))

    assert result is sentinel
    assert captured["max_splats"] == 123
    assert captured["raster_config"] == RasterConfig(
        z_near_cull=0.25,
        cluster_cull=False,
        mip_enabled=True,
        kernel_size=0.1,
    )
    assert captured["densify_mode"] == DensificationMode.STRUCTURE_AWARE
    assert captured["loss_config"] == LossConfig(
        alpha_weight=0.4, ssim_weight=0.3, l2_weight=1.5)
    assert captured["regularization_config"] == RegularizationConfig(
        opacity_weight=0.03, scale_weight=0.07)
    assert captured["appearance_config"] == AppearanceConfig(
        sh_degree=0, band_interval=1000)


def test_training_loop_presents_after_full_iteration() -> None:
    events = []
    loop = object.__new__(TrainingLoop)
    states = iter((True, False))
    context = SimpleNamespace(step_number=1)
    loop.args = Namespace(iters=3)
    from trainer.frontend.viewer.viewer_control import ViewerSession
    loop.runtime = SimpleNamespace(viewer=ViewerSession(None), pipeline=None)
    loop.instrumentation = SimpleNamespace(
        end_iteration=lambda _pipeline: events.append("present"))
    loop.observation_seconds = 0.0
    loop.observation_in_wall_seconds = 0.0

    loop._begin_training_loop = lambda: events.append("begin")
    loop._should_continue = lambda: next(states)
    loop._next_iteration = lambda: events.append("next") or context
    loop._prepare_iteration = lambda ctx: events.append("prepare")
    loop._execute_training_step = lambda ctx: events.append("train")
    loop._apply_mutations = lambda ctx: events.append("mutations")
    loop._produce_observations = lambda ctx: events.append("observations") or True
    loop._complete_training_loop = lambda: events.append("complete") or "result"

    assert TrainingLoop.launch_training_loop(loop) == "result"
    assert events == [
        "begin", "next", "prepare", "train",
        "mutations", "observations", "present", "complete",
    ]


def test_per_camera_renders_do_not_depend_on_the_ground_truth_dump(tmp_path) -> None:
    """render_all_cameras must create its own cam_XX/ dirs.

    It used to rely on the GT dump having made them, so a run with GT writing
    disabled died with FileNotFoundError at the first render_all iteration —
    thousands of iterations in, after all the training work was already done.
    """
    frame = np.zeros((4, 6, 4), dtype=np.float32)
    pipeline = SimpleNamespace(
        render=lambda camera: None,
        read_output_image=lambda: frame,
    )
    scene = SimpleNamespace(images=[SimpleNamespace(camera=object()) for _ in range(3)])
    cams_dir = tmp_path / "cams"          # deliberately never created
    cams_dir.mkdir()

    render_all_cameras(pipeline, scene, 3000, cams_dir)

    assert sorted(p.name for p in cams_dir.iterdir()) == ["cam_00", "cam_01", "cam_02"]
    for cam in range(3):
        assert (cams_dir / f"cam_{cam:02d}" / "iter_003000.png").is_file()


def test_ground_truths_land_one_per_camera_and_are_not_rewritten(tmp_path) -> None:
    targets = object.__new__(TargetDataset)
    targets._use_alpha = False
    targets._srgb = [np.full((3, 5, 3), i / 4.0, dtype=np.float32) for i in range(4)]
    cams_dir = tmp_path / "cams"
    cams_dir.mkdir()

    targets.write_ground_truths(cams_dir)
    written = sorted(p.relative_to(cams_dir).as_posix() for p in cams_dir.rglob("*.png"))
    assert written == [f"cam_{i:02d}/gt.png" for i in range(4)]

    # Order must survive the threaded write: cam_i's gt is target i, not a neighbour.
    for i in range(4):
        pixels = np.asarray(Image.open(cams_dir / f"cam_{i:02d}" / "gt.png"))
        assert pixels.shape == (3, 5, 3)
        assert np.all(pixels == round(i / 4.0 * 255))

    stamps = {p: p.stat().st_mtime_ns for p in cams_dir.rglob("*.png")}
    targets.write_ground_truths(cams_dir)      # rerun: an existing gt is left alone
    assert {p: p.stat().st_mtime_ns for p in cams_dir.rglob("*.png")} == stamps


class _FakeHubViewer:
    """Records what the hub asks the viewer to show.

    Wraps a REAL :class:`Session` rather than a stub, so what the hub publishes
    (dataset lifecycle, notices) is asserted as the panel would read it.
    """

    def __init__(self) -> None:
        self.state = Session(MODE_HUB)
        self.previewed: dict | None = None
        self.attached: list | None = None
        self.snapped = False
        self.suspend_depth = 0
        self.suspended_during_upload = False

    @property
    def messages(self) -> list[str]:
        return list(self.state.snapshot().log)

    @property
    def notices(self) -> list[str]:
        return [n.text for n in self.state.snapshot().notices]

    def suspend_rendering(self) -> None:
        self.suspend_depth += 1

    def resume_rendering(self) -> None:
        self.suspend_depth -= 1

    def preview_points(self, initial_points: dict) -> int:
        self.previewed = initial_points
        self.suspended_during_upload = self.suspend_depth > 0
        return int(initial_points["means"].shape[0])

    def attach_scene(self, camera_sets, snap: bool = False) -> None:
        self.attached = {s.label: [name for name, _cam in s.cameras] for s in camera_sets}
        self.snapped = snap


def _hub_app(viewer: _FakeHubViewer) -> TrainingHub:
    app = object.__new__(TrainingApplication)
    app.viewer = viewer
    app.args = Namespace(width=None, height=None, downsample=1, init_opacity=0.1,
                         eval_holdout=1, eval_holdout_stride=2)
    return TrainingHub(app)


def test_dropping_a_colmap_dataset_previews_its_point_cloud(tmp_path) -> None:
    """The SfM cloud has to be on screen BEFORE Train: that is the only chance to
    see a broken reconstruction without paying for a full run first."""
    write_colmap_scene(tmp_path / "recon")
    viewer = _FakeHubViewer()

    _hub_app(viewer)._hub_preview_scene(str(tmp_path / "recon"))

    assert viewer.previewed is not None
    assert viewer.previewed["means"].shape == (8, 3)
    # Uploading writes the buffers the render thread reads.
    assert viewer.suspended_during_upload
    assert viewer.suspend_depth == 0, "rendering left suspended"
    # Every camera, split as the run will split it, so either set can be flown before Train.
    assert viewer.attached == {"eval": ["a.png", "c.png"], "train": ["b.png", "d.png"]}
    assert viewer.snapped
    assert any("press Train" in m for m in viewer.messages)
    # ... and the panel can now say WHAT was loaded, not just that something was.
    snap = viewer.state.snapshot()
    assert snap.dataset_state is DatasetState.READY
    assert snap.dataset.images == 4
    assert snap.result_state is ResultState.INITIALIZED   # points, not a result
    assert any("loaded" in n for n in viewer.notices)


def test_the_sparse_model_is_parsed_with_rendering_suspended(
        tmp_path, monkeypatch) -> None:
    """Suspension has to cover the PARSE too, not just the buffer upload.

    ``load_colmap``'s binary readers are per-record Python loops and a painting
    frame holds the GIL through its submit wait, so a 0.6s parse next to a live
    render thread took over 40s and the drop looked hung.
    """
    write_colmap_scene(tmp_path / "recon")
    viewer = _FakeHubViewer()
    real_load = hub_module.load_colmap
    depth_during_load: list[int] = []

    def spy(*args, **kwargs):
        depth_during_load.append(viewer.suspend_depth)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(hub_module, "load_colmap", spy)

    _hub_app(viewer)._hub_preview_scene(str(tmp_path / "recon"))

    assert depth_during_load == [1], "parsed with the render thread still painting"
    assert viewer.suspend_depth == 0, "rendering left suspended"


def test_preview_cache_reuses_pristine_points_and_tracks_loader_inputs(tmp_path, monkeypatch):
    from trainer.frontend import scene
    from trainer.data import stereo

    path = write_colmap_scene(tmp_path / "registered")
    hub = _hub_app(_FakeHubViewer())
    hub._hub_preview_scene(str(path))
    imported = hub._cached_colmap(str(path))
    assert imported is not None
    args = hub.app.args
    args.scene = str(path)
    args.stereo_init, args.max_splats = True, 64
    args.densify_from, args.densify_to, args.densify_every = 0, 3, 1
    def reject_reparse(*args, **kwargs):
        raise AssertionError("Starting from preview unexpectedly reopened the COLMAP files")
    monkeypatch.setattr(scene, "load_colmap", reject_reparse)
    def add_point(dataset, train_indices, limit):
        assert train_indices == [1, 3]
        for k, value in dataset.initial_points.items():
            dataset.initial_points[k] = np.concatenate((value, value[:1]))
        return 1
    monkeypatch.setattr(stereo, "seed_from_stereo", add_point)
    for _ in range(2):
        run = scene.load_scene_context(args, hub._cached_colmap(str(path)))
        assert run.initial_gaussian_count == 9
        assert len(imported.initial_points["means"]) == 8
    args.init_opacity = 0.2
    assert hub._cached_colmap(str(path)) is None
    args.init_opacity = 0.1
    assert hub._cached_colmap(str(path)) is imported
    (path / "sparse/0/points3D.txt").unlink()
    assert hub._cached_colmap(str(path)) is None


def test_a_dataset_the_loader_rejects_leaves_the_hub_alive(tmp_path) -> None:
    """A preview failure must not kill the process — the user still has to be able
    to drop a different folder or reconstruct again."""
    viewer = _FakeHubViewer()

    _hub_app(viewer)._hub_preview_scene(str(tmp_path / "not-a-scene"))

    assert viewer.previewed is None
    assert viewer.suspend_depth == 0
    assert any("cannot preview" in m for m in viewer.messages)
    # The failure belongs to the DATASET, with a reason the panel can show.
    snap = viewer.state.snapshot()
    assert snap.dataset_state is DatasetState.ERROR
    assert snap.dataset_error
    assert snap.can_drop, "the hub must still accept another folder"


def test_preview_cache_tracks_native_photo_size_and_directory_choice(tmp_path):
    scene = write_colmap_scene(tmp_path / "photos", [(f"{i}.png", 32, 24, 30) for i in range(4)])
    hub = _hub_app(_FakeHubViewer())
    hub.app.args.downsample = 2
    hub._hub_preview_scene(str(scene))
    assert hub._cached_colmap(str(scene)).image_dir.name == "images"
    photos = scene / "images_2"
    photos.mkdir()
    first = photos / "0.png"
    Image.new("RGB", (16, 12)).save(first)
    assert hub._cached_colmap(str(scene)) is None
    hub._hub_preview_scene(str(scene))
    assert hub._cached_colmap(str(scene)).images[0].camera.image_size == (16, 12)
    Image.new("RGB", (15, 11), color=(100, 200, 50)).save(first)
    assert hub._cached_colmap(str(scene)) is None


# --- STBN --------------------------------------------------------------------

def test_stbn_variables_absent_when_the_feature_is_off() -> None:
    """A run without STBN must hand the pipeline None, so every kernel resolves
    to the full-pixel budget rather than a silently sparse gradient."""
    loop = object.__new__(TrainingLoop)
    loop.stbn_sampler = None
    assert loop._stbn_variables(0) is None


def test_prepare_run_creates_viewer_only_when_absent() -> None:
    for has_viewer in (False, True):
        app = object.__new__(TrainingApplication)
        app.viewer = object() if has_viewer else None
        app.scene_context = SimpleNamespace(name="scene")
        steps: list[str] = []
        app._prepare_scene = lambda scene_dir, prepared=None, s=steps: s.append(f"scene:{scene_dir}:{prepared}")
        app._create_pipeline_for_scene = lambda scene, s=steps: s.append("pipeline")
        app._prepare_textures = lambda s=steps: s.append("textures")
        app._create_viewer = lambda scene, s=steps: s.append("viewer")
        app._start_profiler = lambda s=steps: s.append("profiler")
        app._create_training_services = lambda s=steps: s.append("services")

        app.prepare_run("my_scene", "cached")

        expected = ["scene:my_scene:cached", "pipeline", "textures", "profiler", "services"]
        if not has_viewer:
            expected.insert(3, "viewer")
        assert steps == expected
