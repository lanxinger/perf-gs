# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""The training application: ``main`` -> ``TrainingApplication`` (one run's
lifecycle) -> ``TrainingLoop`` (steps, then scheduled mutations, then observations)."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import slangpy as spy

from trainer.backend.densify.removal import accumulate_removal_cost, prune_by_removal_cost
from trainer.backend.densify.structure_tensor import StructureTensor
from trainer.backend.memory.budget import GIB, allocated_bytes, available_bytes, fit_capacity
from trainer.backend.platform.dxgi import query_video_memory
from trainer.backend.render.pipeline import (
    TrainingPipeline,
    _module_for_reflection,
    estimate_pipeline_bytes,
)
from trainer.config.config import (
    DensificationProperties,
    IterationContextualVariables,
    StbnVariables,
)
from trainer.config.config_types import DensificationMode, LearningRateRecipe
from trainer.config.resolve import RunConfig, shader_specialization
from trainer.config.validation import validate_training_config
from trainer.data.parallel import map_ordered
from trainer.data.targets import TargetDataset
from trainer.frontend.cli import parse_args
from trainer.frontend.densification import DensificationController, DensifyOutcome
from trainer.frontend.evaluation import run_report as run_report_module
from trainer.frontend.evaluation.evaluation import (
    eval_test_psnr_srgb,
    render_all_cameras,
    run_eval_sweep,
)
from trainer.frontend.evaluation.post_training import run_post_training
from trainer.frontend.factory import (
    SlangSessions,
    create_gpu_queue_lock,
    create_pipeline,
    create_training_device,
)
from trainer.frontend.profiling.instrumentation import CaptureFinished, LoopInstrumentation
from trainer.frontend.profiling.profiler import Profiler, apply_profile_session_overrides
from trainer.frontend.profiling.stage_timing import StageTimingRecorder
from trainer.frontend.queue_limiter import QueueLimiter
from trainer.frontend.reporting.error_log import ErrorLog, report_exception
from trainer.frontend.reporting.outputs import RunArtifacts
from trainer.frontend.reporting.training_reporter import TrainingReporter
from trainer.frontend.run_control import RunStopped
from trainer.frontend.sampling import CameraSampler, StbnSampler
from trainer.frontend.scene import SceneContext, load_scene_context
from trainer.frontend.schedule import (
    AppearanceSchedule,
    ConvergenceMonitor,
    ExtinctionHardnessSchedule,
    IterationContext,
    LearningRateSchedule,
    TrainingSchedule,
)
from trainer.frontend.viewer.session import ResultInfo, TrainingState
from trainer.frontend.viewer.viewer_control import ViewerSession

# t0 for the run report's process_wall_s; trails the true process start by the import cost.
_PROCESS_STARTED_AT = time.time()


@dataclass(frozen=True)
class RunPaths:
    root: Path
    cameras: Path

    @classmethod
    def create(cls, run_name: str) -> "RunPaths":
        root = Path("outputs/train") / run_name
        cameras = root / "cams"
        cameras.mkdir(parents=True, exist_ok=True)
        return cls(root=root, cameras=cameras)


@dataclass(frozen=True)
class TrainingResult:
    iterations: int
    wall_time_seconds: float
    psnr_curve: list[dict]
    started_at: float
    # In-window observation time; subtract from wall_time_seconds for active training time.
    observation_seconds: float = 0.0
    observation_total_seconds: float = 0.0
    # StageTimingRecorder when --stage-timing-every is set, else None.
    stage_timing: Any = None
    stop_reason: str | None = None


@dataclass
class TrainingRuntime:
    pipeline: TrainingPipeline
    viewer: ViewerSession
    profiler: Profiler | None
    artifacts: RunArtifacts
    queue_limiter: QueueLimiter
    densifier: DensificationController
    reporter: "TrainingReporter"


class TrainingApplication:
    """Owns the prepare -> train -> finish lifecycle of one experiment."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.paths = RunPaths.create(self.run_name)
        self.profiler: Profiler | None = None
        self.profiling_enabled = False

        self.scene_context: SceneContext | None = None
        self.targets: TargetDataset | None = None
        self.runtime: TrainingRuntime | None = None
        self.config: RunConfig | None = None

        self.device: Any = None
        self.slang_sessions: SlangSessions | None = None
        self.gpu_lock: Any = None
        self.pipeline: TrainingPipeline | None = None
        self.viewer: ViewerSession | None = None
        self.error_log: ErrorLog | None = None
        self.target_textures: dict[int, Any] = {}
        self.structure_textures: dict[int, Any] = {}
        self._stream_targets = False
        self._memory_summary: dict = {}
        self.learning_rate_schedule: LearningRateSchedule | None = None
        self.extinction_hardness_schedule: ExtinctionHardnessSchedule | None = None
        self.appearance_schedule: AppearanceSchedule | None = None

    @classmethod
    def from_cli(cls) -> "TrainingApplication":
        return cls(parse_args())

    def prepare_training_context(self) -> None:
        self.profiling_enabled = apply_profile_session_overrides(self.args)
        self._create_device()
        self._prepare_scene(self.args.scene)
        self._create_pipeline_for_scene(self._sceneContext)
        self._prepare_textures()
        self._start_viewer(self.scene_context)
        self._start_profiler()
        self._create_training_services()

    def _prepare_scene(self, scene_dir, prepared=None) -> None:
        """Load the scene and resolve what depends on it; not the device, pipeline or viewer."""
        self.args.scene = str(scene_dir)
        self.scene_context = scene = load_scene_context(self.args, prepared)
        validate_training_config(self.args, n_cams=scene.camera_count,
                                 n_train=len(scene.train_indices),
                                 n_test=len(scene.test_indices))
        self.config = RunConfig.from_args(self.args, scene.extent)
        self._write_resolved_config()
        self._prepare_target_dataset()
        if getattr(self.args, "auto_lens", False):
            start = time.perf_counter()
            from trainer.data.lens_calibration import fit_lens_strength
            indices = {scene.colmapScene.images[i].image_name: i for i in scene.train_indices}
            strength, count = fit_lens_strength(Path(scene_dir), indices, self.targets.srgb)
            self.args.lens_strength = strength
            self._write_resolved_config()
            print("[auto-lens] " + json.dumps(dict(beta=strength, fit_contrasts=count,
                                                  total_wall_s=time.perf_counter() - start)), flush=True)

    def start_training(self) -> None:
        self.prepare_training_context()
        if sys.platform == "darwin" and self.viewer is not None and self.viewer.enabled:
            worker_error: list[BaseException] = []
            def _train_worker() -> None:
                try:
                    self._run_training_and_post(hold_viewer=False, publish_completion=True)
                except BaseException as exc:
                    worker_error.append(exc)
                    if self.viewer is not None and self.viewer._viewer is not None:
                        self.viewer._viewer.stop()
            worker = threading.Thread(target=_train_worker, daemon=True)
            worker.start()
            try:
                self.viewer.run_blocking()
            finally:
                worker.join(timeout=5.0)
            if worker_error:
                raise worker_error[0]
            return
        self._run_training_and_post()


    def _begin_repeat_run(self) -> None:
        """For a repeat hub run, rotate the output dir and free the previous run's GPU resources."""
        if self.runtime is None and not (self.paths.root / "config.json").exists():
            return
        self._rotate_run_dir()
        self._release_run_resources()

    def _rotate_run_dir(self) -> None:
        """Point ``self.paths`` at an unused run directory; two runs must never share one."""
        base = self.args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        runs = self.paths.root.parent        # RunPaths owns the layout, not this
        name, index = base, 1
        while (runs / name).exists():
            index += 1
            name = f"{base}_{index}"
        self.run_name = name
        self.paths = RunPaths.create(name)
        print(f"[hub] next run writes to {self.paths.root}")

    def _release_run_resources(self) -> None:
        """Drop the finished run's GPU-heavy objects before the next one is built.
        Call with rendering suspended: the trainer's buffers must not be released mid-flight."""
        if self.pipeline is not None:
            try:
                self.pipeline.runtime.wait_for_idle()
            except Exception as error:                 # noqa: BLE001
                report_exception("[hub] draining the queue before release failed",
                                 error, self._viewer.state)
        self.runtime = None
        self.pipeline = None
        self.targets = None
        self.target_textures = {}
        self.structure_textures = {}
        self.scene_context = None
        self.config = None              # re-resolved by _prepare_scene
        self.profiler = None            # already closed by _run_training_and_post


    def _run_training_and_post(self, *, hold_viewer: bool = True,
                               publish_completion: bool | None = None) -> str | None:
        """Train, then write the run's post-training artifacts.
        ``hold_viewer`` blocks on the viewer window afterwards (CLI); the hub passes False
        because blocking there would freeze its panel. ``publish_completion`` defaults to
        ``hold_viewer`` so the hub can publish completion itself while macOS CLI viewer
        runs can publish completion from the worker thread without blocking it."""
        
        # TRAINING LOOP IS HERE
        try:
            result = self.build_training_loop().launch_training_loop()
        except CaptureFinished:
            self._training_runtime.artifacts.close_all()
            return
        
        self._memory_summary = self._report_memory()
        self._training_runtime.artifacts.log_json.write(
            json.dumps({"event": "training_memory", **self._memory_summary}) + "\n")

        viewer = self._training_runtime.viewer
        pipeline = self._training_runtime.pipeline
        viewer.finish_training(pipeline)
        try:
            with viewer.unthrottled():
                viewer.checkpoint()
                self._write_final_test_psnr(result)

                viewer.checkpoint()
                run_post_training(
                    progress=self._publish_finalize,
                    checkpoint=viewer.checkpoint,
                    pipeline=pipeline,
                    scene=self._sceneContext.colmapScene,
                    args=self.args,
                    run_dir=self.paths.root,
                    cams_dir=self.paths.cameras,
                    targets_srgb=self._targets.srgb,
                    targets_linear=self._targets.linear,
                    test_indices=self._sceneContext.test_indices,
                    stride=self._sceneContext.holdout_stride,
                    n_cams=self._sceneContext.camera_count,
                    train_wall_s=result.wall_time_seconds,
                    completed_iters=result.iterations,
                    t_start=result.started_at,
                    psnr_curve=result.psnr_curve,
                )
        except RunStopped:
            viewer.state.log("Run stopped; remaining evaluation and exports skipped. "
                             "The trained model is still available.")
            print("[post-training] stopped; retaining the trained model.")
        finally:
            if self._training_runtime.profiler is not None:
                self._training_runtime.profiler.close()
            self._training_runtime.artifacts.close_all()

        if publish_completion is None:
            publish_completion = hold_viewer
        if publish_completion:
            stats = viewer.state.snapshot().stats
            viewer.state.training_completed(ResultInfo(
                iterations=result.iterations, target_iterations=self.args.iters,
                wall_s=result.wall_time_seconds, splats=stats.gaussians,
                run_dir=str(self.paths.root), stop_reason=result.stop_reason))
        if hold_viewer:
            # Metrics must finish (or be stopped) before the CLI waits for the window to close.
            viewer.hold_after_training(pipeline)
        # The hub publishes the completed transition and result record once this returns.
        return result.stop_reason

    def _report_memory(self) -> dict:
        """Print and return the run's sort-pair demand, owned GPU bytes and VRAM use."""
        pipeline = self._pipeline
        peak_pairs, overflows = (
            int(v) for v in pipeline.buffers.sort.stats.to_numpy().view(np.uint32))
        owned = sum(allocated_bytes(pipeline.buffers).values())
        summary = {"max_splats": pipeline.max_splats,
                   "max_sort_pairs": pipeline.max_sort_pairs,
                   "peak_sort_pairs": peak_pairs, "sort_overflows": overflows,
                   "targets_streamed": self._stream_targets,
                   "allocated_bytes": owned}
        measured = query_video_memory(self.device.info.adapter_name)
        vram = ""
        if measured is not None:
            summary["vram_budget_bytes"], summary["vram_usage_bytes"] = measured
            vram = (f"; VRAM in use {measured[1] / GIB:.2f} of "
                    f"{measured[0] / GIB:.2f} GiB budget")
        if self.args.run_report:
            ram = spy.platform.memory_stats()
            summary["host_ram"] = {
                "working_set_bytes": ram.rss or None,
                "peak_working_set_bytes": ram.peak_rss or None,
                **self._targets.host_image_bytes(),
                "packed_training_images_bytes": sum(
                    image.nbytes for image in self.target_textures.values() if isinstance(image, np.ndarray)),
            }
            # Per-view textures live outside pipeline.buffers. Streamed views share scratch space.
            summary["gpu_view_cache"] = {
                "training_images_bytes": sum(
                    image.memory_usage.device for image in self.target_textures.values() if isinstance(image, spy.Texture)),
                "structure_fields_bytes": sum(
                    field.memory_usage.device for field in self.structure_textures.values() if isinstance(field, spy.Texture)),
            }
        print(f"[memory] peak sort pairs {peak_pairs:,} of "
              f"{pipeline.max_sort_pairs:,}; pipeline buffers {owned / GIB:.2f} GiB{vram}")
        if overflows:
            print(f"[memory] WARNING: {overflows} sorts dropped pairs past the "
                  f"sort capacity (those renders miss splats)")
        return summary

    def _publish_finalize(self, label: str, fraction: float) -> None:
        """Publish post-training progress (no-op without a viewer).
        Also the progress callback of ``run_post_training``."""
        if self.viewer is None:
            return
        state = self._viewer.state
        phase = state.snapshot().training_state
        if phase in (TrainingState.PAUSED, TrainingState.STOPPING):
            return
        if phase != TrainingState.FINALIZING:
            state.training_finalizing(label)
        state.set_progress(label, fraction)

    def _prepare_target_dataset(self) -> None:
        scene = self._sceneContext
        self.targets = TargetDataset(
            scene=scene.colmapScene,
            width=scene.width,
            height=scene.height,
            use_alpha=self.args.w_alpha > 0,
        )
        if self.args.write_ground_truths:
            self.targets.write_ground_truths(self.paths.cameras)

    def _create_device(self) -> None:
        """Create the GPU device and shared queue lock.
        Separate from the pipeline so the hub can start before a dataset is chosen."""
        self.device = create_training_device(self.args)
        self.slang_sessions = SlangSessions(self.device)
        self.gpu_lock = create_gpu_queue_lock(viewer_enabled=not self.args.no_viewer)

    def slang_session(self) -> Any:
        """The Slang session compiling the shader specialization of the current args."""
        return self.slang_sessions.session_for(shader_specialization(self.args))

    def _create_pipeline_for_scene(self, scene: SceneContext) -> None:
        """Build the training pipeline for a loaded scene (device must exist)."""
        session = self.slang_session()
        self.pipeline = create_pipeline(
            device=self.device,
            gpu_lock=self.gpu_lock,
            scene=scene,
            max_splats=self._plan_memory(scene, session),
            max_sort_pairs=self.args.max_sort_pairs,
            config=self._config,
            initial_state=scene.colmapScene.initial_points,
            slang_session=session,
        )
        strength = getattr(self.args, "lens_strength", None)
        self._pipeline.rasterizer.natural_vignetting = self.args.natural_vignetting if strength is None else strength
        self._pipeline.mip_filter.set_cameras(
            [scene.colmapScene.images[i].camera for i in scene.train_indices])

    def _plan_memory(self, scene: SceneContext, session: Any) -> int:
        """Fit splat capacity to the VRAM budget and return it.
        Lowering capacity changes the trained model, so it is a reported last resort; the
        viewer is budgeted at full capacity since it grows to the trainer's splat count."""
        args = self.args
        self._stream_targets = args.image_residency == "stream"
        structure = self._densify_mode == DensificationMode.STRUCTURE_AWARE
        module = _module_for_reflection(self.device, session)
        target_bytes = 0 if self._stream_targets else (
            len(scene.train_indices) * scene.width * scene.height
            * (4 + (8 if structure else 0)))

        def cost(capacity: int) -> int:
            shape = {"max_splats": capacity, "width": scene.width,
                     "height": scene.height, "sh_degree": args.sh_degree}
            total = estimate_pipeline_bytes(module, structure_aware=structure, max_sort_pairs=args.max_sort_pairs, **shape)
            if not args.no_viewer:
                total += estimate_pipeline_bytes(module, viewer_only=True, **shape)
            return total + target_bytes

        requested = args.max_splats
        available = available_bytes(args.vram_budget_gib, self.device.info.adapter_name)
        if available is None:
            print(f"[memory] VRAM budget unknown (pass --vram-budget-gib); "
                  f"capacity {requested:,} splats unchecked")
            return requested
        capacity = fit_capacity(requested, available, cost)
        if capacity < scene.initial_gaussian_count:
            raise RuntimeError(
                f"VRAM budget of {available / GIB:.2f} GiB cannot hold the "
                f"{scene.initial_gaussian_count:,} initial points; lower --sh-degree, "
                f"--downsample more, or pass --no-viewer / --image-residency stream")
        if capacity < requested:
            print(f"[memory] WARNING: --max-splats {requested:,} does not fit the VRAM "
                  f"budget; capacity lowered to {capacity:,} (densification stops "
                  f"growing there)")
        print(f"[memory] {available / GIB:.2f} GiB available; planned "
              f"{cost(capacity) / GIB:.2f} GiB for {capacity:,} splats")
        return capacity


    def _prepare_textures(self) -> None:
        """Upload training targets, then build their structure fields.
        Packing runs on a thread pool but uploads stay on this thread: the device queue is
        shared with the viewer under ``gpu_lock`` and is not thread-safe."""
        pipeline = self._pipeline
        targets = self._targets.srgb
        indices = self._sceneContext.train_indices
        # Streamed fields are derived per step on the GPU, never stored.
        resident_fields = (self._densify_mode == DensificationMode.STRUCTURE_AWARE
                           and not self._stream_targets)
        bytes_per_pixel = 4 + (8 if resident_fields else 0)
        planned_bytes = len(indices) * self._sceneContext.width * self._sceneContext.height * bytes_per_pixel
        where = "host RAM" if self._stream_targets else "GPU"
        print(f"[memory] {len(indices)} training targets"
              f"{'/structure fields' if resident_fields else ''} on {where}: "
              f"{planned_bytes / 1024**3:.2f} GiB")
        packed_targets = map_ordered(
            pipeline.targets.pack, [targets[i] for i in indices])
        if self._stream_targets:
            self.target_textures = dict(zip(indices, packed_targets))
        else:
            self.target_textures = {
                i: pipeline.targets.upload(packed)
                for i, packed in zip(indices, packed_targets)
            }

        self.structure_textures = self._structure_fields()

    def _structure_fields(self) -> dict[int, Any]:
        """Structure-tensor field of each training view, built from its target.
        Streamed targets map every view to the builder; fields are then derived per scoring step."""
        if self._densify_mode != DensificationMode.STRUCTURE_AWARE:
            return {}
        scene = self._sceneContext
        builder = StructureTensor(self.device, scene.width, scene.height)
        if self._stream_targets:
            return dict.fromkeys(scene.train_indices, builder)
        started = time.time()
        fields = builder.build_textures(
            {i: self.target_textures[i] for i in scene.train_indices})
        print(f"[SAD] computed {len(fields)} structure fields on GPU "
              f"in {time.time() - started:.1f}s", flush=True)
        return fields

    def _start_viewer(self, scene: SceneContext) -> None:
        if not self.args.no_viewer:
            # Dataset cameras let the viewer's picker and C/V keys snap to the exact train/eval poses.
            self.viewer = ViewerSession.create(
                device=self.device,
                width=scene.width,
                height=scene.height,
                fovy=getattr(self.args, "fovy", 45.0),
                gpu_lock=self.gpu_lock,
                pipeline=self._pipeline,
                camera_sets=scene.camera_sets(),
            )
        else:
            self.viewer = ViewerSession(None)

        if self.error_log is not None:
            self.error_log.attach(self.viewer.state)
        if not (sys.platform == "darwin" and self.viewer.enabled):
            self.viewer.start()

    def _start_profiler(self):
        if self.profiling_enabled:
            self.profiler = Profiler(self.args)
            self.profiler.open_window(
                self._pipeline, self._sceneContext.width,
                self._sceneContext.height)

    def _create_training_services(self) -> None:
        scene = self._sceneContext
        artifacts = RunArtifacts.create(self.paths.root, self.args)
        # The reporter also publishes to the viewer state, so the panel shows the logged numbers.
        reporter = TrainingReporter(artifacts, self.args, scene,
                                    state=self._viewer.state)
        reporter.write_header()

        config = self._config
        self.learning_rate_schedule = LearningRateSchedule(
            base=config.base_lrs,
            extent=scene.extent,
            mean_steps=config.position_lr_max_steps,
            iterations=config.schedule.optimization_steps,
            scale_rotation_decay=config.scale_rotation_lr_decay,
        )

        self.extinction_hardness_schedule = ExtinctionHardnessSchedule(
            target=self.args.extinction_hardness,
            warmup=self.args.extinction_hardness_warmup,
        )

        self.appearance_schedule = AppearanceSchedule(appearanceConfig=config.appearance)

        queue_limiter = QueueLimiter(self._pipeline.runtime)

        densifier = DensificationController(
            pipeline=self._pipeline,
            args=self.args,
            densify_mode=self._densify_mode,
            densify_cfg=config.densify,
            structure_cfg=config.structure,
            extent=scene.extent,
            image_width=scene.width,
            image_height=scene.height,
            morton_world_min=scene.morton_min,
            morton_world_max=scene.morton_max,
            extinction_hardness_schedule=self.extinction_hardness_schedule,
            queue_limiter=queue_limiter,
        )

        self.runtime = TrainingRuntime(
            pipeline=self._pipeline,
            viewer=self._viewer,
            profiler=self.profiler,
            artifacts=artifacts,
            queue_limiter=queue_limiter,
            densifier=densifier,
            reporter=reporter,
        )

    def build_training_loop(self) -> "TrainingLoop":
        """The configured loop, not yet run; ``prepare_training_context`` must have run first."""
        return TrainingLoop(
            args=self.args,
            scene=self._sceneContext,
            targets=self._targets,
            runtime=self._training_runtime,
            config=self._config,
            target_textures=self.target_textures,
            structure_textures=self.structure_textures,
            learning_rate_scheduler=self._learning_rate_schedule, # THis should not be an individual attribute
            extinction_hardness_scheduler=self.extinction_hardness_schedule, # same for this
            appearance_scheduler=self._appearance_schedule,
            paths=self.paths,
        )

    def _write_resolved_config(self) -> None:
        sceneContext = self._sceneContext
        resolved = self._config
        base_lrs = resolved.base_lrs
        config = {
            "scene": self.args.scene,
            "iters": self.args.iters,
            "width": sceneContext.width,
            "height": sceneContext.height,
            "preview_every": self.args.preview_every,
            "log_every": self.args.log_every,
            "densify_every": self.args.densify_every,
            "densify_from": self.args.densify_from,
            "densify_to": self.args.densify_to,
            "densify_config": {
                "t_opacity": resolved.densify.t_opacity,
                "t_big_scale": resolved.densify.t_big_scale,
                "grow_scale3d": self.args.grow_scale3d,
            },
            "max_splats": self.args.max_splats,
            "z_near_cull": self.args.z_near_cull,
            "backend": self.args.backend,
            "seed": self.args.seed,
            "sample_mode": self.args.sample_mode,
            "mip_enabled": bool(self.args.mip_splatting),
            "mip_kernel_size": resolved.raster.kernel_size,
            "run_name": self.run_name,
            "n_cams": sceneContext.camera_count,
            "n_init_points": sceneContext.initial_gaussian_count,
            "cli_args": dict(vars(self.args)),
            "resolved": {
                "densify_mode": resolved.densify_mode.value,
                "downsample": sceneContext.colmapScene.downsample,
                "width": sceneContext.width,
                "height": sceneContext.height,
                "n_cams": sceneContext.camera_count,
                "n_init_points": sceneContext.initial_gaussian_count,
                "n_train": len(sceneContext.train_indices),
                "lr_schedule_steps": resolved.schedule.optimization_steps,
                "n_test": len(sceneContext.test_indices),
                "holdout_stride": sceneContext.holdout_stride,
                "eff_lambda_l2": resolved.loss.l2_weight,
                "eff_lambda_ssim": resolved.loss.ssim_weight,
                "base_lr_scale": base_lrs.scale,
                "base_lr_rot": base_lrs.rot,
                "base_lr_opacity": base_lrs.opacity,
                "base_lr_color": base_lrs.color,
                "base_lr_sh_rest": base_lrs.sh_rest,
                "scale_rotation_lr_decay": list(resolved.scale_rotation_lr_decay),
            },
        }
        (self.paths.root / "config.json").write_text(json.dumps(config, indent=2))

    def _write_final_test_psnr(self, result: TrainingResult) -> None:
        # --no-evaluation vetoes this: it renders every held-out camera.
        want_report = bool(getattr(self.args, "run_report", False))
        if (self.args.profile_run or not self.args.evaluation
                or not (self.args.final_test_psnr or want_report)):
            return

        self._publish_finalize("Scoring held-out views", 0.05)
        sceneContext = self._sceneContext
        # One render pass serves both files: the mean of the per-view metrics is the paper metric.
        per_view: list[dict] = []
        quality: dict = {}
        if want_report:
            renders_dir = (self.paths.root / "test_renders"
                           if getattr(self.args, "save_test_renders", False)
                           else None)
            per_view, quality = run_report_module.evaluate_test_views(
                self._pipeline, sceneContext.colmapScene,
                sceneContext.test_indices, self._targets.srgb,
                render_dir=renders_dir,
                with_lpips=bool(getattr(self.args, "eval_lpips", False)),
                progress=lambda done, total: self._publish_finalize(
                    f"held-out view {done}/{total}", 0.05 + 0.5 * done / max(total, 1)),
                checkpoint=self._training_runtime.viewer.checkpoint,
            )
            psnr = quality.get("psnr_srgb") or 0.0
            count = self._pipeline.primitives.count()
        else:
            psnr, count = eval_test_psnr_srgb(
                self._pipeline, sceneContext.colmapScene,
                sceneContext.test_indices, self._targets.srgb,
                checkpoint=self._training_runtime.viewer.checkpoint)

        self._training_runtime.viewer.checkpoint()
        active_s = max(0.0, result.wall_time_seconds - result.observation_seconds)
        _, overflows = self._pipeline.buffers.sort.stats.to_numpy().view(np.uint32)
        self._memory_summary["heldout_sort_overflows"] = int(overflows) - self._memory_summary["sort_overflows"]

        if self.args.final_test_psnr:
            (self.paths.root / "test_psnr.json").write_text(json.dumps({
                "psnr_srgb": psnr,
                "n_gaussians": count,
                "n_test_cams": len(sceneContext.test_indices),
                "train_wall_s": result.wall_time_seconds,
                # These separate the loop's own cost from the cost of observing it.
                "train_active_s": active_s,
                "observation_s": result.observation_seconds,
                "iters": result.iterations,
                "flat_bucket_backward": bool(self.args.flat_bucket_backward),
                **self._memory_summary,
            }, indent=2))

            print(f"[final-test-psnr] TEST PSNR sRGB = {psnr:.3f} dB over "
                  f"{len(sceneContext.test_indices)} held-out cams  (n_gaussians={count})")

        if want_report:
            frames = int(getattr(self.args, "render_fps_frames", 0) or 0)
            render = (run_report_module.measure_render_fps(
                          self._pipeline, sceneContext.colmapScene,
                          sceneContext.test_indices[0], frames,
                          checkpoint=self._training_runtime.viewer.checkpoint)
                      if frames > 0 and sceneContext.test_indices
                      else {"frames": 0, "fps": None, "ms_per_frame": None})
            self._training_runtime.viewer.checkpoint()
            report = run_report_module.build_report(
                pipeline=self._pipeline,
                scene_context=sceneContext,
                args=self.args,
                run_name=self.run_name,
                result=result,
                per_view=per_view,
                quality=quality,
                render=render,
                loop_timing={
                    "observation_s": result.observation_seconds,
                    "observation_total_s": result.observation_total_seconds,
                },
                process_started_at=_PROCESS_STARTED_AT,
            )
            report["memory"].update(self._memory_summary)
            if result.stop_reason:
                report["stop_reason"] = result.stop_reason
            if result.stage_timing is not None:
                report["stage_timing"] = result.stage_timing.summary()
            path = run_report_module.write_run_report(report, self.paths.root)
            fps = render.get("fps")
            print(f"[run-report] {len(per_view)} held-out views  "
                  f"psnr={psnr:.3f} dB  ssim={quality.get('ssim_srgb') or 0.0:.4f}  "
                  f"active={active_s:.1f}s (wall {result.wall_time_seconds:.1f}s, "
                  f"observation {result.observation_seconds:.1f}s)  "
                  f"n_g={count}"
                  + (f"  render={fps:.1f} fps" if fps else "")
                  + f"  -> {path}")

    @property
    def _sceneContext(self) -> SceneContext:
        assert self.scene_context is not None
        return self.scene_context

    @property
    def _targets(self) -> TargetDataset:
        assert self.targets is not None
        return self.targets

    @property
    def _training_runtime(self) -> TrainingRuntime:
        assert self.runtime is not None
        return self.runtime

    @property
    def _config(self) -> RunConfig:
        assert self.config is not None
        return self.config

    @property
    def _densify_mode(self) -> DensificationMode:
        return self._config.densify_mode

    @property
    def _pipeline(self) -> TrainingPipeline:
        assert self.pipeline is not None
        return self.pipeline

    @property
    def _viewer(self) -> ViewerSession:
        assert self.viewer is not None
        return self.viewer

    @property
    def _learning_rate_schedule(self) -> LearningRateSchedule:
        assert self.learning_rate_schedule is not None
        return self.learning_rate_schedule

    @property
    def _appearance_schedule(self) -> AppearanceSchedule:
        assert self.appearance_schedule is not None
        return self.appearance_schedule

class TrainingLoop:
    """Executes optimizer steps, then scheduled mutations, then observations."""

    convergence: ConvergenceMonitor | None = None

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        scene: SceneContext,
        targets: TargetDataset,
        runtime: TrainingRuntime,
        config: RunConfig,
        target_textures: dict[int, Any],
        structure_textures: dict[int, Any],
        learning_rate_scheduler: LearningRateSchedule,
        extinction_hardness_scheduler: ExtinctionHardnessSchedule | None = None,
        appearance_scheduler: AppearanceSchedule | None = None,
        paths: RunPaths,
    ) -> None:
        self.args = args
        self.scene = scene
        self.targets = targets
        self.runtime = runtime
        self.densify_mode = config.densify_mode
        self.densify_config = config.densify
        self.target_textures = target_textures
        self.structure_textures = structure_textures
        self.learning_rates = learning_rate_scheduler
        self.extinction_hardness_schedule: ExtinctionHardnessSchedule | None = extinction_hardness_scheduler
        self.appearance_schedule: AppearanceSchedule | None = appearance_scheduler
        self.paths = paths

        rng = np.random.default_rng(args.seed)
        self.sampler = CameraSampler(
            args.sample_mode,
            scene.train_indices,
            scene.test_indices,
            args.seed,
            rng,
        )
        self.schedule = TrainingSchedule(config.schedule)
        growth_end = config.schedule.densify_to
        schedule_steps = config.schedule.optimization_steps
        score_steps = (200 if config.schedule.until_convergence and growth_end >= schedule_steps
                       else min(200, max(0, (schedule_steps - growth_end) // 2)))
        self.image_prune_steps = (
            range(growth_end + 1, growth_end + score_steps + 1)
            if args.final_splat_fraction < 1 and self.densify_mode != DensificationMode.NONE
            else range(0))
        self.deferred_morton = False
        self.stop_reason = None
        if config.schedule.until_convergence:
            last_growth = growth_end if self.densify_mode != DensificationMode.NONE else 0
            self.convergence = ConvergenceMonitor(
                scene.train_indices, max(schedule_steps, last_growth, self.image_prune_steps.stop - 1,
                                         config.appearance.sh_degree * config.appearance.band_interval))

        pipeline = self.runtime.pipeline
        self.stbn_sampler = (StbnSampler(pipeline.stbn_config, pipeline.stbn_slices)
                             if pipeline.stbn_config.enabled else None)

        self.training_step = 0
        self.gaussian_count = scene.initial_gaussian_count
        self.eval_time_total = 0.0
        self.psnr_curve_records: list[dict] = []
        self.started_at = 0.0
        self.post_iteration_clean_timing = 0.0
        # Wall-clock spent inside _produce_observations.
        self.observation_seconds = 0.0
        self.observation_in_wall_seconds = 0.0
        self.instrumentation = LoopInstrumentation(
            profiler=runtime.profiler,
            stage_timing=StageTimingRecorder(every=int(args.stage_timing_every or 0),
                                             total_iters=int(args.iters)),
        )

    def launch_training_loop(self) -> TrainingResult:
        self._begin_training_loop()

        try:
            while self._should_continue():
                self.runtime.viewer.checkpoint()
                it_context = self._next_iteration()
                self._prepare_iteration(it_context)
                self._execute_training_step(it_context)

                # Freeze training time before the final iteration's observations.
                last_step = it_context.step_number == self.args.iters
                if last_step and self.convergence is None:
                    self._record_current_timing()

                self._apply_mutations(it_context)
                if self.convergence is not None:
                    if self._check_convergence(it_context.step_number):
                        self.stop_reason = "plateau"
                        last_step = True
                        if self.args.evaluation and self.args.eval_every is None:
                            it_context = replace(it_context, events=replace(it_context.events, eval=True))
                    if last_step:
                        self._record_current_timing()
                observed_at = time.perf_counter()
                try:
                    should_continue = self._produce_observations(it_context)
                finally:
                    spent = time.perf_counter() - observed_at
                    self.observation_seconds += spent
                    if not last_step:
                        self.observation_in_wall_seconds += spent
                self.instrumentation.end_iteration(self.runtime.pipeline)

                if last_step or not should_continue:
                    break
        except RunStopped:
            pass

        return self._complete_training_loop()

    def advance_to_capture(self, capture_step: int) -> IterationContext:
        """Run the loop up to, but not including, the train_step at ``capture_step``.
        Mirrors ``Profiler.run_capture_if_due``'s freeze point and drains the queue before
        returning that iteration's context."""
        self._begin_training_loop()
        while self._should_continue():
            self.runtime.viewer.checkpoint()
            context = self._next_iteration()
            self._prepare_iteration(context)
            if context.step_number >= capture_step:
                self.runtime.pipeline.runtime.wait_for_idle()
                return context
            self._execute_training_step(context)
            self._apply_mutations(context)
            if not self._produce_observations(context):
                break
        raise RuntimeError(
            f"training ended before reaching capture step {capture_step}")

    def _begin_training_loop(self) -> None:
        if self.args.wait_for_viewer:
            self.runtime.viewer.wait_for_start()
        self.runtime.viewer.set_training_active(True)
        self.started_at = time.time()
        self.runtime.reporter.start(self.started_at)

    def _should_continue(self) -> bool:
        return (
            (not self.args.iters or self.training_step < self.args.iters)
            and not self.instrumentation.should_stop()
            # The viewer's Stop ends the run early but still finalizes, keeping a usable result.
            and not self.runtime.viewer.stop_requested()
        )

    def _stbn_variables(self, camera_index: int) -> StbnVariables | None:
        if self.stbn_sampler is None:
            return None
        return self.stbn_sampler.variables(camera_index, self.training_step)

    def _check_convergence(self, step: int) -> bool:
        monitor = self.convergence
        if not monitor.due(step):
            return False
        score, _ = eval_test_psnr_srgb(self.runtime.pipeline, self.scene.colmapScene, monitor.cameras,
                                       self.targets.srgb, checkpoint=self.runtime.viewer.checkpoint)
        plateau = monitor.observe(score)
        self.runtime.artifacts.log_json.write(json.dumps({
            "event": "convergence_probe", "iter": step, "train_psnr_srgb": score,
            "train_indices": monitor.cameras, "stale_checks": monitor.stale_checks,
        }) + "\n")
        print(f"[quality+] step {step:,}, train probe {score:.3f} dB, "
              f"{monitor.stale_checks}/3 checks below the improvement threshold", flush=True)
        return plateau

    def _next_iteration(self) -> IterationContext:
        self.training_step += 1
        camera_index = self.sampler.next()
        camera_entry = self.scene.colmapScene.images[camera_index]
        score = self._should_accumulate_structure(self.training_step)
        lr_step = min(self.training_step, self.schedule.config.optimization_steps)
        learning_rates = self.learning_rates.at(lr_step - 1)
        if self.args.lr_recipe == LearningRateRecipe.SHORT:
            pipeline = self.runtime.pipeline
            learning_rates, pipeline.loss.config = self.learning_rates.short_objective(
                lr_step, learning_rates, pipeline.loss_config)

        return IterationContext(
            step_number=self.training_step,
            camera_index=camera_index,  # 0-based; the COLMAP id in camera_entry is 1-indexed
            camera_entry=camera_entry,
            learning_rates=learning_rates,
            events=self.schedule.events_at(self.training_step),

            iteration_contextual_variables=IterationContextualVariables(
                hardness_value=(self.extinction_hardness_schedule.at(self.training_step) if self.extinction_hardness_schedule is not None else 1.0),
                sh_degree=(self.appearance_schedule.at(self.training_step) if self.appearance_schedule is not None else int(self.args.sh_degree)),
                densification_variables=DensificationProperties(
                    structure_tensor_texture=self.structure_textures[camera_index] if score else None,
                    # TODO: normalize the hyperparameters instead of rescaling this threshold here.
                    structure_kwargs={"per_view_grad_thresh": (self.args.sad_freq_grad_threshold * self.scene.width * self.scene.height * 3.0) if score else None},
                    track_blend_weight=self._tracks_blend_weight(self.training_step),
                    track_screen_grad=self._tracks_screen_grad(self.training_step),
                ),
                contextual_stbn=self._stbn_variables(camera_index),
            ),
        )

    def _prepare_iteration(self, context: IterationContext) -> None:
        self.runtime.pipeline.targets.select(self.target_textures[context.camera_index])

        if context.step_number == 1:
            print(f"[determinism] iter 1 cam_idx={context.camera_index} (should match across same-seed runs)")

        self.runtime.queue_limiter.before_submit()

    def _execute_training_step(self, context: IterationContext) -> None:
        pipeline = self.runtime.pipeline
        with self.instrumentation.step(context, pipeline, self.runtime.queue_limiter):
            pipeline.train_step(context.camera_entry.camera, lrs=context.learning_rates,
                                contextual=context.iteration_contextual_variables)
        self.runtime.queue_limiter.record(pipeline.last_submit_id)
        if context.step_number % self.args.stream_every == 0:
            self.runtime.viewer.push_snapshot(pipeline)

    def _record_current_timing(self) -> None:
        # Drain the GPU so the final timing includes all submitted work.
        self.runtime.pipeline.runtime.wait_for_idle()
        self.post_iteration_clean_timing = time.time() - self.started_at


    def _apply_mutations(self, context: IterationContext) -> None:
        with self.instrumentation.mutations(context, self.runtime.pipeline):
            self._execute_mutations(context)

    def _execute_mutations(self, context: IterationContext) -> None:
        events = context.events
        pipeline = self.runtime.pipeline

        if events.densify:
            outcome = self.runtime.densifier.run(
                context.step_number, events, self.gaussian_count)
            self.gaussian_count = outcome.count
            self.runtime.reporter.report_densify(outcome)

        if events.opacity_reset:
            pipeline.opacity_reset.reset(self.densify_config)
            self.runtime.reporter.report_opacity_reset(context.step_number)

        step = context.step_number
        morton_due, mip_due = events.morton, events.mip_periodic
        if step in self.image_prune_steps:
            if step == self.image_prune_steps.start:
                pipeline.densify_stats.clear()  # reuse the now-idle blend-weight buffer
            accumulate_removal_cost(pipeline, step, context.iteration_contextual_variables.hardness_value)
            if step + 1 == self.image_prune_steps.stop:
                stats = prune_by_removal_cost(pipeline, self.args.final_splat_fraction)
                self.gaussian_count = stats["final_count"]
                self.runtime.reporter.report_densify(DensifyOutcome(step, self.gaussian_count, "image_prune", stats))
                morton_due |= self.deferred_morton
                mip_due |= pipeline.raster_config.mip_enabled
            else:
                # Keep splat indices stable from the first measurement until compaction.
                self.deferred_morton |= morton_due
                morton_due = False
            self.runtime.queue_limiter.record(pipeline.last_submit_id)

        if morton_due:
            pipeline.morton.reorder(
                self.scene.morton_min,
                self.scene.morton_max,
                carry_eta=(self.densify_mode == DensificationMode.STRUCTURE_AWARE),
            )
            pipeline.densify_stats.clear()

            if pipeline.last_submit_id:
                self.runtime.queue_limiter.record(pipeline.last_submit_id)

        if mip_due:
            pipeline.mip_filter.compute()

    def _produce_observations(self, context: IterationContext) -> bool:
        self.runtime.viewer.checkpoint()
        self.runtime.reporter.report_throughput(context)

        if context.step_number > 1 and (context.events.log or context.events.preview):
            ok, self.gaussian_count = self.runtime.reporter.report_iteration(
                context=context,
                pipeline=self.runtime.pipeline,
                target=self.targets.srgb[context.camera_index][..., :3],
                current_count=self.gaussian_count,
            )
            if not ok:
                return False

        if context.step_number > 1 and context.events.render_all:
            state = self.runtime.viewer.state
            state.begin_activity("Rendering every camera")
            with self.runtime.viewer.unthrottled():
                try:
                    render_all_cameras(
                        self.runtime.pipeline,
                        self.scene.colmapScene,
                        context.step_number,
                        self.paths.cameras,
                        checkpoint=self.runtime.viewer.checkpoint,
                    )
                finally:
                    state.end_activity()

        self._record_psnr_curve_if_due(context)
        self._run_evaluation_if_due(context)
        return True

    def _should_accumulate_structure(self, training_step) -> bool:
        return (
            self.densify_mode == DensificationMode.STRUCTURE_AWARE
            and self.args.densify_from < training_step <= self.args.densify_to
            and training_step % self.args.sad_accum_every == 0
        )

    def _tracks_blend_weight(self, training_step: int) -> bool:
        """Only the windows that end on a blend-pruning tick need blend weights."""
        schedule = self.schedule.config
        first_window = schedule.blend_prune_start - schedule.densify_every
        return (schedule.prune_blend_weight > 0.0
                and first_window < training_step <= schedule.densify_to)

    def _tracks_screen_grad(self, training_step: int) -> bool:
        """No densify tick reads the screen gradient after the last one."""
        return (self.densify_mode != DensificationMode.NONE
                and training_step <= self.args.densify_to)

    def _record_psnr_curve_if_due(self, context: IterationContext) -> None:
        stream = self.runtime.artifacts.psnr_curve_json
        if stream is None or not context.events.psnr_curve:
            return
        started = time.time()
        psnr, count = eval_test_psnr_srgb(
            self.runtime.pipeline,
            self.scene.colmapScene,
            self.scene.test_indices,
            self.targets.srgb,
            checkpoint=self.runtime.viewer.checkpoint,
        )
        record = {
            "iter": context.step_number,
            "psnr_srgb": psnr,
            "n_gaussians": count,
        }
        self.psnr_curve_records.append(record)
        stream.write(json.dumps(record) + "\n")
        self.eval_time_total += time.time() - started
        print(f"[psnr-curve] iter {context.step_number:6d}  psnr_srgb={psnr:5.2f}  "
              f"n_g={count} (pre-densify)")

    def _run_evaluation_if_due(self, context: IterationContext) -> None:
        stream = self.runtime.artifacts.eval_curve_json
        if stream is None or not context.events.eval:
            return
        # The sweep reports its own progress: the iteration counter cannot show it.
        state = self.runtime.viewer.state
        state.begin_activity("Evaluating every camera", 0.0)
        # The sweep blocks the loop, so the viewer has the queue to itself meanwhile.
        with self.runtime.viewer.unthrottled():
            try:
                self.eval_time_total = run_eval_sweep(
                    pipeline=self.runtime.pipeline,
                    scene=self.scene.colmapScene,
                    it=context.step_number,
                    test_indices=self.scene.test_indices,
                    targets_srgb=self.targets.srgb,
                    targets_linear=self.targets.linear,
                    cams_dir=self.paths.cameras,
                    eval_curve_json=stream,
                    t_start=self.started_at,
                    eval_time_total=self.eval_time_total,
                    eval_holdout=bool(self.args.eval_holdout),
                    with_lpips=bool(getattr(self.args, "eval_lpips", False)),
                    progress=lambda done, total: state.set_progress(
                        "Evaluating every camera",
                        (done / total) if total else None,
                        detail=f"camera {done}/{total}"),
                    checkpoint=self.runtime.viewer.checkpoint,
                )
            finally:
                state.end_activity()

    def _complete_training_loop(self) -> TrainingResult:
        # Training is done, so give the viewer the GPU back before the held-out PSNR pass.
        self.runtime.viewer.set_training_active(False)
        self.runtime.pipeline.runtime.wait_for_idle()
        if not self.post_iteration_clean_timing:
            self.post_iteration_clean_timing = time.time() - self.started_at
        if self.extinction_hardness_schedule is not None:
            self.runtime.pipeline.set_extinction_hardness(self.extinction_hardness_schedule.target)
        self.runtime.reporter.report_complete(self.training_step, self.post_iteration_clean_timing)
        if self.convergence is not None:
            self.stop_reason = self.stop_reason or (
                "iteration_limit" if self.args.iters > 0 and self.training_step == self.args.iters else "interrupted")
            self.runtime.artifacts.log_json.write(json.dumps({
                "event": "training_stop", "iter": self.training_step, "stop_reason": self.stop_reason,
            }) + "\n")
            message = f"Quality+ finished at step {self.training_step:,} ({self.stop_reason})."
            self.runtime.viewer.state.log(message)
            print(message)
            self.instrumentation.stage_timing.total_iters = self.training_step
        stage_timing = self.instrumentation.finish(self.paths.root,
                                                   self.post_iteration_clean_timing)
        return TrainingResult(
            iterations=self.training_step,
            wall_time_seconds=self.post_iteration_clean_timing,
            psnr_curve=self.psnr_curve_records,
            started_at=self.started_at,
            observation_seconds=self.observation_in_wall_seconds,
            observation_total_seconds=self.observation_seconds,
            stage_timing=stage_timing,
            stop_reason=self.stop_reason,
        )


def main() -> None:
    app = TrainingApplication.from_cli()
    with ErrorLog(app.paths.root / "errors.log") as errors:
        app.error_log = errors
        try:
            if app.args.hub:
                from trainer.frontend.hub import TrainingHub
                TrainingHub(app).run()
            else:
                app.start_training()
        except Exception as error:
            state = app.viewer.state if app.viewer is not None else None
            report_exception("Application failed", error, state)
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()
