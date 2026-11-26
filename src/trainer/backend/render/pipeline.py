# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""The training pipeline: composes the GPU components and records one training step."""

from __future__ import annotations

import contextlib
from dataclasses import replace

import numpy as np
import slangpy as spy

from trainer.backend.compute.algorithms import PrefixScan, RadixSort
from trainer.backend.compute.device import SHADERS_DIR
from trainer.backend.compute.gpu import CommandRecorder, GpuRuntime
from trainer.backend.compute.kernels import KernelLibrary
from trainer.backend.densify.morton import MortonReorder
from trainer.backend.densify.opacity import OpacityReset
from trainer.backend.densify.simple_densifier import SimpleDensifier
from trainer.backend.densify.statistics import DensifyStatistics
from trainer.backend.densify.structure_densifier import StructureDensifier
from trainer.backend.memory import buffers
from trainer.backend.memory.arena import ArenaCommitter
from trainer.backend.memory.primitive_io import PrimitiveIO, gaussian_records
from trainer.backend.optimize.adam import AdamOptimizer
from trainer.backend.optimize.regularization import Regularizer
from trainer.backend.render.backward import RasterBackward
from trainer.backend.render.losses import ImageLoss
from trainer.backend.render.mip_filter import MipFilter
from trainer.backend.render.rasterizer import Rasterizer
from trainer.backend.render.stbn_mask import StbnMask
from trainer.backend.render.targets import TargetImages
from trainer.common.sh import sh_degree_from_rest_floats, sh_rest_coeffs
from trainer.config.config import (
    AdamLRs,
    AppearanceConfig,
    IterationContextualVariables,
    LossConfig,
    RasterConfig,
    RegularizationConfig,
    StbnConfig,
)
from trainer.config.config_types import DensificationMode
from trainer.data.dataset import Camera


def _module_for_reflection(device: spy.Device,
                           session: spy.SlangSession | None = None) -> spy.Module:
    """The shared-layout module whose struct types size the buffers (bf16 Adam changes them)."""
    session = device.slang_session if session is None else session
    return spy.Module(session.load_module(str(SHADERS_DIR / "common" / "types.slang")))


def default_sort_pairs(max_splats: int, viewer_only: bool = False) -> int:
    """Sort-pair capacity for ``max_splats``; pairs past it are dropped (and counted)."""
    if viewer_only:
        return min(12_000_000, max(2_000_000, max_splats * 6))
    # A splat may overlap many tiles even when the population fits its growth budget.
    return max(buffers.DEFAULT_MAX_SORT_PAIRS, max_splats * 6)


def estimate_pipeline_bytes(module: spy.Module, *, max_splats: int, width: int,
                            height: int, sh_degree: int, structure_aware: bool = False,
                            viewer_only: bool = False, max_sort_pairs: int | None = None) -> int:
    """Base GPU bytes a pipeline of this shape allocates (no GPU work)."""
    sizes = {"max_splats": max_splats, "width": width, "height": height,
             "max_sort_pairs": max_sort_pairs or default_sort_pairs(max_splats, viewer_only),
             "sh_rest_floats": sh_rest_coeffs(sh_degree) * 3}
    if not viewer_only:
        sizes["structure_aware"] = structure_aware
    return buffers.estimate_allocation_bytes(module, viewer_only=viewer_only, **sizes)


class TrainingPipeline:
    """Owns the GPU resources and components of one scene at one resolution.

    ``viewer_only`` builds just the forward path, for the live viewer. ``slang_session``
    carries the shader specialization (default: the device's own).
    """

    def __init__(
        self,
        device: spy.Device,
        width: int,
        height: int,
        max_splats: int = 500_000,
        max_sort_pairs: int | None = None,
        gpu_lock: contextlib.AbstractContextManager | None = None,
        raster_config: RasterConfig | None = None,
        loss_config: LossConfig | None = None,
        regularization_config: RegularizationConfig | None = None,
        appearance_config: AppearanceConfig | None = None,
        stbn_config: StbnConfig | None = None,
        viewer_only: bool = False,
        initial_state: dict | None = None,
        use_persplat_backward: bool = True,
        use_flat_bucket_backward: bool = True,
        densify_mode: DensificationMode = DensificationMode.SIMPLE,
        slang_session: spy.SlangSession | None = None,
    ) -> None:
        self.device = device
        self.slang_session = device.slang_session if slang_session is None else slang_session
        self.gpu_lock = gpu_lock if gpu_lock is not None else contextlib.nullcontext()
        self.runtime = GpuRuntime(device, self.gpu_lock)
        self.commands = CommandRecorder(self.runtime)
        self.kernels = KernelLibrary(device, self.slang_session)
        self.width = width
        self.height = height
        self.max_splats = max_splats
        self.max_sort_pairs = max_sort_pairs or default_sort_pairs(max_splats, viewer_only)
        self.viewer_only = viewer_only
        self.shared_data_module = _module_for_reflection(device, self.slang_session)

        if raster_config is None and initial_state and "mip_kernel_size" in initial_state:
            raster_config = RasterConfig(mip_enabled=True, kernel_size=initial_state["mip_kernel_size"])
        raster_config = raster_config or RasterConfig()
        if viewer_only and raster_config.cluster_cull:
            raster_config = replace(raster_config, cluster_cull=False)
        self.raster_config = raster_config
        self.loss_config = loss_config or LossConfig()
        self.regularization_config = regularization_config or RegularizationConfig()
        rest = (initial_state or {}).get("features_rest")
        if appearance_config is None and rest is not None and "active_sh_degree" in initial_state:
            appearance_config = AppearanceConfig(sh_degree=sh_degree_from_rest_floats(rest.shape[1]))
        self.appearance_config = appearance_config or AppearanceConfig()
        self.stbn_config = stbn_config or StbnConfig()
        self.use_persplat_backward = use_persplat_backward
        self.use_flat_bucket_backward = use_flat_bucket_backward
        self._sh_rest_floats = sh_rest_coeffs(self.appearance_config.sh_degree) * 3
        self._structure_aware = densify_mode == DensificationMode.STRUCTURE_AWARE

        self.stbn = (StbnMask.disabled(device) if viewer_only
                     else StbnMask.for_config(device, self.stbn_config))
        self.buffers = self._allocate()
        self._build_components()

        # Render state of the last training step; renders outside one inherit it.
        self.active_sh_degree = 0
        self.extinction_hardness = 1.0

        if initial_state is not None:
            self.load_snapshot(**initial_state)
            if not self.viewer_only:
                self.optimizer.zero_grad()
                self.densify_stats.clear()
        if self.structure_densifier is not None:
            self.structure_densifier.clear_scores()

    def _allocation_sizes(self) -> dict:
        sizes = dict(max_splats=self.max_splats, width=self.width, height=self.height,
                     max_sort_pairs=self.max_sort_pairs, sh_rest_floats=self._sh_rest_floats)
        if not self.viewer_only:
            sizes["structure_aware"] = self._structure_aware
        return sizes

    def _allocate(self) -> buffers.GaussianBuffers:
        sizes = self._allocation_sizes()
        planned = buffers.estimate_allocation_bytes(
            self.shared_data_module, viewer_only=self.viewer_only, **sizes)
        role = "viewer" if self.viewer_only else "trainer"
        print(f"[memory] {role} base GPU resources: {planned / 1024**3:.2f} GiB "
              "(sampler resources and driver overhead excluded)")
        allocate = buffers.allocate_viewer if self.viewer_only else buffers.allocate
        return allocate(device=self.device, shared_data_module=self.shared_data_module, **sizes)

    def _build_components(self) -> None:
        b, commands, kernels = self.buffers, self.commands, self.kernels
        capacity = self.max_splats
        self.scan = PrefixScan(commands, kernels, b.scan)
        self.sort = RadixSort(commands, kernels, b.sort)
        self.primitives = PrimitiveIO(commands, kernels, b.primitive, capacity, b.sh)
        self.rasterizer = Rasterizer(
            self.device, commands, kernels, primitive=b.primitive, raster=b.raster,
            sh=b.sh, scan=self.scan, sort=self.sort, config=self.raster_config,
            width=self.width, height=self.height, num_tiles_x=b.num_tiles_x,
            num_tiles_y=b.num_tiles_y, capacity=capacity)
        self.targets = self.loss = self.backward = self.regularizer = None
        self.optimizer = self.densify_stats = self.mip_filter = self.opacity_reset = None
        self.simple_densifier = self.structure_densifier = self.morton = None
        if self.viewer_only:
            return

        visible = b.raster.cluster_visible
        self.targets = TargetImages(self.device, commands, b.loss.target_image,
                                    self.width, self.height)
        self.loss = ImageLoss(commands, kernels, resources=b.loss, width=self.width,
                              height=self.height, config=self.loss_config)
        self.backward = RasterBackward(
            commands, kernels, rasterizer=self.rasterizer, primitive=b.primitive,
            raster=b.raster, backward=b.backward, gradients=b.gradients,
            image_gradient=b.loss.image_gradient, tracking=b.densification, sh=b.sh,
            scan=self.scan, capacity=capacity, max_sort_pairs=self.max_sort_pairs,
            per_splat=self.use_persplat_backward, per_bucket=self.use_flat_bucket_backward,
            stbn_compacted=self.stbn_config.backward_compacted)
        self.regularizer = Regularizer(
            commands, kernels, primitive=b.primitive, gradients=b.gradients,
            cluster_visible=visible, config=self.regularization_config, capacity=capacity)
        self.optimizer = AdamOptimizer(
            commands, kernels, primitive=b.primitive, adam=b.adam, gradients=b.gradients,
            sh=b.sh, cluster_visible=visible, capacity=capacity,
            width=self.width, height=self.height)
        self.densify_stats = DensifyStatistics(
            commands, kernels, primitive=b.primitive, resources=b.densification,
            gradients=b.gradients, cluster_visible=visible, capacity=capacity,
            width=self.width, height=self.height)
        self.mip_filter = MipFilter(self.device, commands, kernels, primitive=b.primitive,
                                    resources=b.mip, capacity=capacity)
        self.opacity_reset = OpacityReset(commands, kernels, primitive=b.primitive,
                                          adam=b.adam, capacity=capacity)
        arena = ArenaCommitter(commands, kernels, b.densification.scratch_arena, capacity)
        self.simple_densifier = SimpleDensifier(
            commands, kernels, primitive=b.primitive, adam=b.adam,
            resources=b.densification, eta=b.eta, sh=b.sh, scan=self.scan, arena=arena,
            width=self.width, height=self.height, max_primitives=capacity)
        if b.eta is not None:
            self.structure_densifier = StructureDensifier(
                commands, kernels, primitive=b.primitive, adam=b.adam,
                resources=b.densification, eta=b.eta, split=b.structure,
                gradients=b.gradients, raster=b.raster, sh=b.sh, scan=self.scan,
                arena=arena, opacity=self.opacity_reset, width=self.width,
                height=self.height, capacity=capacity)
        self.morton = MortonReorder(
            commands, kernels, primitive=b.primitive, adam=b.adam, eta=b.eta, sh=b.sh,
            sort=self.sort, scan=self.scan, arena=arena, capacity=capacity)

    @property
    def last_submit_id(self) -> int:
        return self.commands.last_submit_id

    @property
    def iteration(self) -> int:
        """Adam step count; densification uses it as a decorrelation seed."""
        return self.optimizer.step_count if self.optimizer is not None else 0

    @property
    def stbn_slices(self) -> int:
        """Temporal period of the mask; per-camera visit counters wrap at it."""
        return self.stbn.slices

    def set_active_sh_degree(self, degree: int) -> None:
        self.active_sh_degree = max(0, min(int(degree), self.appearance_config.sh_degree))

    def set_extinction_hardness(self, hardness: float) -> None:
        self.extinction_hardness = float(hardness)

    def _current_render_state(self) -> IterationContextualVariables:
        return IterationContextualVariables(hardness_value=self.extinction_hardness,
                                            sh_degree=self.active_sh_degree)

    def _stbn_kwargs(self, contextual: IterationContextualVariables | None) -> dict:
        return self.stbn.step_kwargs(contextual.contextual_stbn if contextual is not None else None)

    def load_snapshot(self, means: np.ndarray, log_scales: np.ndarray,
                      rot_quats: np.ndarray, opacities: np.ndarray,
                      color_logits: np.ndarray, features_rest: np.ndarray | None = None,
                      natural_vignetting: float = 0.0, filter_3d: np.ndarray | None = None,
                      mip_kernel_size: float | None = None, active_sh_degree: int | None = None,
                      opacity_logits: np.ndarray | None = None, *, reset_optimizer: bool = True) -> None:
        """Replace all primitives. Offline rendering can skip resetting unused optimizer buffers."""
        self.rasterizer.natural_vignetting = natural_vignetting
        if filter_3d is not None:
            self.buffers.primitive.filter_3d.storage.copy_from_numpy(np.ascontiguousarray(filter_3d))
        if active_sh_degree is not None:
            self.set_active_sh_degree(active_sh_degree)
        self.set_gaussians(means, log_scales, rot_quats, opacities, color_logits, opacity_logits)
        if not self.viewer_only and reset_optimizer:
            self.optimizer.clear_state()
            self.optimizer.reset()
        if self.buffers.sh_enabled:
            if (not self.viewer_only and reset_optimizer) or features_rest is None:
                self.primitives.clear_features_rest()
            self.primitives.write_features_rest(features_rest)

    def set_gaussians(self, means: np.ndarray, log_scales: np.ndarray,
                      rot_quats: np.ndarray, opacities: np.ndarray,
                      color_logits: np.ndarray, opacity_logits: np.ndarray | None = None) -> None:
        """Upload primitives without touching optimizer state (valid on the viewer)."""
        records = gaussian_records(
            self.primitives.dtype, means=means, log_scales=log_scales,
            rot_quats=rot_quats, opacities=opacities, color_logits=color_logits)
        if opacity_logits is not None:
            records["opacity"] = opacity_logits
        self.primitives.write(records)

    def resize(self, max_splats: int, max_sort_pairs: int | None = None) -> None:
        """Re-allocate a render-only pipeline for ``max_splats``; contents are lost."""
        assert self.viewer_only, "resizing a training pipeline would drop its state"
        self.max_splats = max_splats
        self.max_sort_pairs = max_sort_pairs or default_sort_pairs(max_splats, viewer_only=True)
        # Release the old set first so both never coexist in VRAM.
        self.buffers = self.scan = self.sort = self.primitives = self.rasterizer = None
        self.buffers = buffers.allocate_viewer(
            device=self.device, shared_data_module=self.shared_data_module,
            **self._allocation_sizes())
        self._build_components()

    def on_hot_reload(self) -> None:
        """Refresh the reflection module after ``device.reload_all_programs()``."""
        self.shared_data_module = _module_for_reflection(self.device, self.slang_session)

    def record_forward(self, camera: Camera, contextual: IterationContextualVariables) -> None:
        self.rasterizer.record(camera, contextual)

    def record_loss(self, contextual: IterationContextualVariables) -> None:
        self.targets.flush()
        self.loss.record(rendered=self.buffers.raster.output_image,
                         target=self.targets.active,
                         stbn_kwargs=self._stbn_kwargs(contextual))

    def record_backward(self, contextual: IterationContextualVariables) -> None:
        self.backward.record(contextual, self._stbn_kwargs(contextual))
        variables = contextual.densification_variables
        if variables is not None and variables.structure_tensor_texture is not None:
            structure = variables.structure_tensor_texture
            if not isinstance(structure, spy.Texture):
                structure = self.targets.structure_texture(structure)
            kwargs = dict(variables.structure_kwargs or {})
            if kwargs.get("pass_idx") is None:
                kwargs["pass_idx"] = self.iteration
            with self.commands.marker_group("structure_accum"):
                self.structure_densifier.accumulate_scores(
                    structure, self.rasterizer.camera_buffer, **kwargs)
        if self.backward.tracks_screen_gradient(contextual):
            with self.commands.marker_group("accumulate_iter"):
                self.densify_stats.accumulate()

    def train_step(self, camera: Camera, *, lrs: AdamLRs,
                   contextual: IterationContextualVariables | None = None) -> None:
        """Forward, loss, backward, regularization and Adam in one submit."""
        if contextual is None:
            contextual = self._current_render_state()
        self.set_extinction_hardness(contextual.hardness_value)
        self.set_active_sh_degree(contextual.sh_degree)
        if self.commands._emulate_indirect and self.buffers.primitive.cached_count is None:
            if self.commands.last_submit_id:
                self.runtime.wait_for_submit(self.commands.last_submit_id)
            self.primitives.count()
        group = self.commands.marker_group
        with self.commands.recording("train_step"):
            pre_adam = self.commands._emulate_indirect
            if pre_adam:
                self.optimizer.prepare_step_args()
            with group("forward"):
                self.record_forward(camera, contextual)
            with group("loss"):
                self.record_loss(contextual)
            with group("backward"):
                self.record_backward(contextual)
            with group("regularization"):
                self.regularizer.record()
            with group("adam"):
                self.optimizer.step(lrs, contextual.sh_degree, args_prepared=pre_adam)

    def rasterize(self, camera: Camera,
                  contextual: IterationContextualVariables | None = None) -> None:
        if contextual is None:
            contextual = self._current_render_state()
        self.record_forward(camera, contextual)

    def render(self, camera: Camera) -> None:
        with self.commands.recording("render"):
            self.rasterize(camera)
        self.runtime.wait_for_idle()

    def read_output_image(self) -> np.ndarray:
        with self.gpu_lock:
            return self.buffers.raster.output_image.to_numpy()

    def read_bin_debug(self) -> tuple[np.ndarray, np.ndarray, int, int]:
        b, r = self.buffers, self.buffers.raster
        with self.gpu_lock:
            starts = r.tile_start.to_numpy().view(np.uint32)
            ends = r.tile_end.to_numpy().view(np.uint32)
            overlap = r.gaussian_count_per_pixel.to_numpy().view(np.uint32)
        n = b.num_tiles
        tile_counts = np.clip(ends[:n].astype(np.int64) - starts[:n].astype(np.int64),
                              0, None).astype(np.uint32)
        return tile_counts, overlap.reshape(self.height, self.width), b.num_tiles_x, b.num_tiles_y
