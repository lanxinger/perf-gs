# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Test-only TrainingPipeline subclass: split steps and read intermediates back."""

from __future__ import annotations

import numpy as np

from trainer.backend.compute.execution import TILE_HEIGHT, TILE_WIDTH
from trainer.backend.memory.buffers import GRAD_SCALAR_COMPONENTS, decode_grad_buffer
from trainer.backend.render.pipeline import TrainingPipeline, default_sort_pairs
from trainer.config.config import (
    AdamLRs,
    DensificationProperties,
    IterationContextualVariables,
    StbnVariables,
)
from trainer.config.config_types import DensificationMode
from trainer.data.dataset import Camera


class TestPipeline(TrainingPipeline):
    """Pipeline plus test-only introspection / step-splitting helpers."""

    __test__ = False   # a pipeline, not a pytest test class

    def __init__(self, device, width, height, max_splats=500_000, max_sort_pairs=None, *args, **kwargs):
        # Numerical tests exercise both densifiers on the same small pipeline.
        kwargs.setdefault("densify_mode", DensificationMode.STRUCTURE_AWARE)
        if max_sort_pairs is None:
            # A splat can emit at most one pair per tile in this image.
            tiles = ((width + TILE_WIDTH - 1) // TILE_WIDTH) * ((height + TILE_HEIGHT - 1) // TILE_HEIGHT)
            max_sort_pairs = min(default_sort_pairs(max_splats, kwargs.get("viewer_only", False)), max_splats * tiles)
        super().__init__(device, width, height, max_splats, max_sort_pairs, *args, **kwargs)

    def train_step(self, camera: Camera, lrs: AdamLRs | None = None,
                   active_sh_degree: int | None = None, hardness_value: float = 1.0,
                   structure_tensor=None, structure_kwargs: dict | None = None,
                   stbn: StbnVariables | None = None) -> None:
        degree = self.active_sh_degree if active_sh_degree is None else int(active_sh_degree)
        super().train_step(camera, lrs=lrs if lrs is not None else AdamLRs(),
            contextual=IterationContextualVariables(
                hardness_value=hardness_value, sh_degree=degree,
                densification_variables=DensificationProperties(structure_tensor, structure_kwargs),
                contextual_stbn=stbn))

    def set_primitive_from_numpy(self, **snapshot) -> None:
        """Re-prime an existing pipeline (resets Adam + SH) so finite-difference
        probes can reuse one device instead of rebuilding per perturbation."""
        self.load_snapshot(**snapshot)

    def forward_and_loss(self, camera: Camera,
                         stbn: StbnVariables | None = None) -> None:
        """Forward + loss only (no wait); ``stbn=None`` samples every pixel."""
        with self.commands.recording("forward_and_loss"):
            self.rasterize(camera)
            self.record_loss(IterationContextualVariables(
                hardness_value=self.extinction_hardness,
                sh_degree=self.active_sh_degree,
                contextual_stbn=stbn,
            ))

    def step_split_for_active_band_test(self, camera: Camera, lrs: AdamLRs,
                                        adam_degree: int) -> None:
        """Compare active-band Adam without changing the forward/backward degree."""
        contextual = self._current_render_state()
        with self.commands.recording("train_step"):
            self.record_forward(camera, contextual)
            self.record_loss(contextual)
            self.record_backward(contextual)
            self.regularizer.record()
            self.optimizer.step(lrs, adam_degree)

    def run_backward(self, structure_tensor=None, structure_kwargs: dict | None = None,
                     stbn: StbnVariables | None = None) -> None:
        self.record_backward(IterationContextualVariables(
            hardness_value=self.extinction_hardness, sh_degree=self.active_sh_degree,
            densification_variables=DensificationProperties(structure_tensor, structure_kwargs),
            contextual_stbn=stbn))

    def regularize(self) -> None:
        """Only the opacity/scale regularization pass, onto the grad buffers."""
        self.regularizer.record()

    def read_gradients(self) -> dict[str, np.ndarray]:
        """The 14 SoA grad buffers as float64."""
        n = self.primitives.count()
        return {name: decode_grad_buffer(self.buffers.gradients.params[name], n)
                for name in GRAD_SCALAR_COMPONENTS}

    def read_loss(self) -> float:
        """CPU L1 between output and target, matching l1_l2_loss.slang (both sRGB)."""
        rendered = self.buffers.raster.output_image.to_numpy()[..., :3]
        target = self.buffers.loss.target_image.to_numpy()[..., :3]
        if target.dtype == np.uint8:
            target = target.astype(np.float32) / 255.0
        return float(np.sum(np.abs(rendered - target)))

    def cap_opacity(self, cap_value: float = 0.8) -> None:
        self.opacity_reset.cap(cap_value, marker="opacity_reset")
