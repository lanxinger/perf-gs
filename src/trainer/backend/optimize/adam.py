# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Adam update of the packed primitive parameters and the SH rest coefficients."""

from __future__ import annotations

from typing import TYPE_CHECKING

from trainer.backend.compute.gpu import CommandRecorder
from trainer.backend.compute.kernels import KernelLibrary
from trainer.backend.memory.buffers import (
    AdamResources,
    GradientResources,
    PrimitiveStorage,
    SphericalHarmonicsResources,
)
from trainer.common.sh import sh_rest_coeffs

if TYPE_CHECKING:
    import slangpy as spy

    from trainer.config.config import AdamLRs

# Must match optimizers/adam.slang.
_BETA1 = 0.9
_BETA2 = 0.999


class AdamOptimizer:
    """Owns the Adam dispatches, the moments and the bias-correction step count."""

    KERNELS = {
        "prepare_args": ("densify/densify_dispatch.slang", "prepareDensifyArgs"),
        "adam": ("optimizers/adam.slang", "adamStep"),
        "adam_rest": ("optimizers/adam.slang", "adamStepRest"),
        "clear_state": ("optimizers/grad_utils.slang", "clearAdamState"),
        "clear_grad": ("optimizers/grad_utils.slang", "clearGradBuffer"),
        "clear_uint": ("optimizers/grad_utils.slang", "clearUintBuffer"),
    }

    def __init__(self, commands: CommandRecorder, kernels: KernelLibrary, *,
                 primitive: PrimitiveStorage, adam: AdamResources,
                 gradients: GradientResources, sh: SphericalHarmonicsResources | None,
                 cluster_visible: spy.NDBuffer, capacity: int,
                 width: int, height: int) -> None:
        self._commands = commands
        self._k = kernels.load(self.KERNELS)
        self._primitive = primitive
        self._adam = adam
        self._gradients = gradients
        self._sh = sh
        self._cluster_visible = cluster_visible
        self._capacity = capacity
        self._pixels = float(width * height)
        self.step_count = 0

    def reset(self) -> None:
        self.step_count = 0

    def prepare_step_args(self) -> None:
        """Compute indirect dispatch args for ``step`` early so Metal can prefetch them
        on an earlier flush instead of stalling after the backward pass."""
        commands = self._commands
        count = self._primitive.active_count
        args = self._adam.dispatch_args
        commands.dispatch(self._k.prepare_args, "adam_prepare_args",
            thread_count=[1, 1, 1],
            count_buf=count.storage,
            dispatch_args=args.storage,
            max_splats=self._capacity)
        if commands.batch is not None:
            commands.batch.prefetch_indirect(args)

    def step(self, lrs: AdamLRs, sh_degree: int, *, args_prepared: bool = False) -> None:
        self.step_count += 1
        bc1 = 1.0 - _BETA1 ** float(self.step_count)
        bc2 = 1.0 - _BETA2 ** float(self.step_count)
        commands = self._commands
        count = self._primitive.active_count
        args = self._adam.dispatch_args

        if not args_prepared:
            commands.dispatch(self._k.prepare_args, "adam_prepare_args",
                thread_count=[1, 1, 1],
                count_buf=count.storage,
                dispatch_args=args.storage,
                max_splats=self._capacity)
        with commands.marker_group("adam_main"):
            commands.dispatch_indirect(self._k.adam, "adam", args,
                gaussians=self._primitive.values.storage,
                adam_state=self._adam.state.storage,
                num_gaussians_counter=count.storage,
                **self._gradients.param_kwargs(),
                cluster_visible=self._cluster_visible.storage,
                cfg={
                    "lr_mean": lrs.mean,
                    "lr_scale": lrs.scale,
                    "lr_rot": lrs.rot,
                    "lr_opacity": lrs.opacity,
                    "lr_color": lrs.color,
                    "bias_correction1": bc1,
                    "bias_correction2": bc2,
                    "step": self.step_count,
                },
                pixels_per_step=self._pixels)

        if self._sh is None:
            return
        rest_floats = self._sh.rest_floats
        active_rest = min(sh_rest_coeffs(sh_degree) * 3, rest_floats) if sh_degree else 0
        with commands.marker_group("adam_rest"):
            commands.dispatch_indirect(self._k.adam_rest, "adam_rest", args,
                features_rest=self._sh.features_rest.storage,
                features_rest_grad=self._sh.features_rest_grad.storage,
                adam_rest=self._sh.adam_rest.storage,
                num_gaussians_counter=count.storage,
                cluster_visible=self._cluster_visible.storage,
                lr_rest=lrs.sh_rest,
                bias_correction1=bc1,
                bias_correction2=bc2,
                pixels_per_step=self._pixels,
                rest_floats=rest_floats,
                active_rest_floats=active_rest,
                max_splats=self._capacity,
                step=self.step_count)

    def clear_state(self) -> None:
        """Zero every moment (after a wholesale parameter replacement)."""
        self._commands.dispatch(self._k.clear_state, "clear_adam_state",
            thread_count=[self._capacity, 1, 1],
            adam_state=self._adam.state.storage,
            max_splats=self._capacity)
        if self._sh is not None:
            count = self._capacity * self._sh.moment_planes
            self._commands.dispatch(self._k.clear_uint, "clear_adam_rest",
                thread_count=[count, 1, 1],
                buffer=self._sh.adam_rest.storage, count=count)

    def zero_grad(self) -> None:
        """Zero the parameter gradients and 2D seeds, for isolated forward/backward probes.

        Training steps never need it: projection and Adam reset what they consume.
        """
        for buffer in (*self._gradients.params.values(), *self._gradients.seeds.values()):
            self._commands.dispatch(self._k.clear_grad, "grad_clear",
                thread_count=[self._capacity, 1, 1],
                grad_buffer=buffer.storage,
                max_splats=self._capacity)
