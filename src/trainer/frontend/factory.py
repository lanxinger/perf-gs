# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Builds a run's GPU device, queue lock, Slang session and training pipeline from its config."""

from __future__ import annotations

import argparse
import threading
from typing import Any

import slangpy as spy

from trainer.backend.compute.device import create_device, create_slang_session
from trainer.backend.render.pipeline import TrainingPipeline
from trainer.backend.render.stbn_table import pixels_per_tile
from trainer.config.config import ShaderSpecialization
from trainer.config.config_types import SplatKernel
from trainer.config.resolve import RunConfig, shader_specialization
from trainer.frontend.scene import SceneContext


def shader_defines(spec: ShaderSpecialization) -> dict[str, str] | None:
    """The Slang macros of a specialization; None keeps every shader default."""
    defines: dict[str, str] = {}
    if spec.backward_pixel_stride != 1:
        defines["BACKWARD_PIXEL_STRIDE"] = f"{spec.backward_pixel_stride}u"
    if spec.hardness_is_one:
        defines["COMPOSITOR_HARDNESS_ONE"] = "1"
    if spec.stbn_enabled:
        defines["STBN_ENABLED"] = "1"
    if spec.stbn_staged_fraction > 0.0:
        defines["STBN_MAX_TILE_PIXELS"] = f"{pixels_per_tile(spec.stbn_staged_fraction)}u"
    if spec.adam_bf16:
        defines["ADAM_MOMENTS_BF16"] = "1"
    if spec.splat_kernel is not SplatKernel.GAUSSIAN:
        defines["SPLAT_KERNEL"] = str(list(SplatKernel).index(spec.splat_kernel))
    if spec.splat_profile_power != 1.0:
        defines["SPLAT_PROFILE_POWER"] = repr(float(spec.splat_profile_power))
    if spec.splat_profile_onset != 0.0:
        defines["SPLAT_PROFILE_ONSET"] = repr(float(spec.splat_profile_onset))
    return defines or None


def create_training_device(args: argparse.Namespace) -> spy.Device:
    # Vulkan keeps resource labels only with debug layers, so enable them for external capture.
    debug_names = args.nsight_debug_names
    if debug_names is None:
        debug_names = bool(args.capture_external)
    device = create_device(
        backend=args.backend,
        enable_debug=debug_names,
        shader_debug_info=args.profile_run,
        defines=shader_defines(shader_specialization(args)),
    )
    if args.profile_run:
        print("[profile-run] shaders compiled with source-level debug info "
              "(SPIR-V NonSemantic.Shader.DebugInfo) for Nsight correlation.")
    if debug_names:
        print("[profile-run] Vulkan debug layers ON so resource debug names "
              "(VK_EXT_debug_utils) reach Nsight's Object Browser / Memory view.")
    return device


def create_gpu_queue_lock(*, viewer_enabled: bool) -> Any:
    """The lock the trainer and the viewer thread share around queue submits."""
    return threading.RLock() if viewer_enabled else None


class SlangSessions:
    """One Slang session per shader specialization, so a hub run can change compile-time
    settings on the hub's one device. The device's own session serves the defines it has."""

    def __init__(self, device: spy.Device) -> None:
        self._device = device
        own = dict(device.slang_session.desc.compiler_options.defines or {})
        self._backend_defines = {
            k: v for k, v in own.items() if k == "TRAINER_METAL_BACKEND"
        }
        self._sessions = {frozenset(own.items()): device.slang_session}

    def session_for(self, spec: ShaderSpecialization) -> spy.SlangSession:
        defines = {**self._backend_defines, **(shader_defines(spec) or {})}
        key = frozenset(defines.items())
        if key not in self._sessions:
            self._sessions[key] = create_slang_session(self._device, defines)
        return self._sessions[key]


def create_pipeline(*, device: spy.Device, gpu_lock: Any, scene: SceneContext,
                    max_splats: int, config: RunConfig, max_sort_pairs: int | None = None,
                    initial_state: dict | None = None,
                    slang_session: spy.SlangSession | None = None) -> TrainingPipeline:
    capacity_override = {} if max_sort_pairs is None else {"max_sort_pairs": max_sort_pairs}
    return TrainingPipeline(
        device=device,
        width=scene.width,
        height=scene.height,
        max_splats=max_splats,
        gpu_lock=gpu_lock,
        raster_config=config.raster,
        loss_config=config.loss,
        regularization_config=config.regularization,
        appearance_config=config.appearance,
        stbn_config=config.stbn,
        initial_state=initial_state,
        densify_mode=config.densify_mode,
        slang_session=slang_session,
        **capacity_override,
    )
