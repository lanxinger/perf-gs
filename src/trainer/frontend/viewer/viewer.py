# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Interactive in-process viewer sharing the trainer's GPU device and queue."""
from __future__ import annotations

import math
import os
import queue
import threading
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np
import slangpy as spy

from trainer.backend.compute.device import SHADERS_DIR
from trainer.backend.compute.gpu import GpuRuntime
from trainer.backend.render.pipeline import TrainingPipeline
from trainer.config.config import AppearanceConfig, RasterConfig
from trainer.data.dataset import Camera
from trainer.frontend.reporting.error_log import report_exception
from trainer.frontend.viewer.session import ERROR, MODE_HUB, Session
from trainer.frontend.viewer.ui import InputMap, Panel
from trainer.frontend.viewer.ui.sections import (
    Actions,
    CameraControls,
    configuration_section,
    controls_section,
    dataset_section,
    diagnostics_section,
    header,
    loss_section,
    result_section,
    training_section,
    viewer_section,
)

if TYPE_CHECKING:
    from trainer.frontend.scene import CameraSet

# No-scene backdrop, written straight into the non-sRGB present format, so already display-encoded.
BACKDROP = spy.float4(0.055, 0.065, 0.080, 1.0)


def _build_view_matrix(eye: np.ndarray, forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = forward / np.linalg.norm(forward)
    r = np.cross(up, f)
    r = r / np.linalg.norm(r)
    u = np.cross(f, r)
    M = np.eye(4, dtype=np.float32)
    M[0, :3] = r
    M[1, :3] = u
    M[2, :3] = f
    M[0, 3] = -np.dot(r, eye)
    M[1, 3] = -np.dot(u, eye)
    M[2, 3] = -np.dot(f, eye)
    return M


def _build_proj_matrix(fovy_deg: float, aspect: float,
                       z_near: float = 0.01, z_far: float = 100.0) -> np.ndarray:
    fovy = math.radians(fovy_deg)
    fy = 1.0 / math.tan(fovy * 0.5)
    fx = fy / aspect
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = fx
    M[1, 1] = fy
    M[2, 2] = z_far / (z_far - z_near)
    M[2, 3] = -z_far * z_near / (z_far - z_near)
    M[3, 2] = 1.0
    return M


def _fovy_from_proj(proj: np.ndarray) -> float | None:
    """Vertical FOV (degrees) from the ``cot(fovy/2)`` (1,1) projection entry, or None if degenerate."""
    m11 = float(proj[1, 1])
    if m11 <= 1e-6:
        return None
    return math.degrees(2.0 * math.atan(1.0 / m11))


def _copy_splats(encoder, src, dst, count: int, capacity: int) -> None:
    """Copy the first ``count`` splats of a per-splat buffer, not the allocation.

    ``src`` holds ``capacity`` elements, so the stride is derived from its size; ``dst`` capacity may differ.
    """
    stride = src.size // max(1, capacity)
    size = min(count * stride, src.size, dst.size)
    if size > 0:
        encoder.copy_buffer(dst, 0, src, 0, size)


def _linear_present_format(surface: spy.Surface) -> spy.Format:
    """Choose a non-sRGB swapchain format for presentation.

    ``output_image`` is already sRGB-encoded; an sRGB surface would encode it twice (a washed-out viewer).
    """
    def name(f) -> str:
        return getattr(f, "name", str(f)).lower()

    info = surface.info
    try:
        formats = list(info.formats)
    except Exception:
        formats = []
    preferred = info.preferred_format
    if "srgb" not in name(preferred):
        return preferred

    twins = {
        "bgra8_unorm_srgb": "bgra8_unorm",
        "rgba8_unorm_srgb": "rgba8_unorm",
        "bgrx8_unorm_srgb": "bgrx8_unorm",
    }
    want = twins.get(name(preferred))
    if want is not None:
        for f in formats:
            if name(f) == want:
                return f
    for f in formats:
        if "srgb" not in name(f) and name(f) in ("bgra8_unorm", "rgba8_unorm", "bgrx8_unorm"):
            return f
    for f in formats:
        if "srgb" not in name(f):
            return f
    return preferred


class FlyCamera:

    def __init__(self, eye=None, yaw=0.0, pitch=0.0, roll=0.0, speed=1.0):
        self.eye = np.array(eye if eye is not None else [0.0, 0.0, -2.0], dtype=np.float64)
        self.speed = speed
        self.sensitivity = 0.2
        self._set_basis_from_euler(yaw, pitch, roll)
        # Roll-free mouse-look reference: yaw rotates about it and right stays perpendicular; not world +Y.
        self._ref_up = self._up.copy()

    def _set_basis_from_euler(self, yaw: float, pitch: float, roll: float) -> None:
        """Seed the (right, up, forward) basis from Euler angles in the world-up (+Y) convention."""
        yaw_r = math.radians(yaw)
        pitch_r = math.radians(pitch)
        f = np.array([
            math.cos(pitch_r) * math.sin(yaw_r),
            -math.sin(pitch_r),
            math.cos(pitch_r) * math.cos(yaw_r),
        ], dtype=np.float64)
        world_up = np.array([0.0, 1.0, 0.0])
        r0 = np.cross(world_up, f)
        n = np.linalg.norm(r0)
        # Near-vertical forward leaves right ill-defined; fall back to +X so the basis stays finite.
        r0 = r0 / n if n > 1e-8 else np.array([1.0, 0.0, 0.0])
        u0 = np.cross(f, r0)
        a = math.radians(roll)
        self._fwd = f
        self._right = math.cos(a) * r0 + math.sin(a) * u0
        self._up = math.cos(a) * u0 - math.sin(a) * r0
        self._orthonormalize()

    def _orthonormalize(self) -> None:
        """Re-orthonormalize the basis (right x up = forward) from forward + right, removing float drift."""
        f = self._fwd / (np.linalg.norm(self._fwd) + 1e-12)
        r = self._right - f * np.dot(self._right, f)
        r = r / (np.linalg.norm(r) + 1e-12)
        u = np.cross(f, r)          # right x up = forward  ->  up = forward x right
        self._fwd, self._right, self._up = f, r, u

    def _rebuild_from_ref(self) -> None:
        """Rebuild (right, up) from forward, keeping right perpendicular to ``_ref_up`` (all roll lives there)."""
        f = self._fwd / (np.linalg.norm(self._fwd) + 1e-12)
        r = np.cross(self._ref_up, f)
        n = np.linalg.norm(r)
        if n < 1e-6:
            # forward ~parallel to ref_up: keep previous right, projected off f.
            r = self._right - f * np.dot(self._right, f)
            n = np.linalg.norm(r)
            if n < 1e-6:
                r = np.array([1.0, 0.0, 0.0]) - f * float(f[0])
                n = np.linalg.norm(r)
        r = r / (n + 1e-12)
        self._fwd, self._right, self._up = f, r, np.cross(f, r)

    @staticmethod
    def _rotate_about(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
        """Rodrigues rotation of ``v`` about unit ``axis`` by ``angle`` (rad)."""
        c, s = math.cos(angle), math.sin(angle)
        return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1.0 - c)

    @property
    def forward(self) -> np.ndarray:
        return self._fwd

    @property
    def right(self) -> np.ndarray:
        return self._right

    @property
    def up(self) -> np.ndarray:
        return self._up

    def roll_by(self, degrees: float) -> None:
        ref = self._rotate_about(self._ref_up, self._fwd, math.radians(degrees))
        self._ref_up = ref / (np.linalg.norm(ref) + 1e-12)
        self._rebuild_from_ref()

    def set_ref_up_to_current(self) -> None:
        self._ref_up = self._up / (np.linalg.norm(self._up) + 1e-12)

    def move(self, keys: set, dt: float):
        vel = self.speed * dt
        if "w" in keys:
            self.eye += self._fwd * vel
        if "s" in keys:
            self.eye -= self._fwd * vel
        if "a" in keys:
            self.eye -= self._right * vel
        if "d" in keys:
            self.eye += self._right * vel
        # Q/E translate along the current (roll-aware) up, so they track roll.
        if "q" in keys:
            self.eye += self._up * vel
        if "e" in keys:
            self.eye -= self._up * vel

    def adjust_speed(self, steps: float) -> None:
        if steps:
            self.speed *= 1.25 ** steps
            self.speed = float(max(1e-3, min(self.speed, 1e5)))

    def rotate(self, dx: float, dy: float):
        yaw = math.radians(dx * self.sensitivity)     # drag-right -> look right
        pitch = math.radians(-dy * self.sensitivity)  # drag-up -> look up
        f = self._rotate_about(self._fwd, self._ref_up, yaw)
        r = np.cross(self._ref_up, f)
        n = np.linalg.norm(r)
        r = r / n if n > 1e-6 else self._right
        f_pitched = self._rotate_about(f, r, pitch)
        # Stop pitch from crossing the ref_up pole (keeps right well-defined).
        if np.linalg.norm(np.cross(self._ref_up, f_pitched)) > math.sin(math.radians(0.5)):
            f = f_pitched
        self._fwd = f
        self._rebuild_from_ref()

    def set_from_view_matrix(self, view: np.ndarray) -> None:
        view = np.asarray(view, dtype=np.float64)
        R = view[:3, :3]
        t = view[:3, 3]
        self.eye = (-R.T @ t).astype(np.float64)
        self._right = R[0].astype(np.float64)
        self._up = R[1].astype(np.float64)
        self._fwd = R[2].astype(np.float64)
        self._orthonormalize()
        self._ref_up = self._up.copy()

    def get_camera(self, width: int, height: int, fovy: float = 45.0,
                   aspect: float | None = None) -> Camera:
        view = _build_view_matrix(
            self.eye.astype(np.float32),
            self._fwd.astype(np.float32),
            self._up.astype(np.float32),
        )
        proj = _build_proj_matrix(fovy, aspect if aspect is not None else width / height)
        return Camera(view_matrix=view, proj_matrix=proj, image_size=(width, height))


#: Key tags whose motion drops the "parked on a dataset camera" state.
MOVE_TAGS = frozenset({"w", "a", "s", "d", "q", "e", "r", "t"})
ROLL_SPEED_DEG_S = 60.0
TRAINING_SCENE_FPS = 15.0


class _CameraRig:

    def __init__(self, *, fovy: float, render_size: tuple[int, int],
                 camera_sets: Sequence[CameraSet] = (),
                 on_change: Callable[[], None] | None = None) -> None:
        self.cam = FlyCamera()
        self.fovy = float(fovy)
        self._render_size = render_size
        self.sets: tuple[CameraSet, ...] = tuple(camera_sets)
        self.set_index = 0              # the set the dropdown and C/V cycle through
        self.index = -1                 # -1 == free-fly
        self.exact: Camera | None = None  # rendered verbatim while parked
        # Parked + training re-renders only on demand, so an unannounced snap would leave the old camera up.
        self._on_change = on_change or (lambda: None)

    @property
    def cameras(self) -> Sequence[tuple[str, Camera]]:
        return self.sets[self.set_index].cameras if self.sets else ()

    @property
    def set_label(self) -> str:
        return self.sets[self.set_index].label if self.sets else ""

    def set_names(self) -> list[str]:
        return [f"{s.label}  ({len(s.cameras)})" for s in self.sets]

    def select_set(self, set_index: int) -> None:
        """List another set; the view stays where it is, in free-fly."""
        if set_index != self.set_index:
            self.set_index = set_index
            self.snap_to(-1)

    def names(self) -> list[str]:
        return ["Free-fly"] + [f"{i + 1:02d}  {name}"
                               for i, (name, _) in enumerate(self.cameras)]

    def selection(self) -> int:
        """Dropdown index: 0 == free-fly, k == camera k-1 of the listed set."""
        return self.index + 1

    def label(self) -> str:
        if self.index < 0 or not self.cameras:
            return "free-fly"
        name, _ = self.cameras[self.index]
        exact = "exact proj" if self.exact is not None else "fly proj"
        return (f"{self.set_label} cam {self.index + 1}/{len(self.cameras)}  "
                f"{name}  ({exact})")

    def set_camera_sets(self, camera_sets: Sequence[CameraSet], snap: bool = False) -> None:
        """Replace the sets, keeping the chosen one listed."""
        self.sets = tuple(camera_sets)
        self.set_index = min(self.set_index, max(len(self.sets) - 1, 0))
        self.snap_to(0 if snap and self.cameras else -1)

    def snap_to(self, index: int) -> None:
        """Park on camera ``index`` of the listed set (-1 => free-fly)."""
        if index < 0 or not self.cameras:
            self.index, self.exact = -1, None
            self._on_change()
            return
        index %= len(self.cameras)
        name, target = self.cameras[index]
        self.cam.set_from_view_matrix(target.view_matrix)  # WASD continues here
        fovy = _fovy_from_proj(target.proj_matrix)
        if fovy is not None:
            self.fovy = fovy
        self.index = index
        self.exact = (target if tuple(target.image_size) == self._render_size
                      else None)
        self._on_change()
        print(f"[Viewer] snapped to {self.set_label} cam {index + 1}/{len(self.cameras)} "
              f"({name}), exact_proj={self.exact is not None}, "
              f"fovy={self.fovy:.1f} deg")

    def step(self, delta: int) -> None:
        if not self.cameras:
            print("[Viewer] no cameras in the listed set to snap to")
            return
        self.snap_to((self.index + delta) % len(self.cameras))

    def leave(self) -> None:
        """Manual navigation drops the parked state (keeps the current pose)."""
        if self.index != -1:
            self.snap_to(-1)

    def advance(self, held: set[str], dt: float) -> None:
        """Apply one frame of held-key motion (move + roll)."""
        self.cam.move(held, dt)
        if "r" in held:
            self.cam.roll_by(+ROLL_SPEED_DEG_S * dt)
        if "t" in held:
            self.cam.roll_by(-ROLL_SPEED_DEG_S * dt)
        if held & MOVE_TAGS:
            self.leave()

    def camera(self, width: int, height: int, aspect: float) -> Camera:
        if self.exact is not None:
            return self.exact
        return self.cam.get_camera(width, height, self.fovy, aspect=aspect)

    def current_name(self) -> str:
        if self.index < 0 or not self.cameras:
            return "(free-fly)"
        return self.cameras[self.index][0]


class _ShaderWatcher:
    """Polls .slang mtimes and reloads all programs once a change has been debounced."""

    DEBOUNCE_S = 0.4
    POLL_INTERVAL = 0.5

    def __init__(self, device: spy.Device, pipeline: TrainingPipeline, runtime: GpuRuntime):
        self._device = device
        self._pipeline = pipeline
        self._runtime = runtime
        self._mtimes: dict[str, float] = {}
        self._last_poll = 0.0
        self._change_at: float | None = None
        self._scan(initial=True)

    def _scan(self, initial: bool = False) -> bool:
        changed = False
        for root, _, files in os.walk(SHADERS_DIR):
            for f in files:
                if not f.endswith(".slang"):
                    continue
                path = os.path.join(root, f)
                try:
                    mt = os.path.getmtime(path)
                except OSError:
                    continue
                if not initial and self._mtimes.get(path) != mt:
                    changed = True
                self._mtimes[path] = mt
        return changed

    def poll(self):
        """Call each frame. Triggers reload_all_programs after debounce."""
        now = time.monotonic()
        if now - self._last_poll >= self.POLL_INTERVAL:
            self._last_poll = now
            if self._scan():
                self._change_at = now

        if self._change_at is not None and now - self._change_at >= self.DEBOUNCE_S:
            self._change_at = None
            with self._runtime.queue_scope():
                self._runtime.wait_for_idle()
                self._device.reload_all_programs()
            try:
                self._pipeline.on_hot_reload()
                print("[Viewer] shaders reloaded")
            except Exception as e:
                report_exception("[Viewer] on_hot_reload failed", e)


class Viewer:
    def __init__(self, device: spy.Device, width: int = 960, height: int = 540,
                 max_splats: int = 500_000, fovy: float = 45.0,
                 hot_reload: bool = False, max_sort_pairs: int | None = None,
                 gpu_lock=None, sh_degree: int = 0,
                 mip_enabled: bool = False, mip_kernel_size: float = 0.3,
                 camera_sets: Sequence[CameraSet] = (),
                 state: Session | None = None):
        self.device = device
        self.width = width
        self.height = height
        self.max_splats = max_splats
        self.fovy = fovy
        self.hot_reload = hot_reload
        self.sh_degree = sh_degree
        self.state = state if state is not None else Session()
        self._hub = self.state.mode == MODE_HUB
        self._show_camera_controls = not self._hub
        self._scene_dirty = threading.Event()
        self._scene_dirty.set()
        self._last_scene_s = 0.0
        self._frame_budget_off = threading.Event()
        self._rig = _CameraRig(fovy=fovy, render_size=(width, height),
                               camera_sets=camera_sets,
                               on_change=self.invalidate_scene)

        import contextlib
        self.gpu_lock = gpu_lock if gpu_lock is not None else contextlib.nullcontext()
        self.runtime = GpuRuntime(device, self.gpu_lock)
        self._ui_thread_id: int | None = None
        self._started = threading.Event()
        self._exited = threading.Event()
        self._main_thread_calls: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._closed_by_user = threading.Event()

        self._start_event = threading.Event()
        self._pause_event = threading.Event()
        self._stop_training = threading.Event()
        self._suspend = threading.Event()
        self._training_active = False
        self._grab = threading.Event()

        self._render_lock = threading.RLock()
        self._explicit_sort_pairs = max_sort_pairs
        self._mip_enabled = mip_enabled
        self._mip_kernel_size = mip_kernel_size
        self._slang_session = device.slang_session
        self._pipeline = self._build_view_pipeline(
            max_splats, sh_degree, mip_enabled, mip_kernel_size, max_sort_pairs)

        self._has_data = False


    def _build_view_pipeline(self, max_splats: int, sh_degree: int,
                             mip_enabled: bool, mip_kernel_size: float,
                             max_sort_pairs: int | None) -> TrainingPipeline:
        """The viewer's own render-only pipeline, sized for the snapshots it will receive."""
        pipeline = TrainingPipeline(
            device=self.device, width=self.width, height=self.height,
            max_splats=max_splats,
            max_sort_pairs=max_sort_pairs,
            gpu_lock=self.gpu_lock,
            viewer_only=True,
            appearance_config=AppearanceConfig(sh_degree=sh_degree),
            raster_config=RasterConfig(mip_enabled=mip_enabled,
                                       kernel_size=mip_kernel_size),
            slang_session=self._slang_session,
        )
        if sh_degree > 0:
            pipeline.set_active_sh_degree(sh_degree)
        return pipeline

    def reconfigure_model(self, *, sh_degree: int, mip_enabled: bool,
                          mip_kernel_size: float, slang_session: spy.SlangSession) -> bool:
        if (sh_degree == self.sh_degree
                and mip_enabled == self._mip_enabled
                and mip_kernel_size == self._mip_kernel_size
                and slang_session is self._slang_session):
            return False
        with self._render_lock:      # no frame in flight while we swap pipelines
            with self.runtime.queue_scope():
                self.runtime.wait_for_idle()   # old buffers may still be in use
            self.sh_degree = sh_degree
            self._mip_enabled = mip_enabled
            self._mip_kernel_size = mip_kernel_size
            self._slang_session = slang_session
            self._pipeline = self._build_view_pipeline(
                self.max_splats, sh_degree, mip_enabled, mip_kernel_size,
                self._explicit_sort_pairs)
            # The new buffers are empty: don't rasterize until a snapshot lands.
            self._has_data = False
        print(f"[Viewer] model reconfigured: max_splats={self.max_splats:,} "
              f"sh_degree={sh_degree} mip={mip_enabled}")
        return True

    def _grow(self, count: int, limit: int) -> None:
        """Re-allocate the viewer's buffers for ``count`` splats, doubling up to ``limit``.

        The caller holds the render lock and re-uploads before releasing it.
        """
        with self.runtime.queue_scope():
            self.runtime.wait_for_idle()   # in-flight frames/copies still use the old buffers
        self.max_splats = max(count, min(2 * self.max_splats, limit))
        self._pipeline.resize(self.max_splats, self._explicit_sort_pairs)
        print(f"[Viewer] capacity grown to {self.max_splats:,} splats")

    def live_splat_count(self, train_pipeline: TrainingPipeline) -> int:
        with self.runtime.queue_scope():
            return train_pipeline.primitives.count()

    def update_gaussians(self, train_pipeline: TrainingPipeline):
        count = self.live_splat_count(train_pipeline)
        # Densify keeps the count on the GPU, so this snapshot read is the panel's live source.
        self.state.publish_training(gaussians=count)
        with self._render_lock:      # serialize GPU upload + cached_count with render_frame()
            if count > self.max_splats:
                self._grow(count, limit=train_pipeline.max_splats)
            self._copy_snapshot(train_pipeline, count)

    def _copy_snapshot(self, train_pipeline: TrainingPipeline, count: int) -> None:
        src, dst = train_pipeline.buffers, self._pipeline.buffers
        batch = self.runtime.command_batch("viewer_update")
        encoder = batch.encoder
        _copy_splats(encoder, src.primitive.values.storage, dst.primitive.values.storage,
                     count, src.max_splats)
        if src.sh_enabled and dst.sh_enabled:
            _copy_splats(encoder, src.sh.features_rest.storage,
                         dst.sh.features_rest.storage, count, src.max_splats)
        if self._pipeline.raster_config.mip_enabled:
            _copy_splats(encoder, src.primitive.filter_3d.storage,
                         dst.primitive.filter_3d.storage, count, src.max_splats)
        encoder.upload_buffer_data(dst.primitive.active_count.storage, 0, np.array([count], dtype=np.uint32))
        self.runtime.submit(batch)
        dst.primitive.cached_count = int(count)
        self._pipeline.rasterizer.natural_vignetting = train_pipeline.rasterizer.natural_vignetting
        self._pipeline.set_active_sh_degree(train_pipeline.active_sh_degree)
        self._pipeline.set_extinction_hardness(train_pipeline.extinction_hardness)
        self._has_data = count > 0
        self.invalidate_scene()

    def preview_points(self, initial_points: dict) -> int:
        means = np.asarray(initial_points["means"], dtype=np.float32)
        count = means.shape[0]
        with self._render_lock:      # no frame reading the buffers mid-upload
            if count > self.max_splats:
                self._grow(count, limit=count)
            with self.runtime.queue_scope():
                self._pipeline.set_gaussians(
                    means=means,
                    log_scales=initial_points["log_scales"],
                    rot_quats=initial_points["rot_quats"],
                    opacities=initial_points["opacities"],
                    color_logits=initial_points["color_logits"],
                )
                if count and self._pipeline.raster_config.mip_enabled:
                    self._pipeline.buffers.primitive.filter_3d.storage.copy_from_numpy(
                        np.zeros(count, dtype=np.float32))
            self._pipeline.rasterizer.natural_vignetting = False
            self._pipeline.set_active_sh_degree(0)
            self._pipeline.set_extinction_hardness(1.0)
            self._has_data = count > 0
        self.invalidate_scene()
        return count

    def clear_view(self) -> None:
        with self._render_lock:      # no frame mid-read of the buffers
            self._has_data = False
        self._rig.set_camera_sets(())
        self._training_active = False
        print("[Viewer] view cleared")

    def start_requested(self) -> bool:
        """True once the user pressed 'Start Training' in the viewer panel."""
        return self._start_event.is_set()

    def wait_for_start(self, poll: float = 0.05) -> None:
        """Block until 'Start Training' is pressed or the viewer closes, so a closed viewer can't hang training."""
        while not self._start_event.is_set():
            if self._stop_event.is_set() or self._exited.is_set():
                break
            time.sleep(poll)

    def invalidate_scene(self) -> None:
        self._scene_dirty.set()

    def suspend_frame_budget(self) -> None:
        self._frame_budget_off.set()
        self.invalidate_scene()

    def resume_frame_budget(self) -> None:
        self._frame_budget_off.clear()

    def _training_frame_interval(self) -> float:
        if (not self._training_active or self._frame_budget_off.is_set()
                or self._pause_event.is_set()):
            return 0.0
        return 1.0 / TRAINING_SCENE_FPS

    def _scene_due(self, now: float) -> bool:
        interval = self._training_frame_interval()
        if not interval or self._scene_dirty.is_set():
            return True
        if self._rig.index >= 0:            # parked: only new data moves the image
            return False
        return (now - self._last_scene_s) >= interval

    def set_training_active(self, active: bool = True) -> None:
        self._training_active = active
        if active:
            self.state.training_running()

    def attach_scene(self, camera_sets: Sequence[CameraSet], snap: bool = False) -> None:
        self._rig.set_camera_sets(camera_sets, snap=snap)

    def suspend_rendering(self) -> None:
        self._suspend.set()

    def resume_rendering(self) -> None:
        self._suspend.clear()

    def is_paused(self) -> bool:
        return self._pause_event.is_set()

    def pause_training(self) -> None:
        # Optimizer activity only throttles rendering; camera sweeps and post-training analysis stay pausable.
        if self.state.snapshot().can_pause and not self._pause_event.is_set():
            self._pause_event.set()

    def resume_training(self) -> None:
        if self._pause_event.is_set():
            self._pause_event.clear()

    def request_stop_training(self) -> None:
        if not self.state.training_stopping():
            return
        # Publish STOPPING before waking a paused worker, so this request can't overwrite its completion.
        self._stop_training.set()
        self._pause_event.clear()

    def stop_requested(self) -> bool:
        return self._stop_training.is_set() or self._closed_by_user.is_set()

    def clear_stop_request(self) -> None:
        self._stop_training.clear()
        self._pause_event.clear()

    def wait_while_paused(self, poll: float = 0.05) -> None:
        if not self._pause_event.is_set():
            return
        self.state.training_paused()
        while self._pause_event.is_set():
            if self._stop_event.is_set() or self._exited.is_set():
                break
            time.sleep(poll)
        if not self.stop_requested():
            self.state.training_resumed()

    def mark_starting(self) -> None:
        """Record handoff to the main thread so ``is_alive()`` is true before ``run()`` enters."""
        self._exited.clear()
        self._started.set()

    def is_alive(self) -> bool:
        return self._started.is_set() and not self._exited.is_set()

    def has_exited(self) -> bool:
        return self._exited.is_set()

    def wait_until_closed(self) -> None:
        if self._started.is_set():
            self._exited.wait()

    def run(self) -> None:
        """Run the viewer window and event loop on the calling (main) thread."""
        self._started.set()
        self._ui_thread_id = threading.get_ident()
        try:
            if not self._stop_event.is_set():
                self._run_inner()
        except KeyboardInterrupt:
            self._closed_by_user.set()
            self._stop_event.set()
            raise
        except Exception as e:
            self._report_error("crashed", e)
        finally:
            self._ui_thread_id = None
            self._stop_event.set()
            self._pause_event.clear()
            self._exited.set()

    def call_on_main_thread(self, callback):
        """Run an OS UI operation on the viewer's main thread and return its result."""
        if not self._started.is_set() or threading.get_ident() == self._ui_thread_id:
            return callback()
        if self._exited.is_set():
            raise RuntimeError("viewer closed before the UI operation could run")
        completed = threading.Event()
        result: dict[str, object] = {}
        self._main_thread_calls.put((callback, completed, result))
        while not completed.wait(0.05):
            if self._exited.is_set():
                raise RuntimeError("viewer closed before the UI operation could run")
        if "error" in result:
            raise result["error"]
        return result.get("value")

    def _run_main_thread_calls(self) -> None:
        while True:
            try:
                callback, completed, result = self._main_thread_calls.get_nowait()
            except queue.Empty:
                return
            try:
                result["value"] = callback()
            except BaseException as error:
                result["error"] = error
                if isinstance(error, KeyboardInterrupt):
                    raise
            finally:
                completed.set()

    def stop(self) -> None:
        self._closed_by_user.set()
        self._stop_event.set()

    def _report_error(self, context: str, error: Exception) -> None:
        signature = (context, type(error), str(error))
        if signature == getattr(self, "_last_error", None):
            return
        self._last_error = signature
        self.state.notify(f"Viewer {context} - see diagnostics", ERROR)
        report_exception(f"[Viewer] {context}", error, self.state)

    def _set_up_ref(self) -> None:
        self._rig.cam.set_ref_up_to_current()
        print("[Viewer] up reference set to current view")

    def _save_frame(self) -> None:
        """Write the last rendered frame to a PNG for offline diffing against the eval output."""
        try:
            from PIL import Image
            img = self._pipeline.read_output_image()[..., :3]
            png = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
            name = self._rig.current_name()
            label = (f"{self._rig.set_label}cam_" + name.replace("/", "_").replace("\\", "_")
                     if self._rig.index >= 0 else "freefly")
            path = os.path.abspath(f"viewer_grab_{label}.png")
            Image.fromarray(png, mode="RGB").save(path)
            print(f"[Viewer] saved frame for {name} -> {path}")
        except Exception as e:  # noqa: BLE001 - a failed grab must not stop the loop
            self._report_error("frame grab failed", e)

    def _build_input(self, mouse: dict) -> InputMap:
        keys = InputMap().hold_many([
            (spy.KeyCode.w, "w"), (spy.KeyCode.a, "a"), (spy.KeyCode.s, "s"),
            (spy.KeyCode.d, "d"), (spy.KeyCode.q, "q"), (spy.KeyCode.e, "e"),
            (spy.KeyCode.r, "r"), (spy.KeyCode.t, "t"),  # R/T roll
        ])
        keys.press(spy.KeyCode.c, lambda: self._rig.step(+1))
        keys.press(spy.KeyCode.v, lambda: self._rig.step(-1))
        keys.press(spy.KeyCode.p, self._grab.set)
        keys.press(spy.KeyCode.y, self._set_up_ref)
        keys.press(spy.KeyCode.escape, lambda: mouse.update(captured=False))
        return (keys
                .document("WASD / QE", "move")
                .document("drag", "look around")
                .document("wheel", "move speed")
                .document("R / T", "roll view")
                .document("Y", "level horizon to view")
                .document("C / V", "next / prev camera")
                .document("P", "save frame to PNG")
                .document("F1", "show / hide panel"))

    def _panel_actions(self) -> Actions:
        return Actions(
            browse=lambda: self.state.request("browse"),
            install_tools=lambda: self.state.request("install_tools"),
            reconstruct=lambda: self.state.request("reconstruct"),
            select_matcher=self.state.set_matcher_index,
            set_recon_cuda=self.state.set_recon_cuda,
            new_dataset=lambda: self.state.request("new_dataset"),
            configure_next_run=lambda: self.state.request("configure_next_run"),
            back_to_result=lambda: self.state.request("back_to_result"),
            start_training=lambda: self.state.request("train"),
            restart_training=lambda: self.state.request("restart"),
            pause=self.pause_training,
            resume=self.resume_training,
            stop=self.request_stop_training,
            export=lambda: self.state.request("export"),
            reveal_export=lambda: self.state.request("reveal"),
            confirm=lambda: self.state.request("confirm"),
            confirm_alt=lambda: self.state.request("confirm_alt"),
            cancel=lambda: self.state.request("cancel"),
            toggle_log=self.state.toggle_log,
            clear_log=self.state.clear_log,
        )

    def _build_panel(self, view, keys: InputMap) -> Panel:
        panel = Panel(self.device, title="PERF-GS", anchor="left",
                      thickness=430.0, min_thickness=380.0)
        actions = self._panel_actions()
        header(panel, view)

        if self._hub:
            dataset_section(panel, view, actions)
            result_section(panel, view, actions)
        training_section(panel, view, actions, self.state.setup if self._hub else None)
        if self._hub:
            configuration_section(
                panel, view, self.state.setup,
                show_all=lambda: view.show_all_config,
                toggle_all=self.state.toggle_config_detail)
        loss_section(panel, view)
        def has_scene() -> bool:
            return not self._hub or view.trainable or view.has_result

        if self._hub:
            panel.checkbox("Camera controls and shortcuts", False,
                           lambda enabled: setattr(self, "_show_camera_controls", enabled), visible=has_scene)

        def camera_controls_open() -> bool:
            return has_scene() and self._show_camera_controls

        viewer_section(panel, CameraControls(
            sets=self._rig.set_names,
            set_selection=lambda: self._rig.set_index,
            select_set=lambda index: self._rig.select_set(int(index)),
            names=self._rig.names,
            selection=self._rig.selection,
            select=lambda index: self._rig.snap_to(int(index) - 1),
            label=self._rig.label,
            speed=lambda: self._rig.cam.speed,
            fovy=lambda: self._rig.fovy,
        ), visible=camera_controls_open)
        controls_section(panel, keys.describe(), visible=camera_controls_open)
        diagnostics_section(panel, view, actions, show=lambda: view.show_log)
        return panel

    def _run_inner(self):
        window = spy.Window(width=self.width, height=self.height, title="PERF-GS")

        surface = self.device.create_surface(window)
        present_format = _linear_present_format(surface)
        print(f"[Viewer] surface preferred={surface.info.preferred_format} "
              f"-> presenting as {present_format}")
        surface.configure(width=self.width, height=self.height, format=present_format)
        surf_w, surf_h = self.width, self.height
        render_lock = self._render_lock   # main-loop vs on_resize vs reconfigure

        view = self.state.view()
        mouse = {"captured": False, "x": 0.0, "y": 0.0}
        keys = self._build_input(mouse)
        ui = self._build_panel(view, keys)
        keys.press(spy.KeyCode.f1, ui.toggle)
        print(f"[Viewer] panel: {ui.dock.thickness_px:.0f}px wide "
              f"(display scale {ui.dock.scale:.2f}x)")

        def on_key(event):
            if ui.handle_keyboard(event):
                return
            keys.handle(event)

        def on_mouse(event):
            if ui.handle_mouse(event):
                mouse["captured"] = False  # don't drive the camera from panel clicks
                return
            if event.is_scroll():
                self._rig.cam.adjust_speed(float(event.scroll.y))
                return
            if event.is_button_down():
                mouse.update(captured=True, x=event.pos[0], y=event.pos[1])
            elif event.is_button_up():
                mouse["captured"] = False
            elif event.is_move() and mouse["captured"]:
                dx = event.pos[0] - mouse["x"]
                dy = event.pos[1] - mouse["y"]
                mouse["x"], mouse["y"] = event.pos[0], event.pos[1]
                self._rig.cam.rotate(dx, dy)
                self._rig.leave()

        def on_drop(paths) -> None:
            if paths:
                self.state.drop(str(paths[0]))

        def _reconfigure(w, h):
            nonlocal surf_w, surf_h
            if w <= 0 or h <= 0 or (w, h) == (surf_w, surf_h):
                return
            with self.runtime.queue_scope():
                self.runtime.wait_for_idle()
                surface.configure(width=w, height=h, format=present_format)
            surf_w, surf_h = w, h
            self.invalidate_scene()

        def render_frame():
            with render_lock:
                if self._suspend.is_set():   # hub setup owns the GPU queue
                    return
                w, h = surf_w, surf_h
                ui.layout(w, h)
                ui.refresh()
                surface_tex = surface.acquire_next_image()
                if surface_tex is None:      # can happen transiently on resize
                    return
                batch = self.runtime.command_batch("viewer_frame")
                if self._has_data:
                    now = time.perf_counter()
                    if self._scene_due(now):
                        camera = self._rig.camera(self.width, self.height, w / h)
                        with self._pipeline.commands.attached(batch):
                            self._pipeline.rasterize(camera)
                        self._scene_dirty.clear()
                        self._last_scene_s = now
                    # Fetch batch.encoder AFTER rasterize: Metal indirect emulation may reopen it.
                    enc = batch.encoder
                    enc.blit(surface_tex, self._pipeline.buffers.raster.output_image)
                else:
                    enc = batch.encoder
                    enc.clear_texture_float(surface_tex, clear_value=BACKDROP)
                ui.begin(w, h, window)
                ui.end(surface_tex, enc)
                submit = self.runtime.submit(batch, after_submit=surface.present)
                self.runtime.wait_for_submit(submit)
                if self._grab.is_set():
                    self._grab.clear()
                    self._save_frame()

        def on_resize(w, h):
            if self._suspend.is_set():
                return
            try:
                _reconfigure(w, h)
                render_frame()
            except Exception as e:
                self._report_error("resize failed", e)

        window.on_keyboard_event = on_key
        window.on_mouse_event = on_mouse
        window.on_resize = on_resize
        if self._hub:
            window.on_drop_files = on_drop

        watcher = _ShaderWatcher(
            self.device, self._pipeline, self.runtime) if self.hot_reload else None
        last_time = time.perf_counter()

        try:
            while not window.should_close() and not self._stop_event.is_set():
                self._run_main_thread_calls()
                window.process_events()
                if self._suspend.is_set():  # hub setup owns the GPU queue right now
                    time.sleep(0.02)
                    continue
                if watcher:
                    watcher.poll()

                _reconfigure(window.width, window.height)

                now = time.perf_counter()
                dt = min(now - last_time, 0.1)
                last_time = now
                self._rig.advance(keys.held, dt)
                view.sample()   # the frame's single, consistent state snapshot

                try:
                    render_frame()
                    self._last_error = None
                except Exception as e:
                    self._report_error("frame failed; viewer stopped", e)
                    # A frame failure may leave the acquired surface image unpresented;
                    # close the viewer without aborting active training.
                    self._stop_event.set()
                    break

                # Even a cached scene submits a UI pass and presents on the training GPU.
                # Wait outside both locks, and count time already spent drawing this frame.
                remaining = self._training_frame_interval() - (time.perf_counter() - now)
                if remaining > 0:
                    self._stop_event.wait(remaining)

            if window.should_close():
                self._closed_by_user.set()
        finally:
            try:
                window.close()
            except Exception as error:
                self._report_error("closing the window failed", error)
