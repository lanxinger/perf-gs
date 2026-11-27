# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Lifecycle of the optional in-process live viewer (a render thread sharing the device under ``gpu_lock``).

Methods are no-ops under ``--no-viewer``, but a :class:`~trainer.frontend.viewer.session.Session` always exists.
"""
from __future__ import annotations

import contextlib
import threading
import time
from typing import TYPE_CHECKING

from trainer.frontend.run_control import RunStopped
from trainer.frontend.viewer.session import MODE_HUB, MODE_TRAIN, OK, Session

if TYPE_CHECKING:
    import slangpy as spy

    from trainer.backend.render.pipeline import TrainingPipeline
    from trainer.frontend.scene import CameraSet
    from trainer.frontend.settings import TrainingSetup
    from trainer.frontend.viewer.viewer import Viewer


class ViewerSession:
    """Owns the optional live viewer, its full lifecycle, and the session state."""

    def __init__(self, viewer: Viewer | None,
                 state: Session | None = None) -> None:
        self._viewer = viewer
        # One state object shared with the viewer's panel.
        if state is None:
            state = viewer.state if viewer is not None else Session(MODE_TRAIN)
        self.state = state

    @property
    def enabled(self) -> bool:
        """Whether a live viewer is attached (``False`` under ``--no-viewer``)."""
        return self._viewer is not None

    @classmethod
    def create(
        cls,
        *,
        device: spy.Device,
        width: int,
        height: int,
        fovy: float,
        gpu_lock,
        pipeline: TrainingPipeline,
        camera_sets: tuple[CameraSet, ...] = (),
    ) -> ViewerSession:
        """Build the viewer and push the initialized gaussians without starting the render thread.

        Setup uploads still run without ``gpu_lock`` (a queue race), so call :meth:`start` once setup is done.
        """
        from trainer.frontend.viewer.viewer import Viewer

        state = Session(MODE_TRAIN)
        viewer = Viewer(
            device=device, width=width, height=height,
            fovy=fovy, gpu_lock=gpu_lock,
            sh_degree=pipeline.appearance_config.sh_degree,
            mip_enabled=pipeline.raster_config.mip_enabled,
            mip_kernel_size=pipeline.raster_config.kernel_size,
            camera_sets=camera_sets,
            state=state,
        )
        viewer.update_gaussians(pipeline)
        print("[viewer] created (render thread starts after setup completes).")
        return cls(viewer, state)

    @classmethod
    def create_hub(
        cls,
        *,
        device: spy.Device,
        width: int,
        height: int,
        fovy: float,
        gpu_lock,
        sh_degree: int,
        mip_enabled: bool,
        mip_kernel_size: float,
        setup: TrainingSetup | None = None,
    ) -> ViewerSession:
        """Build a scene-less hub viewer; the orchestrator later feeds it gaussians and cameras.

        ``setup`` is the shared store the panel edits until Train (see :mod:`trainer.frontend.settings`).
        """
        from trainer.frontend.viewer.viewer import Viewer

        state = Session(MODE_HUB, setup=setup)
        viewer = Viewer(
            device=device, width=width, height=height,
            fovy=fovy, gpu_lock=gpu_lock,
            sh_degree=sh_degree, mip_enabled=mip_enabled,
            mip_kernel_size=mip_kernel_size,
            state=state,
        )
        print("[viewer] hub created (no dataset yet).")
        return cls(viewer, state)

    def attach_scene(self, camera_sets: tuple[CameraSet, ...], snap: bool = False) -> None:
        """Populate the camera pickers once a scene is available; ``snap`` also parks on the first camera."""
        if self._viewer is not None:
            self._viewer.attach_scene(camera_sets, snap=snap)

    def preview_points(self, initial_points: dict) -> int:
        """Display a COLMAP point cloud before training; returns the number of points shown (0 with no viewer)."""
        if self._viewer is None:
            return 0
        return self._viewer.preview_points(initial_points)

    def reconfigure_model(self, *, sh_degree: int,
                          mip_enabled: bool, mip_kernel_size: float,
                          slang_session) -> bool:
        """Re-shape the viewer's pipeline after the hub's setup changed the model (no-op if unchanged/no viewer)."""
        if self._viewer is None:
            return False
        return self._viewer.reconfigure_model(
            sh_degree=sh_degree,
            mip_enabled=mip_enabled, mip_kernel_size=mip_kernel_size,
            slang_session=slang_session)

    def suspend_rendering(self) -> None:
        """Park the render thread (no GPU work, no frames) until :meth:`resume_rendering`.

        Uploads/pipeline swaps must not race it on the shared queue, and it holds the GIL for most
        of every frame, starving heavy pure-Python work (e.g. COLMAP parsing) on other threads.
        """
        if self._viewer is not None:
            self._viewer.suspend_rendering()

    def resume_rendering(self) -> None:
        if self._viewer is not None:
            self._viewer.resume_rendering()

    def start(self) -> None:
        """Start the background render thread (call once setup submits are done)."""
        if self._viewer is None:
            return
        self._viewer.start()
        print("[viewer] render thread started.", flush=True)

    def run_blocking(self) -> None:
        """Run the viewer loop on the current thread (used on macOS main thread)."""
        if self._viewer is None:
            return
        print("[viewer] running render loop on main thread.", flush=True)
        self._viewer.run_blocking()

    def call_on_ui_thread(self, fn):
        if self._viewer is None:
            return fn()
        return self._viewer.call_on_ui_thread(fn)

    def is_alive(self) -> bool:
        """Whether the viewer loop has been started and is still running."""
        if self._viewer is None:
            return False
        return self._viewer.is_alive()

    def wait_for_start(self) -> None:
        """Block until the viewer's 'Start Training' button, Enter in this terminal, or the window closing."""
        if self._viewer is None:
            return
        print("[wait-for-viewer] showing initialized gaussians. Waiting for the "
              "viewer's 'Start Training' button (or press Enter here)...")
        enter = threading.Event()

        def _wait_enter() -> None:
            try:
                input()
            except EOFError:
                pass
            enter.set()

        threading.Thread(target=_wait_enter, daemon=True).start()
        while not self._viewer.start_requested() and not enter.is_set():
            if not self.is_alive():
                print("[wait-for-viewer] viewer window closed; starting training.")
                break
            time.sleep(0.05)
        print("[wait-for-viewer] starting training.")

    def set_training_active(self, active: bool = True) -> None:
        """Flag optimizer stepping so the viewport can ration its GPU work.

        Pause/Stop follow the session phase, not this flag; with no viewer it still publishes RUNNING.
        """
        if self._viewer is not None:
            self._viewer.set_training_active(active)
        elif active:
            self.state.training_running()

    @contextlib.contextmanager
    def unthrottled(self):
        """Render at full rate for the duration: the trainer is not stepping (eval sweep, render-all)."""
        if self._viewer is None:
            yield
            return
        self._viewer.suspend_frame_budget()
        try:
            yield
        finally:
            self._viewer.resume_frame_budget()

    def is_paused(self) -> bool:
        """Whether the viewer's Pause button currently has training suspended."""
        return self._viewer is not None and self._viewer.is_paused()

    def stop_requested(self) -> bool:
        """Whether the user pressed Stop or closed the viewer."""
        return self._viewer is not None and self._viewer.stop_requested()

    def clear_stop_request(self) -> None:
        """Before a new run: a stop belongs to the run that was stopped."""
        if self._viewer is not None:
            self._viewer.clear_stop_request()

    def wait_while_paused(self) -> None:
        """Block run work while the viewer's Pause button is engaged."""
        if self._viewer is not None:
            self._viewer.wait_while_paused()

    def checkpoint(self) -> None:
        """Pause or stop at a safe work boundary, also after waking from Pause."""
        if self.stop_requested():
            raise RunStopped()
        self.wait_while_paused()
        if self.stop_requested():
            raise RunStopped()

    def push_snapshot(self, pipeline: TrainingPipeline) -> None:
        """Push a gaussian snapshot: a GPU->GPU copy of the active splat range plus a 4-byte count readback."""
        if self._viewer is not None:
            self._viewer.update_gaussians(pipeline)

    def clear_view(self) -> None:
        """Blank the displayed gaussians and the camera list (the hub's Clear)."""
        if self._viewer is not None:
            self._viewer.clear_view()

    def finish_training(self, pipeline: TrainingPipeline) -> None:
        """Publish the end of the training loop and push its final gaussians, without blocking.

        The phase is left alone: metrics and export still run before the orchestrator publishes ``PHASE_DONE``.
        """
        if self._viewer is not None:
            self._viewer.set_training_active(False)
            try:
                self._viewer.update_gaussians(pipeline)
            except Exception:
                pass

    def hold_after_training(self, pipeline: TrainingPipeline) -> None:
        """Push the final state and keep the window live until the user closes it (Ctrl-C stops the viewer)."""
        self.finish_training(pipeline)
        self.state.notify("Training complete", OK)
        self.state.log("training complete — close the viewer window to exit")
        if self._viewer is None:
            return
        print("[viewer] training done — viewer still live; close its window to exit.")
        try:
            thread = self._viewer._thread
            if thread is not None:
                thread.join()
        except KeyboardInterrupt:
            self._viewer.stop()
