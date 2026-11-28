# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Interactive session lifecycle around a reusable training application."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from trainer.backend.render import stbn_table
from trainer.data.colmap import ColmapDataset, _find_sparse_dir, load_colmap
from trainer.frontend.profiling.profiler import apply_profile_session_overrides
from trainer.frontend.reporting.error_log import report_exception
from trainer.frontend.scene import holdout_split, split_camera_sets
from trainer.frontend.viewer.session import (
    Confirm, DatasetInfo, ERROR, KIND_COLMAP, KIND_IMAGES, OK,
    Session, ToolsState, WARN, format_duration,
)
from trainer.frontend.viewer.viewer_control import ViewerSession

if TYPE_CHECKING:
    from trainer.frontend.app import TrainingApplication


def _progress_relay(state: Session, parser):
    """Relay subprocess lines to the session log and, via ``parser``, to the progress bar."""
    def relay(line: str) -> None:
        state.log(line)
        parsed = parser.feed(line)
        if parsed is not None:
            state.set_progress(parsed.label, parsed.fraction)
    return relay

class TrainingHub:
    def __init__(self, app: TrainingApplication) -> None:
        self.app = app
        self._pending_drop: str | None = None
        self._scene_cache: tuple[tuple, ColmapDataset] | None = None
        # NVIDIA's STBN masks carry their own license, so they download only after the user accepts it.
        self._stbn_accepted = False

    def _colmap_key(self, scene_dir, dataset):
        args = self.app.args
        path = Path(scene_dir).resolve()
        sparse, fmt = _find_sparse_dir(path)
        sources = {path / "images", path / f"images_{dataset.downsample}"}
        if args.downsample is None:
            sources.update((path / "images_2", path / "poses_bounds.npy"))
        photos = dataset.image_dir
        if photos.is_dir() and photos.name != "images" and (args.width is None or args.height is None):
            sample = min((p for p in photos.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")), default=None)
            if sample is not None:
                sources.add(sample)
        if sparse is not None:
            suffix = "txt" if fmt == "text" else "bin"
            sources.update(sparse / f"{name}.{suffix}" for name in ("cameras", "images", "points3D"))
        stamps = []
        for source in sorted(sources):
            if source.exists():
                stat = source.stat()
                stamps.append((source, stat.st_size, stat.st_mtime_ns))
        return path, args.width, args.height, args.downsample, args.init_opacity, tuple(stamps)

    def _cached_colmap(self, scene_dir: str) -> ColmapDataset | None:
        if self._scene_cache is not None:
            try:
                if self._scene_cache[0] == self._colmap_key(scene_dir, self._scene_cache[1]):
                    return self._scene_cache[1]
            except OSError:
                pass  # A moved or concurrently reconstructed dataset must go through the loader.
        return None

    def run(self) -> None:
        """Run the interactive hub until the window closes; a session outlives its runs.
        The viewer and this loop communicate only through ``session.state``: the panel posts
        one-shot commands, this loop consumes them and publishes phase/progress/log back."""
        from trainer.frontend.reconstruction import ReconTools
        from trainer.frontend.settings import TrainingSetup

        self.app.args.no_viewer = False  # the hub is the viewer
        self.app.profiling_enabled = apply_profile_session_overrides(self.app.args)
        self.app._create_device()

        # Seeded from the CLI args; the panel edits it and it is applied back on Train.
        setup = TrainingSetup.from_args(self.app.args)
        self.app.viewer = ViewerSession.create_hub(
            device=self.app.device,
            width=self.app.args.width or 960,
            height=self.app.args.height or 540,
            fovy=getattr(self.app.args, "fovy", 45.0),
            gpu_lock=self.app.gpu_lock,
            sh_degree=self.app.args.sh_degree,
            mip_enabled=self.app.args.mip_splatting,
            mip_kernel_size=(self.app.args.mip_kernel_size if self.app.args.mip_splatting else 0.3),
            setup=setup,
        )
        if self.app.error_log is not None:
            self.app.error_log.attach(self.app.viewer.state)

        state = self.app._viewer.state
        state.set_recon_cuda(self.app.args.recon_cuda)
        tools = ReconTools(colmap_bin=self.app.args.colmap_bin,
                           glomap_bin=self.app.args.glomap_bin,
                           recon_cuda=self.app.args.recon_cuda)
        state.tools(ToolsState.READY if tools.is_ready() else ToolsState.MISSING)

        def serve() -> None:
            if self.app.args.scene:
                self._hub_dropped(str(self.app.args.scene))
            else:
                state.notify("Open or drop a dataset folder", OK)
            print("[hub] viewer ready.")

            while True:
                scene_dir = self._hub_choose_scene(tools)
                if scene_dir is None:
                    print("[hub] window closed; exiting.")
                    return
                self._hub_train_scene(scene_dir, setup)

        self.app.run_with_viewer(serve)

    def _hub_choose_scene(self, tools) -> "str | None":
        """Service panel requests until a scene is ready; ``None`` once the window closes.
        Every branch publishes a state transition so the UI reacts on the same frame."""
        state = self.app._viewer.state
        while self.app._viewer.is_alive():
            dropped = state.take_dropped()
            if dropped:
                self._hub_dropped(dropped)

            command = state.take_command()
            if command in ("confirm", "confirm_alt"):
                command = state.take_confirm(accept=True,
                                             alt=command == "confirm_alt")
                if command == "export_then_load":
                    if self._hub_export_ply():
                        command = "load_dropped"
                    else:
                        self._pending_drop = None
                        command = None
                elif command in ("download_stbn_then_train", "download_stbn_then_restart"):
                    self._stbn_accepted = True
                    command = command.removeprefix("download_stbn_then_")
            elif command == "cancel":
                state.take_confirm(accept=False)
                self._pending_drop = None
                state.notify("Cancelled", OK)
                command = None

            if command == "install_tools":
                self._hub_setup_tools(tools)
            elif command == "reconstruct":
                self._hub_reconstruct(tools)
            elif command == "export":
                self._hub_export_ply()
            elif command == "reveal":
                self._hub_reveal_export()
            elif command == "configure_next_run" and state.snapshot().can_restart_training:
                scene = self._cached_colmap(state.scene_dir)
                if scene is None:
                    self._hub_preview_scene(state.scene_dir)
                    scene = self._cached_colmap(state.scene_dir)
                else:
                    self.app.viewer.preview_points(scene.initial_points)
                state.show_next_run_setup(scene is not None)
            elif command == "back_to_result" and state.snapshot().next_run_setup:
                self.app.viewer.push_snapshot(self.app._pipeline)
                state.show_next_run_setup(False)
            elif command in ("browse", "new_dataset"):
                from slangpy import platform
                call_ui = self.app._viewer.call_on_main_thread
                try:
                    folder = call_ui(platform.choose_folder_dialog)
                except (AttributeError, OSError, RuntimeError):
                    state.notify("Folder picker unavailable. Drop a dataset folder onto the window.", WARN)
                else:
                    if folder:
                        self._hub_dropped(str(folder))
            elif command == "load_dropped":
                self._hub_load_pending_drop()
            elif command in ("train", "restart"):
                scene = self._hub_request_training(restart=command == "restart")
                if scene is not None:
                    return scene
            time.sleep(0.05)
        return None

    def _hub_dropped(self, path: str) -> None:
        """Load a dropped folder, asking first if it would replace an existing result."""
        state = self.app._viewer.state
        if state.snapshot().has_result:
            self._pending_drop = path
            if not state.snapshot().unexported:
                self._hub_load_pending_drop()
                return
            state.ask(Confirm(
                key="dataset", title="Save the current model?",
                body=f"Opening {Path(path).name} will replace the model in memory.",
                command="export_then_load", confirm_label="Export and open",
                alt_label="Open without saving", alt_command="load_dropped"))
            return
        self._hub_load_dataset(path)

    def _hub_load_pending_drop(self) -> None:
        """The confirmed half of :meth:`_hub_dropped`."""
        path, self._pending_drop = self._pending_drop, None
        if path is None:
            return
        self._hub_clear_session(notify=False)
        self._hub_load_dataset(path)

    def _hub_load_dataset(self, path: str) -> None:
        """INSPECTING -> detected -> (LOADING) -> READY / ERROR."""
        from trainer.frontend.reconstruction import classify_dataset

        self._scene_cache = None
        state = self.app._viewer.state
        state.dataset_inspecting(path)
        state.log(f"dropped {path}")
        kind = classify_dataset(path)
        state.dataset_detected(kind, scene_dir=path if kind == KIND_COLMAP else "")
        state.log(f"detected: {kind}")
        if kind == KIND_COLMAP:
            self._hub_preview_scene(path)
            return
        if kind == KIND_IMAGES:
            state.dataset_ready(DatasetInfo(path=path, kind=kind))
            state.notify("Image folder loaded - reconstruct it to train", OK)
            return
        state.dataset_failed("No images and no sparse model here. A dataset is a "
                             "folder of images, or one with sparse/ + images/.")
        state.notify("Could not load dataset", ERROR)

    def _hub_request_training(self, *, restart: bool = False) -> "str | None":
        """Only the restart action can replace a result. Both paths check the STBN license."""
        state = self.app._viewer.state
        snap = state.snapshot()
        if not snap.trainable:
            state.notify("No trainable scene yet", WARN)
            return None
        if not (snap.can_restart_training if restart else snap.can_start_training):
            return None
        if not self._stbn_accepted and self._stbn_download_needed(state.setup):
            state.ask(Confirm(
                key="training", title="This run needs NVIDIA's STBN masks",
                body="About 30 MB from github.com/NVIDIA-RTX/STBN, licensed for non-commercial "
                     "research only. Balanced and Quality don't use them.",
                command="download_stbn_then_restart" if restart else "download_stbn_then_train",
                confirm_label="Download and train"))
            return None
        return state.scene_dir

    def _hub_train_scene(self, scene_dir: str, setup) -> None:
        """Build and train one run on ``scene_dir``, feeding the running viewer.
        Setup runs with rendering suspended (queue races); a failure leaves the hub usable."""
        state = self.app._viewer.state
        print(f"[hub] training on {scene_dir}")
        started = time.perf_counter()
        # Keep the previous model available if a required download fails.
        if not self._hub_ensure_stbn(setup):
            return
        state.reset_for_new_run()
        state.training_initializing(setup.get("iters"))
        state.begin_activity("Preparing training images and GPU buffers")
        self.app.viewer.clear_stop_request()
        state.log(f"preparing training on {scene_dir}")
        self.app.viewer.suspend_rendering()
        try:
            self.app._begin_repeat_run()
            # The panel's settings must become the run's args before anything reads them.
            self._apply_hub_setup(setup)
            self.app.prepare_run(scene_dir, self._cached_colmap(scene_dir))
        except Exception as error:                     # noqa: BLE001
            self._hub_run_failed("building the run", error)
            return
        finally:
            self.app.viewer.resume_rendering()

        self.app.viewer.attach_scene(self.app._sceneContext.camera_sets())
        self.app.viewer.push_snapshot(self.app._pipeline)  # show the initial gaussians
        state.log(f"Ready to train after {time.perf_counter() - started:.2f} s of setup")
        try:
            summary = self.app.run_training_and_post(hold_viewer=False)
        except Exception as error:                     # noqa: BLE001
            self._hub_run_failed("training", error)
            return
        if summary is None:
            return
        # The pipeline stays alive: Export writes it and the viewer keeps showing it.
        stopped = self.app.viewer.stop_requested()
        auto_export = self.app.paths.root / "output.ply"
        if (self.app.args.save_ply or self.app.args.compute_final_metrics) and auto_export.is_file():
            state.result_exported(str(auto_export.resolve()), auto_export.stat().st_size / 1024**2)
        state.notify(f"Training stopped at iteration {summary.iterations:,}" if stopped
                     else f"Training complete in "
                          f"{format_duration(summary.wall_s)}", OK)
        state.log("Training finished")

    def _stbn_download_needed(self, setup) -> bool:
        """Whether the run samples NVIDIA's STBN masks and they are not cached yet."""
        args = self.app.args
        return (setup.get("stbn_pixel_fraction") < 1.0
                and args.stbn_source == stbn_table.SOURCE_NVIDIA and not args.stbn_file
                and not stbn_table.nvidia_volume_path().is_file())

    def _hub_ensure_stbn(self, setup) -> bool:
        """Download the accepted masks if needed. A failed download leaves the previous model available."""
        if not (self._stbn_accepted and self._stbn_download_needed(setup)):
            return True
        from trainer.frontend import stbn_download

        state = self.app._viewer.state
        state.training_initializing(setup.get("iters"))
        state.begin_activity("Downloading NVIDIA's STBN masks")
        try:
            stbn_download.download(log=state.log)
        except Exception as error:                     # noqa: BLE001
            self._hub_run_failed("downloading NVIDIA's STBN masks", error)
            return False
        return True

    def _hub_run_failed(self, what: str, error: Exception) -> None:
        """Report a failed hub run and leave the session usable."""
        state = self.app._viewer.state
        state.training_failed(f"{what}: {error}")
        state.notify(f"Training failed while {what}", ERROR)
        report_exception(f"[hub] {what} failed", error, state)

    def _hub_export_ply(self) -> bool:
        """Choose a destination before downloading the trained model from the GPU."""
        from slangpy import platform
        from trainer.backend.export import save_ply

        state = self.app._viewer.state
        if self.app.pipeline is None:
            state.result_failed("Nothing to export yet - train a scene first.")
            state.notify("Nothing to export", WARN)
            return False
        call_ui = self.app._viewer.call_on_main_thread
        try:
            filters = [platform.FileDialogFilter("Gaussian model", "*.ply")]
            path = call_ui(lambda: platform.save_file_dialog(filters))
        except (AttributeError, OSError, RuntimeError):
            path = self.app.paths.root / "output.ply"
            if path.exists():
                state.result_failed(f"Save dialog unavailable. A model already exists at {path}.")
                return False
        if not path:
            return False
        if path.suffix.lower() != ".ply":
            path = path.with_name(path.name + ".ply")
            if path.exists():
                state.result_failed(f"{path.name} already exists. Select that file to replace it.")
                return False
        state.begin_activity(f"Exporting {path.name}")
        state.log(f"exporting {path.name} …")
        # save_ply reads back the gaussian buffers, so the render thread must be off the queue.
        self.app.viewer.suspend_rendering()
        try:
            with TemporaryDirectory(dir=path.parent) as staging:
                pending = Path(staging) / path.name
                save_ply(self.app._pipeline, pending)
                pending.replace(path)
        except Exception as error:                     # noqa: BLE001
            state.result_failed(f"Could not write {path}: {error}")
            state.notify("Export failed", ERROR)
            report_exception("[hub] export failed", error, state)
            return False
        finally:
            self.app.viewer.resume_rendering()
        size_mb = path.stat().st_size / (1024 * 1024)
        state.result_exported(str(path.resolve()), size_mb)
        state.notify(f"Exported {path.name}", OK)
        state.log(f"exported {size_mb:,.1f} MB -> {path.resolve()}")
        print(f"[hub] exported {path.resolve()} ({size_mb:,.1f} MB)")
        return True

    def _hub_reveal_export(self) -> None:
        """Open the exported file's folder in the OS file manager."""
        state = self.app._viewer.state
        exported = state.snapshot().result.exported_path
        if not exported:
            return
        folder = Path(exported).parent
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            elif os.name == "nt":
                os.startfile(str(folder))   # user-initiated, this machine only
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except (AttributeError, OSError) as error:
            state.result_failed(f"Could not open {folder}: {error}")
            state.notify("Could not open the folder", ERROR)
            report_exception("[hub] opening the export folder failed", error, state)

    def _hub_clear_session(self, *, notify: bool = True) -> None:
        """Forget the dataset and the finished run so the hub can start over.
        The window, device, installed tools and log survive."""
        self._scene_cache = None
        state = self.app._viewer.state
        self.app.viewer.suspend_rendering()
        try:
            self.app._release_run_resources()
        finally:
            self.app.viewer.resume_rendering()
        self.app.viewer.clear_view()
        state.clear_session()
        if notify:
            state.notify("Session cleared", OK)
        state.log("session cleared — drag another dataset onto the window")
        print("[hub] session cleared")

    def _apply_hub_setup(self, setup) -> None:
        """Fold the panel's training setup into this run's args; call with rendering suspended.
        Also rebuilds the viewer's pipeline, whose SH buffers, Mip raster path and compiled
        shader specialization are fixed at construction."""
        state = self.app._viewer.state
        for change in setup.apply_to(self.app.args):
            state.log(f"[setup] {change}")
        self.app.viewer.reconfigure_model(
            sh_degree=self.app.args.sh_degree,
            mip_enabled=self.app.args.mip_splatting,
            mip_kernel_size=(self.app.args.mip_kernel_size
                             if self.app.args.mip_splatting else 0.3),
            slang_session=self.app.slang_session(),
        )

    def _hub_setup_tools(self, tools) -> None:
        """Download the colmap/glomap binaries, with a determinate bar."""
        from trainer.frontend.reconstruction.progress import setup_parser

        state = self.app._viewer.state
        state.tools(ToolsState.INSTALLING)
        # The panel's checkbox is the install-time build choice (see ReconTools).
        tools.recon_cuda = state.recon_cuda
        ready = tools.setup(progress=_progress_relay(state, setup_parser()))
        state.tools(ToolsState.READY if ready else ToolsState.MISSING)
        state.notify("Reconstruction tools ready" if ready
                     else "Tool install failed - see diagnostics",
                     OK if ready else ERROR)
        state.log("reconstruction tools ready" if ready
                  else "tool setup failed — see the log above")

    def _hub_reconstruct(self, tools) -> None:
        """Run images -> COLMAP for the dropped dataset, with a determinate bar."""
        from trainer.frontend.reconstruction import reconstruct
        from trainer.frontend.reconstruction.progress import reconstruction_parser

        state = self.app._viewer.state
        source = state.dataset_path
        if not source:
            state.notify("No dataset to reconstruct", WARN)
            return
        self._scene_cache = None
        state.dataset_reconstructing()
        out = Path(source) / "reconstruction"
        result = reconstruct(tools, source, out, matcher=state.matcher,
                             progress=_progress_relay(state, reconstruction_parser()))
        if result is None:
            state.dataset_failed("Reconstruction failed - see diagnostics for the "
                                 "COLMAP output.")
            state.notify("Reconstruction failed", ERROR)
            return
        state.dataset_detected(KIND_IMAGES, scene_dir=str(result))
        state.notify("Reconstruction complete", OK)
        state.log("reconstruction complete")
        self._hub_preview_scene(str(result))

    def _hub_preview_scene(self, scene_dir: str) -> None:
        """Show the dataset's SfM points before Train so a bad reconstruction is visible early.
        Best-effort: a scene the loader rejects leaves the hub usable."""
        self._scene_cache = None
        state = self.app._viewer.state
        state.dataset_loading("Reading the COLMAP model")
        state.log("loading the SfM point cloud …")
        started = time.time()
        # Suspend during the load too: the render thread holds the GIL and starves the parsers.
        self.app.viewer.suspend_rendering()
        try:
            dataset = load_colmap(
                scene_dir,
                width=self.app.args.width,
                height=self.app.args.height,
                downsample=self.app.args.downsample,
                init_opacity=self.app.args.init_opacity,
            )
            shown = self.app.viewer.preview_points(dataset.initial_points)
            self._scene_cache = self._colmap_key(scene_dir, dataset), dataset
        except Exception as error:                    # noqa: BLE001
            state.dataset_failed(str(error))
            state.notify("Could not load dataset", ERROR)
            report_exception(f"[hub] cannot preview this dataset: {scene_dir}", error, state)
            return
        finally:
            self.app.viewer.resume_rendering()

        # The run's split is fixed by --eval / --eval-holdout-stride, so the preview can show it already.
        args = self.app.args
        test, train = holdout_split(len(dataset.images), max(args.eval_holdout_stride, 1),
                                    bool(args.eval_holdout))
        self.app.viewer.attach_scene(split_camera_sets(dataset.images, test, train), snap=True)
        first = dataset.images[0].camera if dataset.images else None
        info = state.snapshot().dataset
        state.dataset_ready(DatasetInfo(
            path=info.path or scene_dir, kind=info.kind or KIND_COLMAP,
            images=len(dataset.images),
            width=int(getattr(first, "width", 0) or 0),
            height=int(getattr(first, "height", 0) or 0),
            points=int(shown), scene_dir=info.scene_dir or scene_dir))
        state.result_initialized(int(shown))
        state.notify(f"{info.kind_label} loaded - {len(dataset.images)} images", OK)
        state.log(f"previewing {shown:,} SfM points from {len(dataset.images)} "
                  f"cameras — press Train to start")
        print(f"[hub] previewing {shown:,} initial points from {scene_dir} "
              f"({time.time() - started:.1f}s)")
