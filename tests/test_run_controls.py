# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Cooperative controls across optimizer, evaluation, and finalization phases."""
from __future__ import annotations

import io
import json
import queue
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from trainer.frontend import app as app_module
from trainer.frontend.app import RunPaths, TrainingApplication, TrainingLoop, TrainingResult
from trainer.frontend.evaluation import evaluation, post_training, run_report
from trainer.frontend.run_control import RunStopped
from trainer.frontend.viewer.session import MODE_HUB, ResultInfo, Session, TrainingState
from trainer.frontend.viewer.viewer import Viewer
from trainer.frontend.viewer.viewer_control import ViewerSession


def _viewer() -> Viewer:
    """Use the real event-backed controls without constructing a GPU or window."""
    viewer = object.__new__(Viewer)
    viewer.state = Session(MODE_HUB)
    viewer.state.training_running()
    viewer.state.training_finalizing("Final metrics")
    viewer._training_active = False
    viewer._ui_thread_id = None
    viewer._main_thread_calls = queue.Queue()
    for name in ("_pause_event", "_stop_training", "_stop_event",
                 "_closed_by_user", "_started", "_exited",
                 "_frame_budget_off", "_scene_dirty"):
        setattr(viewer, name, threading.Event())
    return viewer


class _Pipeline:
    def __init__(self) -> None:
        self.rendered: list[int] = []
        self.on_render = lambda _count: None
        self.runtime = SimpleNamespace(wait_for_idle=lambda: None)
        self.max_splats, self.max_sort_pairs = 7, 64
        self.primitives = SimpleNamespace(count=lambda: 7)
        self.buffers = SimpleNamespace(sort=SimpleNamespace(
            stats=SimpleNamespace(to_numpy=lambda: np.array([9, 0], dtype=np.uint32))))

    def render(self, camera) -> None:
        self.rendered.append(camera)
        self.on_render(len(self.rendered))

    def read_output_image(self) -> np.ndarray:
        return np.full((16, 16, 4), 0.25, dtype=np.float32)

    def read_bin_debug(self) -> tuple[np.ndarray, np.ndarray, int, int]:
        return (np.full(4, 3, dtype=np.uint32), np.full((16, 16), 2, dtype=np.uint32),
                2, 2)


def _scene():
    return SimpleNamespace(images=[SimpleNamespace(camera=i, image_name=f"{i}.png")
                                   for i in range(3)])


def _targets() -> list[np.ndarray]:
    return [np.full((16, 16, 3), 0.5, dtype=np.float32) for _ in range(3)]


def _run_pass(kind, pipeline, session, directory, stream):
    scene, targets = _scene(), _targets()
    if kind == "camera_dump":
        return evaluation.render_all_cameras(
            pipeline, scene, 3, directory / "cams", checkpoint=session.checkpoint)
    if kind == "test_psnr":
        return evaluation.eval_test_psnr_srgb(
            pipeline, scene, [0, 1, 2], targets, checkpoint=session.checkpoint)
    if kind == "eval_sweep":
        return evaluation.run_eval_sweep(
            pipeline=pipeline, scene=scene, it=3, test_indices=[0, 1, 2],
            targets_srgb=targets, targets_linear=targets.__getitem__,
            cams_dir=directory / "cams", eval_curve_json=stream,
            t_start=time.time(), eval_time_total=0.0, eval_holdout=True,
            checkpoint=session.checkpoint)
    if kind == "test_report":
        return run_report.evaluate_test_views(
            pipeline, scene, [0, 1, 2], targets, render_dir=None,
            checkpoint=session.checkpoint)
    if kind == "render_fps":
        return run_report.measure_render_fps(
            pipeline, scene, 0, 3, checkpoint=session.checkpoint)
    assert kind == "final_metrics"
    return post_training.run_post_training(
        pipeline=pipeline, scene=scene,
        args=SimpleNamespace(profile_run=False, compute_final_metrics=True,
                             evaluation=True, psnr_curve=False,
                             mip_splatting=False, iters=3, eval_holdout=True),
        run_dir=directory, cams_dir=directory / "cams",
        targets_srgb=targets, targets_linear=targets.__getitem__,
        test_indices=[0, 1, 2], stride=1, n_cams=3, train_wall_s=1.0,
        completed_iters=3, t_start=time.time(), checkpoint=session.checkpoint)


PASSES = ("camera_dump", "test_psnr", "eval_sweep", "test_report", "render_fps",
          "final_metrics")


def test_save_ply_without_scoring(tmp_path, monkeypatch):
    saved = []
    monkeypatch.setattr(post_training, "save_ply", lambda pipeline, path: saved.append(path))
    post_training.run_post_training(
        pipeline=None, scene=None,
        args=SimpleNamespace(profile_run=False, compute_final_metrics=False, evaluation=False,
                             save_ply=True, psnr_curve=False, mip_splatting=True),
        run_dir=tmp_path, cams_dir=tmp_path, targets_srgb=[], targets_linear=None,
        test_indices=[], stride=8, n_cams=0, train_wall_s=0, completed_iters=3, t_start=time.time())
    assert saved == [tmp_path / "output.ply"]


@pytest.mark.parametrize("kind", PASSES)
@pytest.mark.parametrize("stop_after", [1, 3])
def test_stop_interrupts_each_camera_pass_without_publishing_partial_scores(
    kind, stop_after, tmp_path,
) -> None:
    viewer, pipeline = _viewer(), _Pipeline()
    session, stream = ViewerSession(viewer), io.StringIO()
    pipeline.on_render = lambda count: (viewer.request_stop_training()
                                         if count == stop_after else None)

    with pytest.raises(RunStopped):
        _run_pass(kind, pipeline, session, tmp_path, stream)

    assert len(pipeline.rendered) == stop_after
    assert viewer.state.snapshot().training_state == TrainingState.STOPPING
    assert stream.getvalue() == ""
    assert not (tmp_path / "final_psnr.json").exists()
    assert not (tmp_path / "output.ply").exists()


@pytest.mark.parametrize("kind", PASSES)
def test_uninterrupted_passes_still_complete(kind, tmp_path, monkeypatch) -> None:
    pipeline = _Pipeline()
    stream = io.StringIO()
    monkeypatch.setattr(post_training, "save_ply",
                        lambda _pipeline, path: path.write_text("model"))

    _run_pass(kind, pipeline, ViewerSession(None), tmp_path, stream)

    expected = (run_report.FPS_WARMUP_FRAMES + 3 if kind == "render_fps"
                else 6 if kind == "final_metrics" else 3)
    assert len(pipeline.rendered) == expected
    if kind == "eval_sweep":
        assert json.loads(stream.getvalue())["n_eval_cams"] == 3
    if kind == "final_metrics":
        summary = json.loads((tmp_path / "final_psnr.json").read_text())
        assert summary["n_test_cams"] == 3
        assert (tmp_path / "output.ply").read_text() == "model"


@pytest.mark.parametrize("action", ["resume", "stop", "close"])
def test_final_evaluation_waits_for_resume_or_exits_on_stop(
    action, tmp_path, monkeypatch,
) -> None:
    viewer, pipeline = _viewer(), _Pipeline()
    session, stream = ViewerSession(viewer), io.StringIO()
    paused, done = threading.Event(), threading.Event()
    errors = []
    publish_pause = viewer.state.training_paused

    def acknowledge_pause():
        publish_pause()
        paused.set()

    monkeypatch.setattr(viewer.state, "training_paused", acknowledge_pause)
    pipeline.on_render = lambda count: viewer.pause_training() if count == 1 else None

    def evaluate():
        try:
            _run_pass("eval_sweep", pipeline, session, tmp_path, stream)
        except Exception as error:
            errors.append(error)
        finally:
            done.set()

    worker = threading.Thread(target=evaluate, daemon=True)
    worker.start()
    try:
        assert paused.wait(3), "evaluation never acknowledged Pause"
        assert pipeline.rendered == [0]
        assert not done.is_set()
        assert viewer.state.snapshot().training_state == TrainingState.PAUSED
        if action == "resume":
            viewer.resume_training()
        elif action == "stop":
            viewer.request_stop_training()
        else:
            viewer.stop()
        assert done.wait(3), "evaluation did not respond to the control"
    finally:
        viewer.request_stop_training()
        worker.join(3)

    if action == "resume":
        assert not errors
        assert pipeline.rendered == [0, 1, 2]
        assert json.loads(stream.getvalue())["n_eval_cams"] == 3
    else:
        assert len(errors) == 1 and isinstance(errors[0], RunStopped)
        assert pipeline.rendered == [0]
        assert stream.getvalue() == ""


def test_stop_also_interrupts_the_final_metrics_second_render_pass(tmp_path) -> None:
    viewer, pipeline = _viewer(), _Pipeline()
    pipeline.on_render = lambda count: viewer.request_stop_training() if count == 4 else None

    with pytest.raises(RunStopped):
        _run_pass("final_metrics", pipeline, ViewerSession(viewer), tmp_path, io.StringIO())

    assert len(pipeline.rendered) == 4
    assert not (tmp_path / "final_psnr.json").exists()
    assert not (tmp_path / "output.ply").exists()


def test_stop_interrupts_measured_fps_frames_not_only_warmup() -> None:
    viewer, pipeline = _viewer(), _Pipeline()
    stop_after = run_report.FPS_WARMUP_FRAMES + 1
    pipeline.on_render = lambda count: viewer.request_stop_training() if count == stop_after else None

    with pytest.raises(RunStopped):
        run_report.measure_render_fps(
            pipeline, _scene(), 0, 3, checkpoint=ViewerSession(viewer).checkpoint)

    assert len(pipeline.rendered) == stop_after


def test_fps_measurement_does_not_count_time_waiting_at_a_checkpoint(monkeypatch) -> None:
    pipeline = _Pipeline()
    clock = [0.0]
    pipeline.on_render = lambda _count: clock.__setitem__(0, clock[0] + 0.1)
    monkeypatch.setattr(run_report.time, "perf_counter", lambda: clock[0])

    def checkpoint():
        clock[0] += 30.0

    result = run_report.measure_render_fps(pipeline, _scene(), 0, 3,
                                         checkpoint=checkpoint)

    assert result["fps"] == pytest.approx(10.0)
    assert result["total_s"] == pytest.approx(0.3)


def _application(tmp_path, viewer, pipeline, events):
    app = object.__new__(TrainingApplication)
    app.args = SimpleNamespace(iters=3, profile_run=False, evaluation=True,
                               final_test_psnr=True, run_report=False,
                               flat_bucket_backward=False)
    app.run_name = "control-test"
    app.paths = RunPaths(root=tmp_path, cameras=tmp_path / "cams")
    app.viewer = ViewerSession(viewer)
    viewer.update_gaussians = lambda _pipeline: events.append("snapshot")
    app.pipeline = pipeline
    app.device = SimpleNamespace(info=SimpleNamespace(adapter_name="test"))
    app._stream_targets = False
    app._memory_summary = {}
    app.targets = SimpleNamespace(srgb=_targets(), linear=_targets().__getitem__)
    app.scene_context = SimpleNamespace(colmapScene=_scene(), test_indices=[0, 1, 2],
                                        holdout_stride=1, camera_count=3)
    app.runtime = SimpleNamespace(
        pipeline=pipeline, viewer=app.viewer,
        profiler=SimpleNamespace(close=lambda: events.append("profiler closed")),
        artifacts=SimpleNamespace(log_json=io.StringIO(), close_all=lambda: events.append("artifacts closed")))
    result = TrainingResult(iterations=3, wall_time_seconds=1.0,
                            psnr_curve=[], started_at=time.time())
    app.build_training_loop = lambda: SimpleNamespace(launch_training_loop=lambda: result)
    return app


@pytest.mark.parametrize("want_report", [False, True])
def test_final_psnr_and_run_report_use_the_viewer_checkpoint(want_report, tmp_path) -> None:
    viewer, pipeline = _viewer(), _Pipeline()
    app = _application(tmp_path, viewer, pipeline, [])
    app.args.run_report = want_report
    pipeline.on_render = lambda _count: viewer.request_stop_training()

    with pytest.raises(RunStopped):
        app._write_final_test_psnr(app.build_training_loop().launch_training_loop())

    assert pipeline.rendered == [0]
    assert not (tmp_path / "test_psnr.json").exists()
    assert not (tmp_path / "run_report.json").exists()


def test_final_psnr_separates_training_and_heldout_sort_overflows(tmp_path) -> None:
    viewer, pipeline = _viewer(), _Pipeline()
    app = _application(tmp_path, viewer, pipeline, [])
    app._memory_summary = {"sort_overflows": 2}
    pipeline.buffers.sort.stats.to_numpy = lambda: np.array([9, 3], dtype=np.uint32)
    app._write_final_test_psnr(app.build_training_loop().launch_training_loop())
    report = json.loads((tmp_path / "test_psnr.json").read_text())
    assert (report["sort_overflows"], report["heldout_sort_overflows"]) == (2, 1)


def test_stopping_final_evaluation_retains_the_pipeline_and_closes_resources(
    tmp_path, monkeypatch,
) -> None:
    viewer, pipeline, events = _viewer(), _Pipeline(), []
    app = _application(tmp_path, viewer, pipeline, events)
    pipeline.on_render = lambda _count: viewer.request_stop_training()
    monkeypatch.setattr(app_module, "run_post_training",
                        lambda **_kwargs: pytest.fail("another pass started after Stop"))

    summary = app.run_training_and_post(hold_viewer=False)

    assert app.pipeline is pipeline
    assert pipeline.rendered == [0]
    assert events == ["snapshot", "profiler closed", "artifacts closed"]
    assert viewer.stop_requested()
    assert viewer.state.snapshot().training_state == TrainingState.COMPLETED
    assert viewer.state.snapshot().result == summary
    assert not viewer._frame_budget_off.is_set()


def test_cli_metrics_complete_before_waiting_for_the_viewer(tmp_path, monkeypatch) -> None:
    viewer, pipeline, events = _viewer(), _Pipeline(), []
    app = _application(tmp_path, viewer, pipeline, events)
    monkeypatch.setattr(app, "_write_final_test_psnr", lambda _result: events.append("held-out"))
    monkeypatch.setattr(app_module, "run_post_training", lambda **_kwargs: events.append("metrics"))
    monkeypatch.setattr(app.viewer, "hold_after_training", lambda _pipeline: events.append("hold"))

    app.run_training_and_post()

    assert events == ["snapshot", "held-out", "metrics", "profiler closed",
                      "artifacts closed", "hold"]
    assert viewer.state.snapshot().training_state == TrainingState.COMPLETED


def test_stop_during_pause_does_not_start_another_optimizer_iteration(monkeypatch) -> None:
    viewer = _viewer()
    session = ViewerSession(viewer)
    loop = object.__new__(TrainingLoop)
    loop.runtime = SimpleNamespace(viewer=session)
    loop._begin_training_loop = lambda: None
    loop._should_continue = lambda: True
    loop._next_iteration = lambda: pytest.fail("optimizer advanced after Stop")
    loop._complete_training_loop = lambda: "retained model"
    monkeypatch.setattr(viewer, "wait_while_paused", viewer.request_stop_training)

    assert loop.launch_training_loop() == "retained model"


def test_late_finalize_progress_does_not_revive_a_stopping_run(tmp_path) -> None:
    viewer = _viewer()
    app = _application(tmp_path, viewer, _Pipeline(), [])
    viewer.request_stop_training()

    app._publish_finalize("Final metrics", 0.5)

    assert viewer.state.snapshot().training_state == TrainingState.STOPPING
    assert viewer.state.snapshot().activity.label == "Stopping training"


def test_late_stop_does_not_reopen_a_completed_run() -> None:
    viewer = _viewer()
    result = ResultInfo(iterations=3, target_iterations=3)
    viewer.state.training_completed(result)

    viewer.request_stop_training()

    assert viewer.state.snapshot().training_state == TrainingState.COMPLETED
    assert viewer.state.snapshot().result == result
    assert not viewer.stop_requested()


def test_stop_does_not_overwrite_completion_after_waking_the_worker(monkeypatch) -> None:
    viewer = _viewer()
    result = ResultInfo(iterations=3, target_iterations=3)
    monkeypatch.setattr(viewer._pause_event, "clear",
                        lambda: viewer.state.training_completed(result))

    viewer.request_stop_training()

    assert viewer.stop_requested()
    assert viewer.state.snapshot().training_state == TrainingState.COMPLETED
    assert viewer.state.snapshot().result == result


def test_run_with_viewer_runs_ui_loop_and_dialogs_on_main_thread() -> None:
    viewer = _viewer()
    app = object.__new__(TrainingApplication)
    app.viewer = ViewerSession(viewer)
    seen: dict[str, int] = {}

    def fake_inner() -> None:
        seen["viewer"] = threading.get_ident()
        while not viewer._stop_event.is_set():
            viewer._run_main_thread_calls()
            time.sleep(0.005)

    viewer._run_inner = fake_inner

    def worker() -> None:
        seen["worker"] = threading.get_ident()
        assert app.viewer.is_alive()
        seen["dialog"] = app.viewer.call_on_main_thread(threading.get_ident)
        viewer.stop()
        viewer.wait_until_closed()

    app.run_with_viewer(worker)

    main_id = threading.get_ident()
    assert seen["viewer"] == main_id
    assert seen["dialog"] == main_id
    assert seen["worker"] != main_id
    assert viewer.has_exited()


def test_run_with_viewer_without_viewer_runs_inline() -> None:
    app = object.__new__(TrainingApplication)
    app.viewer = ViewerSession(None)
    ran_on: list[int] = []
    app.run_with_viewer(lambda: ran_on.append(threading.get_ident()))
    assert ran_on == [threading.get_ident()]


def test_worker_failure_closes_viewer_and_propagates() -> None:
    viewer = _viewer()
    viewer._run_inner = lambda: viewer._stop_event.wait(5.0)
    app = object.__new__(TrainingApplication)
    app.viewer = ViewerSession(viewer)

    def worker() -> None:
        raise RuntimeError("boom during run")

    with pytest.raises(RuntimeError, match="boom during run"):
        app.run_with_viewer(worker)

    assert viewer.has_exited()


def test_viewer_crash_does_not_abort_active_worker() -> None:
    viewer = _viewer()
    app = object.__new__(TrainingApplication)
    app.viewer = ViewerSession(viewer)
    viewer._run_inner = lambda: (_ for _ in ()).throw(RuntimeError("swapchain lost"))
    completed: list[bool] = []

    def worker() -> None:
        viewer.wait_until_closed()
        assert not app.viewer.stop_requested()
        completed.append(True)

    app.run_with_viewer(worker)
    assert completed == [True]
