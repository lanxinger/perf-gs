# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Errors survive native stderr writes and exits that cannot unwind Python."""
from __future__ import annotations

import faulthandler
import os
import subprocess
import sys
import textwrap
import threading
from types import SimpleNamespace

import pytest
import slangpy as spy

from trainer.frontend import app as app_module
from trainer.frontend.app import RunPaths, TrainingApplication
from trainer.frontend.hub import TrainingHub
from trainer.frontend.reporting.error_log import ErrorLog, report_exception
from trainer.frontend.viewer.session import MODE_HUB, Session, TrainingState
from trainer.frontend.viewer.viewer import Viewer
from trainer.frontend.viewer.viewer_control import ViewerSession


def test_python_and_native_errors_reach_disk_terminal_and_viewer(tmp_path, capfd):
    path = tmp_path / "errors.log"
    state = Session(MODE_HUB)
    previous = sys.stderr
    previous_fault_handler = faulthandler.is_enabled()
    with ErrorLog(path) as log:
        log.attach(state)
        print("python stderr", file=sys.stderr)
        os.write(2, b"native stderr\n")
        spy.log_error("VK_ERROR_OUT_OF_DEVICE_MEMORY")
        log.flush()
        # The file must be readable/flushed while the native writer is still open.
        text = path.read_text(encoding="utf-8")
        assert "python stderr" in text
        assert "native stderr" in text
        assert "VK_ERROR_OUT_OF_DEVICE_MEMORY" in text
        assert any("VK_ERROR_OUT_OF_DEVICE_MEMORY" in line
                   for line in state.snapshot().log)
        assert state.snapshot().show_log
    assert sys.stderr is previous
    assert faulthandler.is_enabled() == previous_fault_handler
    os.write(2, b"after capture\n")
    assert "after capture" not in path.read_text(encoding="utf-8")
    terminal = capfd.readouterr()
    assert "python stderr" in terminal.err
    assert "native stderr" in terminal.err
    assert "VK_ERROR_OUT_OF_DEVICE_MEMORY" in terminal.err
    assert "after capture" in terminal.err


def test_attach_replays_early_setup_errors_and_chunked_unicode(tmp_path, capfd):
    path = tmp_path / "errors.log"
    with ErrorLog(path) as log:
        encoded = "early setup: caf\u00e9\n".encode("utf-8")
        os.write(2, encoded[:-2])
        log.flush()
        os.write(2, encoded[-2:])
        log.flush()
        state = Session(MODE_HUB)
        log.attach(state)
        assert "early setup: caf\u00e9" in state.snapshot().log
        assert state.snapshot().show_log
        os.write(2, b"last line without newline")
    assert "last line without newline" in state.snapshot().log
    assert "caf\u00e9" in path.read_text(encoding="utf-8")


def test_background_forwarder_shows_errors_without_explicit_flush(tmp_path, capfd):
    ready = threading.Event()
    state = Session(MODE_HUB)
    original = state.log

    def publish(message):
        original(message)
        if message == "background error":
            ready.set()

    state.log = publish
    with ErrorLog(tmp_path / "errors.log") as log:
        log.attach(state)
        os.write(2, b"background error\n")
        assert ready.wait(3), "stderr was not forwarded to the live panel"


def test_hub_capture_remains_attached_across_run_resets(tmp_path, capfd):
    state = Session(MODE_HUB)
    with ErrorLog(tmp_path / "errors.log") as log:
        log.attach(state)
        state.reset_for_new_run()
        os.write(2, b"second run error\n")
        log.flush()
        assert "second run error" in state.snapshot().log
        assert state.snapshot().show_log


def test_caught_exception_keeps_chained_traceback_and_does_not_require_capture(capfd):
    state = Session(MODE_HUB)
    try:
        try:
            raise ValueError("allocation detail")
        except ValueError as cause:
            raise RuntimeError("Vulkan allocation failed") from cause
    except RuntimeError as error:
        report_exception("creating training pipeline", error, state)
    assert any("creating training pipeline" in line for line in state.snapshot().log)
    assert state.snapshot().show_log
    terminal = capfd.readouterr().err
    assert "ValueError: allocation detail" in terminal
    assert "RuntimeError: Vulkan allocation failed" in terminal
    assert "Traceback (most recent call last)" in terminal


def test_escaping_exception_is_persisted_before_stderr_is_restored(tmp_path, capfd):
    path = tmp_path / "errors.log"
    previous = sys.stderr
    with pytest.raises(RuntimeError, match="setup failure"), ErrorLog(path):
        raise RuntimeError("setup failure")
    assert sys.stderr is previous
    assert "RuntimeError: setup failure" in path.read_text(encoding="utf-8")


def test_capture_initialization_failure_restores_stderr(tmp_path, monkeypatch, capfd):
    previous = sys.stderr
    log = ErrorLog(tmp_path / "errors.log")
    monkeypatch.setattr(faulthandler, "is_enabled", lambda: False)
    monkeypatch.setattr(faulthandler, "disable", lambda: None)

    def fail(**_kwargs):
        raise RuntimeError("fault handler setup failed")

    monkeypatch.setattr(faulthandler, "enable", fail)
    with pytest.raises(RuntimeError, match="fault handler setup failed"), log:
        pytest.fail("capture unexpectedly started")
    assert sys.stderr is previous
    assert log._file.closed
    assert log._reader.closed
    assert log._stderr.closed
    os.write(2, b"restored after initialization failure\n")
    assert "restored after initialization failure" in capfd.readouterr().err


@pytest.mark.parametrize("hub", [False, True])
def test_main_installs_capture_before_device_setup_and_keeps_failure_exit_code(
    hub, tmp_path, monkeypatch, capfd,
):
    app = object.__new__(TrainingApplication)
    app.args = SimpleNamespace(hub=hub)
    app.paths = RunPaths(tmp_path, tmp_path / "cams")
    app.viewer = None

    def fail_during_setup():
        assert isinstance(app.error_log, ErrorLog)
        os.write(2, b"native GPU setup failed\n")
        raise RuntimeError("VK_ERROR_INITIALIZATION_FAILED")

    monkeypatch.setattr(TrainingHub, "run", lambda self: fail_during_setup())
    app.start_training = fail_during_setup
    monkeypatch.setattr(app_module.TrainingApplication, "from_cli", lambda: app)
    with pytest.raises(SystemExit) as result:
        app_module.main()
    assert result.value.code == 1
    text = (tmp_path / "errors.log").read_text(encoding="utf-8")
    assert "native GPU setup failed" in text
    assert "RuntimeError: VK_ERROR_INITIALIZATION_FAILED" in text


def test_hub_failure_reports_to_existing_diagnostics_panel(tmp_path, capfd):
    app = object.__new__(TrainingApplication)
    app.viewer = ViewerSession(None, Session(MODE_HUB))
    with ErrorLog(tmp_path / "errors.log") as log:
        log.attach(app.viewer.state)
        try:
            raise RuntimeError("VK_ERROR_DEVICE_LOST")
        except RuntimeError as error:
            TrainingHub(app)._hub_run_failed("building the run", error)
        log.flush()
    snap = app.viewer.state.snapshot()
    assert snap.training_state == TrainingState.FAILED
    assert snap.show_log
    assert any("Traceback" in line for line in snap.log)
    assert any("VK_ERROR_DEVICE_LOST" in line for line in snap.log)


def test_viewer_reports_different_errors_but_does_not_flood_each_frame(capfd):
    viewer = object.__new__(Viewer)
    viewer.state = Session(MODE_HUB)
    viewer._report_error("frame error", RuntimeError("first failure"))
    viewer._report_error("frame error", RuntimeError("first failure"))
    viewer._report_error("frame error", RuntimeError("device lost"))
    text = capfd.readouterr().err
    assert text.count("[error] [Viewer] frame error: RuntimeError: first failure") == 1
    assert "device lost" in text
    assert viewer.state.snapshot().show_log
    assert len(viewer.state.snapshot().log) == 2


def test_viewer_thread_failure_is_logged_and_wakes_paused_training(capfd):
    viewer = object.__new__(Viewer)
    viewer.state = Session(MODE_HUB)
    for name in ("_stop_event", "_closed_by_user", "_stop_training",
                 "_pause_event", "_started", "_exited"):
        setattr(viewer, name, threading.Event())
    viewer._pause_event.set()

    def fail():
        raise RuntimeError("VK_ERROR_DEVICE_LOST")

    viewer._run_inner = fail
    viewer.run()
    assert viewer._stop_event.is_set()
    assert viewer.has_exited()
    assert not viewer.stop_requested()
    assert not viewer._pause_event.is_set()
    assert viewer.state.snapshot().show_log
    assert any("crashed" in line for line in viewer.state.snapshot().log)
    assert "RuntimeError: VK_ERROR_DEVICE_LOST" in capfd.readouterr().err


def test_abrupt_native_style_exit_keeps_error_without_running_cleanup(tmp_path):
    path = tmp_path / "errors.log"
    child = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""
            import os, sys
            from pathlib import Path
            from trainer.frontend.reporting.error_log import ErrorLog
            with ErrorLog(Path(sys.argv[1])):
                os.write(2, b'Assertion failed: Vulkan returned a failure\\n')
                os._exit(27)
            """), str(path)], capture_output=True, check=False, timeout=30)
    assert child.returncode == 27, child.stderr.decode("utf-8", "replace")
    assert "Assertion failed: Vulkan returned a failure" in path.read_text(encoding="utf-8")


def test_native_abort_keeps_fatal_stack_and_prior_error(tmp_path):
    path = tmp_path / "errors.log"
    child = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""
            import ctypes, faulthandler, os, sys
            from pathlib import Path
            from trainer.frontend.reporting.error_log import ErrorLog
            if sys.platform == 'win32':
                # Suppress Windows crash-report dialogs in this disposable child.
                ctypes.CDLL('ucrtbase')._set_abort_behavior(0, 3)
            else:
                import resource
                resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            faulthandler.disable()
            with ErrorLog(Path(sys.argv[1])):
                os.write(2, b'VK_ERROR_DEVICE_LOST before abort\\n')
                os.abort()
            """), str(path)], capture_output=True, check=False, timeout=30)
    assert child.returncode != 0
    text = path.read_text(encoding="utf-8")
    assert "VK_ERROR_DEVICE_LOST before abort" in text
    assert "Fatal Python error" in text
    assert '<module>' in text
