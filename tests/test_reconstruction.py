# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""The reconstruction subsystem: tool resolution, install safety, the pipeline.

Nothing here downloads anything or needs colmap installed — the binaries are
faked and the subprocess calls are recorded, so what is under test is our own
logic: which build wins, what a failed CUDA install falls back to, which colmap
commands run in which order, and the glomap database trim.
"""
from __future__ import annotations

import os
import sqlite3
import zipfile
from pathlib import Path

import pytest

from trainer.frontend.reconstruction import pipeline as pl
from trainer.frontend.reconstruction import tools as tl
from trainer.frontend.reconstruction.tools import ReconTools


def _fake_install(root: Path, tool: str = "colmap") -> Path:
    """A tool laid out the way its zip extracts: exe in a bin/ dir beside DLLs."""
    exe = root / "bin" / (f"{tool}.exe" if os.name == "nt" else tool)
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_text("")
    return exe


# --------------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------------

def test_an_explicit_path_beats_everything(tmp_path) -> None:
    exe = _fake_install(tmp_path / "elsewhere")
    tools = ReconTools(colmap_bin=str(exe), tools_dir=tmp_path / "t")
    assert tools.resolve("colmap") == str(exe)
    assert tools.is_ready()


def test_an_env_var_is_used_when_no_flag_was_given(tmp_path, monkeypatch) -> None:
    exe = _fake_install(tmp_path / "envcopy", "glomap")
    monkeypatch.setenv("GLOMAP_PATH", str(exe))
    assert ReconTools(tools_dir=tmp_path / "t").resolve("glomap") == str(exe)


def test_the_preferred_build_wins_but_the_other_is_still_used(tmp_path) -> None:
    """prefer_cuda ORDERS the search; it never drops a tool that is installed."""
    _fake_install(tmp_path / "colmap-nocuda")
    tools = ReconTools(recon_cuda=True, tools_dir=tmp_path)
    assert "nocuda" in tools.resolve("colmap")        # only one there
    _fake_install(tmp_path / "colmap-cuda")
    assert "colmap-cuda" in tools.resolve("colmap")   # both there -> preferred
    assert "nocuda" in ReconTools(recon_cuda=False, tools_dir=tmp_path).resolve("colmap")


def test_nothing_installed_resolves_to_none_not_a_crash(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(tl.shutil, "which", lambda _n: None)
    tools = ReconTools(tools_dir=tmp_path / "empty")
    assert tools.resolve("colmap") is None
    assert not tools.is_ready()


def test_readiness_is_colmap_only_glomap_is_optional(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(tl.shutil, "which", lambda _n: None)
    _fake_install(tmp_path / "colmap-cuda")
    tools = ReconTools(tools_dir=tmp_path)
    assert tools.is_ready() and tools.resolve("glomap") is None
    assert "colmap incremental" in tools.summary()


# --------------------------------------------------------------------------
# install
# --------------------------------------------------------------------------

def test_release_urls_are_https_github_release_assets() -> None:
    for tool in ("colmap", "glomap"):
        for cuda in (True, False):
            url = tl._url(tool, cuda)
            assert url.startswith(f"https://github.com/colmap/{tool}/releases/latest/")
            assert url.endswith(f"{tool}-x64-windows-{'cuda' if cuda else 'nocuda'}.zip")


def test_a_non_github_url_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match="refusing url"):
        tl._download("http://github.com/evil.zip", tmp_path / "x", lambda _m: None)
    with pytest.raises(ValueError, match="refusing url"):
        tl._download("https://evil.example/x.zip", tmp_path / "x", lambda _m: None)


def test_a_zip_that_escapes_its_destination_is_rejected(tmp_path) -> None:
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("../escaped.txt", "nope")
    with pytest.raises(ValueError, match="escapes"):
        tl._unzip(bad, tmp_path / "dest")
    assert not (tmp_path / "escaped.txt").exists()


def test_a_cuda_build_that_cannot_start_falls_back_to_the_no_cuda_zip(tmp_path,
                                                                     monkeypatch) -> None:
    """A CUDA binary that cannot load its DLLs must be caught HERE, not halfway
    through a reconstruction."""
    installed: list[str] = []

    def fake_download(url: str, dest: Path, _progress) -> bool:
        installed.append(url)
        with zipfile.ZipFile(dest, "w") as z:
            z.writestr("bin/colmap.exe", "")
        return True

    monkeypatch.setattr(tl, "_WINDOWS", True)
    monkeypatch.setattr(tl, "_download", fake_download)
    # STATUS_DLL_NOT_FOUND from the cuda build, fine from the no-cuda one.
    monkeypatch.setattr(tl, "_starts", lambda exe, _p: "nocuda" in str(exe))
    tools = ReconTools(tools_dir=tmp_path)
    result = tools._install("colmap", True, lambda _m: None)
    assert result is not None and "nocuda" in result
    assert [("cuda" if "nocuda" not in u else "nocuda") for u in installed] == \
           ["cuda", "nocuda"]
    assert not (tmp_path / "colmap-cuda").exists()   # the bad install is cleaned up


def test_a_dll_load_exit_code_means_the_binary_does_not_start(monkeypatch) -> None:
    class _Result:
        returncode = -1073741515        # STATUS_DLL_NOT_FOUND

    monkeypatch.setattr(tl.subprocess, "run", lambda *_a, **_k: _Result())
    assert not tl._starts(Path("colmap.exe"), lambda _m: None)


def test_setup_does_not_redownload_what_is_already_there(tmp_path, monkeypatch) -> None:
    _fake_install(tmp_path / "colmap-cuda")
    _fake_install(tmp_path / "glomap-cuda", "glomap")
    monkeypatch.setattr(tl, "_download", lambda *_a, **_k: pytest.fail("downloaded"))
    assert ReconTools(tools_dir=tmp_path).setup() is True


def test_a_failed_glomap_install_still_leaves_the_tools_ready(tmp_path,
                                                              monkeypatch) -> None:
    """glomap only replaces colmap's mapper; losing it is a slower run, not a
    failed setup."""
    _fake_install(tmp_path / "colmap-cuda")
    monkeypatch.setattr(tl.shutil, "which", lambda _n: None)
    monkeypatch.setattr(tl, "_download", lambda *_a, **_k: False)
    lines: list[str] = []
    assert ReconTools(tools_dir=tmp_path).setup(lines.append) is True
    assert any("tools READY" in line for line in lines)


def test_off_windows_setup_says_what_to_install_instead_of_failing_silently(
        tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(tl, "_WINDOWS", False)
    monkeypatch.setattr(tl.shutil, "which", lambda _n: None)
    lines: list[str] = []
    assert ReconTools(tools_dir=tmp_path).setup(lines.append) is False
    assert any("conda install" in line for line in lines)
    assert any("SETUP FAILED" in line for line in lines)


# --------------------------------------------------------------------------
# dataset classification
# --------------------------------------------------------------------------

def test_a_folder_of_images_is_images_and_a_sparse_model_is_colmap(tmp_path) -> None:
    assert pl.classify_dataset(tmp_path / "nope") == "unknown"
    (tmp_path / "images").mkdir()
    assert pl.classify_dataset(tmp_path) == "unknown"      # empty
    (tmp_path / "images" / "a.JPG").write_bytes(b"")
    assert pl.classify_dataset(tmp_path) == "images"       # case-insensitive
    sparse = tmp_path / "sparse" / "0"
    sparse.mkdir(parents=True)
    for name in ("cameras.txt", "images.txt", "points3D.txt"):
        (sparse / name).write_text("")
    assert pl.classify_dataset(tmp_path) == "colmap"


# --------------------------------------------------------------------------
# the pipeline: which commands run, in which order
# --------------------------------------------------------------------------

def _recorded(monkeypatch, *, rcs: dict[str, int] | None = None) -> list[list[str]]:
    """Record every tool invocation; fake up the outputs each step is judged on."""
    calls: list[list[str]] = []
    rcs = rcs or {}

    def fake_run(cmd: list[str], _progress) -> int:
        calls.append(cmd)
        verb = cmd[1]
        rc = rcs.get(verb, 0)
        if rc == 0 and verb == "mapper":
            model = Path(cmd[cmd.index("--output_path") + 1]) / "0"
            model.mkdir(parents=True, exist_ok=True)
            (model / "cameras.bin").write_bytes(b"")
        if rc == 0 and verb == "image_undistorter":
            out = Path(cmd[cmd.index("--output_path") + 1])
            (out / "sparse").mkdir(parents=True, exist_ok=True)
            (out / "sparse" / "cameras.bin").write_bytes(b"")
            (out / "images").mkdir(parents=True, exist_ok=True)
            (out / "images" / "a.jpg").write_bytes(b"")
        return rc

    monkeypatch.setattr(pl, "_run", fake_run)
    monkeypatch.setattr(pl, "gpu_flag", lambda _exe, verb:
                        "--SiftExtraction.use_gpu" if "extractor" in verb
                        else "--SiftMatching.use_gpu")
    return calls


def _dataset(tmp_path) -> Path:
    images = tmp_path / "scene" / "images"
    images.mkdir(parents=True, exist_ok=True)
    (images / "a.jpg").write_bytes(b"")
    return tmp_path / "scene"


def test_the_four_colmap_steps_run_in_order_and_produce_a_loadable_dataset(
        tmp_path, monkeypatch) -> None:
    calls = _recorded(monkeypatch)
    tools = ReconTools(colmap_bin=str(_fake_install(tmp_path / "c")),
                       tools_dir=tmp_path / "t")
    out = tmp_path / "out"
    assert pl.reconstruct(tools, _dataset(tmp_path), out) == out
    assert [c[1] for c in calls] == ["feature_extractor", "exhaustive_matcher",
                                     "mapper", "image_undistorter"]
    assert pl.classify_dataset(out) == "colmap"
    assert (out / "images" / "a.jpg").exists()
    assert not (out / "_recon_work").exists()      # the scratch dir is cleaned up


def test_the_chosen_matcher_is_the_one_that_runs(tmp_path, monkeypatch) -> None:
    calls = _recorded(monkeypatch)
    tools = ReconTools(colmap_bin=str(_fake_install(tmp_path / "c")),
                       tools_dir=tmp_path / "t")
    pl.reconstruct(tools, _dataset(tmp_path), tmp_path / "out", matcher="sequential")
    assert "sequential_matcher" in [c[1] for c in calls]


def test_the_gpu_is_asked_for_first_and_the_cpu_is_the_fallback(tmp_path,
                                                                monkeypatch) -> None:
    calls = _recorded(monkeypatch, rcs={"feature_extractor": 1})
    tools = ReconTools(colmap_bin=str(_fake_install(tmp_path / "c")),
                       tools_dir=tmp_path / "t")
    assert pl.reconstruct(tools, _dataset(tmp_path), tmp_path / "out") is None
    extractions = [c for c in calls if c[1] == "feature_extractor"]
    assert [c[-1] for c in extractions] == ["1", "0"]      # GPU, then CPU
    assert all(c[-2] == "--SiftExtraction.use_gpu" for c in extractions)


def test_a_cpu_only_run_never_asks_for_the_gpu(tmp_path, monkeypatch) -> None:
    calls = _recorded(monkeypatch)
    tools = ReconTools(colmap_bin=str(_fake_install(tmp_path / "c")),
                       recon_cuda=False, tools_dir=tmp_path / "t")
    pl.reconstruct(tools, _dataset(tmp_path), tmp_path / "out")
    assert [c[-1] for c in calls if c[1] == "feature_extractor"] == ["0"]


def test_a_build_without_a_use_gpu_option_runs_the_verb_bare(tmp_path,
                                                             monkeypatch) -> None:
    """An unknown option makes colmap exit non-zero, so a build that does not
    advertise use_gpu is called without it rather than guessed at."""
    calls = _recorded(monkeypatch)
    monkeypatch.setattr(pl, "gpu_flag", lambda _exe, _verb: None)
    tools = ReconTools(colmap_bin=str(_fake_install(tmp_path / "c")),
                       tools_dir=tmp_path / "t")
    pl.reconstruct(tools, _dataset(tmp_path), tmp_path / "out")
    extractions = [c for c in calls if c[1] == "feature_extractor"]
    assert len(extractions) == 1
    assert not any(arg.endswith("use_gpu") for arg in extractions[0])


def test_glomap_maps_when_present_and_colmap_takes_over_when_it_fails(
        tmp_path, monkeypatch) -> None:
    calls = _recorded(monkeypatch, rcs={"mapper": 0})
    glomap = _fake_install(tmp_path / "g", "glomap")
    tools = ReconTools(colmap_bin=str(_fake_install(tmp_path / "c")),
                       glomap_bin=str(glomap), tools_dir=tmp_path / "t")
    pl.reconstruct(tools, _dataset(tmp_path), tmp_path / "out1")
    mappers = [c[0] for c in calls if c[1] == "mapper"]
    assert mappers == [str(glomap)]                     # glomap only

    calls.clear()
    monkeypatch.setattr(pl, "_run", _failing_glomap_mapper(calls, tmp_path))
    pl.reconstruct(tools, _dataset(tmp_path), tmp_path / "out2")
    mappers = [c[0] for c in calls if c[1] == "mapper"]
    assert len(mappers) == 2 and mappers[1].endswith(("colmap", "colmap.exe"))


def _failing_glomap_mapper(calls: list[list[str]], _tmp_path):
    def fake_run(cmd: list[str], _progress) -> int:
        calls.append(cmd)
        if cmd[1] == "mapper" and "glomap" in cmd[0]:
            return 1
        if cmd[1] == "mapper":
            model = Path(cmd[cmd.index("--output_path") + 1]) / "0"
            model.mkdir(parents=True, exist_ok=True)
            (model / "cameras.bin").write_bytes(b"")
        if cmd[1] == "image_undistorter":
            out = Path(cmd[cmd.index("--output_path") + 1])
            (out / "sparse").mkdir(parents=True, exist_ok=True)
            (out / "sparse" / "cameras.bin").write_bytes(b"")
            (out / "images").mkdir(parents=True, exist_ok=True)
        return 0
    return fake_run


def test_an_already_reconstructed_output_is_not_redone(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pl, "_run", lambda *_a: pytest.fail("re-ran"))
    out = tmp_path / "out"
    sparse = out / "sparse" / "0"
    sparse.mkdir(parents=True)
    for name in ("cameras.txt", "images.txt", "points3D.txt"):
        (sparse / name).write_text("")
    tools = ReconTools(colmap_bin=str(_fake_install(tmp_path / "c")),
                       tools_dir=tmp_path / "t")
    assert pl.reconstruct(tools, _dataset(tmp_path), out) == out


def test_without_colmap_reconstruction_says_so_instead_of_starting(tmp_path,
                                                                   monkeypatch) -> None:
    monkeypatch.setattr(tl.shutil, "which", lambda _n: None)
    lines: list[str] = []
    tools = ReconTools(tools_dir=tmp_path / "empty")
    assert pl.reconstruct(tools, _dataset(tmp_path), tmp_path / "out",
                          progress=lines.append) is None
    assert any("Set up recon tools" in line for line in lines)


# --------------------------------------------------------------------------
# the glomap database copy
# --------------------------------------------------------------------------

def _colmap4_database(path: Path) -> None:
    """The tables (and the schema quirks) that made glomap refuse the real file."""
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE cameras (camera_id INTEGER PRIMARY KEY AUTOINCREMENT, model INTEGER);
        CREATE TABLE images (image_id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT);
        CREATE TABLE keypoints (image_id INTEGER PRIMARY KEY, data BLOB);
        CREATE TABLE descriptors (image_id INTEGER PRIMARY KEY, data BLOB);
        CREATE TABLE matches (pair_id INTEGER PRIMARY KEY, data BLOB);
        CREATE TABLE two_view_geometries (pair_id INTEGER PRIMARY KEY, data BLOB);
        CREATE TABLE pose_priors (image_id INTEGER PRIMARY KEY, position BLOB);
        CREATE INDEX idx_name ON images (name);
    """)
    con.execute("INSERT INTO cameras VALUES (1, 1)")
    con.execute("INSERT INTO images VALUES (1, 'a.jpg')")
    con.execute("INSERT INTO keypoints VALUES (1, x'00')")
    con.execute("INSERT INTO descriptors VALUES (1, x'ff')")
    con.execute("INSERT INTO matches VALUES (1, x'00')")
    con.execute("INSERT INTO pose_priors VALUES (1, x'00')")
    con.execute("PRAGMA user_version = 42")
    con.commit()
    con.close()


def _tables(path: Path) -> dict[str, str]:
    con = sqlite3.connect(path)
    try:
        return {n: s for n, s in con.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table'")}
    finally:
        con.close()


def test_the_copy_drops_only_the_tables_glomap_chokes_on(tmp_path) -> None:
    src = tmp_path / "database.db"
    _colmap4_database(src)
    copy = pl._glomap_database(src, tmp_path, lambda _m: None)
    assert copy != src
    tables = _tables(copy)
    assert "descriptors" not in tables and "pose_priors" not in tables
    # Everything the mapper reads survives, with the schema colmap wrote.
    for name in ("cameras", "images", "keypoints", "matches", "two_view_geometries"):
        assert tables[name] == _tables(src)[name]
    assert _tables(src).keys() >= {"descriptors", "pose_priors"}   # source untouched


def test_the_copy_keeps_the_rows_and_the_schema_version(tmp_path) -> None:
    src = tmp_path / "database.db"
    _colmap4_database(src)
    copy = pl._glomap_database(src, tmp_path, lambda _m: None)
    con = sqlite3.connect(copy)
    try:
        assert con.execute("SELECT name FROM images").fetchall() == [("a.jpg",)]
        assert con.execute("SELECT data FROM keypoints").fetchone()[0] == b"\x00"
        assert con.execute("PRAGMA user_version").fetchone()[0] == 42
    finally:
        con.close()


def test_a_database_that_cannot_be_copied_hands_over_the_original(tmp_path) -> None:
    src = tmp_path / "database.db"
    src.write_text("not a database")
    lines: list[str] = []
    assert pl._glomap_database(src, tmp_path, lines.append) == src
    assert any("using the original" in line for line in lines)
