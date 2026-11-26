# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Every interactive widget in the panel must have a UNIQUE ImGui id.

ImGui identifies a widget by its LABEL (``window->GetID(label)``) and tracks
exactly ONE active id. Our controls put their caption on the line above (sgl's
hard-coded item width pushes a real label off the panel), so they were all
constructed with the same empty label — which meant every drag, combo and text
field in the window shared one id. The visible symptom was that typing into a
text field committed and lost focus after a SINGLE character, because the first
same-id widget drawn each frame reclaimed the active id; drags could also scrub
a field other than the one under the cursor.

The fix is a hidden ``##`` suffix (``Panel._widget_id``), which ImGui does not
display but does hash into the id. This test builds the REAL panel — the same
section builders the viewer uses, so it covers every widget rather than a
hand-picked sample — and asserts the ids are unique and stable.

Needs a device (``spy.ui.Context`` is a real ImGui context) but no window.
"""

from __future__ import annotations

import pytest

from trainer.backend.compute.device import create_device
from trainer.frontend.viewer.session import Session
from trainer.frontend.viewer.ui.panel import Panel

BACKENDS = ["vulkan", "d3d12", "metal"]

# Widget kinds whose identity ImGui derives from the label. ``Text``/``Group``
# are excluded: text is not interactive, and group titles are already distinct
# prose.
INTERACTIVE = ("Button", "CheckBox", "ComboBox", "InputText", "ListBox",
               "SliderFloat", "SliderInt", "DragFloat", "DragInt",
               "InputFloat", "InputInt")


def _build_real_panel(device) -> Panel:
    """The viewer's own sections, rendered into a real (device-backed) Panel."""
    from trainer.frontend.viewer.ui import sections as sections_module

    panel = Panel(device, title="Viewer", scale=1.0)
    session = Session()
    view = session.view()

    class _AnyAction:
        """Section builders only ever store the callables; nothing is invoked."""
        def __getattr__(self, name):
            return lambda *a, **k: None

    actions = _AnyAction()
    sections_module.header(panel, view)
    sections_module.dataset_section(panel, view, actions)
    sections_module.training_section(panel, view, actions, session.setup)
    sections_module.result_section(panel, view, actions)
    sections_module.configuration_section(
        panel, view, session.setup,
        show_all=lambda: True, toggle_all=lambda: None)
    return panel


def _interactive_labels(widget, found: list | None = None) -> list:
    found = [] if found is None else found
    for child in widget.children:
        if type(child).__name__ in INTERACTIVE:
            found.append((type(child).__name__, child.label))
        _interactive_labels(child, found)
    return found


@pytest.mark.parametrize("backend", BACKENDS)
def test_every_interactive_widget_has_a_unique_id(backend: str) -> None:
    try:
        device = create_device(backend=backend)
    except Exception as exc:                                   # noqa: BLE001
        pytest.skip(f"backend {backend!r} unavailable: {exc}")

    try:
        panel = _build_real_panel(device)
    except (AttributeError, TypeError) as exc:
        pytest.skip(f"section builder signature changed: {exc}")

    panel.refresh()          # dynamic labels resolve here; ids must survive it
    labels = _interactive_labels(panel.window)
    assert labels, "no interactive widgets found — the walk or the build broke"

    ids = [label.split("##")[-1] for _, label in labels]
    duplicates = {i for i in ids if ids.count(i) > 1}
    assert not duplicates, (
        f"{len(duplicates)} duplicated ImGui id(s) across {len(ids)} widgets: "
        f"{sorted(duplicates)[:5]} — a text field sharing an id loses focus "
        f"after one keystroke")

    # Every one must actually carry a hidden suffix, not merely happen to differ
    # (two buttons reading "Export" would collide the moment both are visible).
    missing = [(kind, label) for kind, label in labels if "##" not in label]
    assert not missing, f"widgets without a hidden id suffix: {missing[:5]}"


@pytest.mark.parametrize("backend", BACKENDS)
def test_ids_are_stable_across_frames(backend: str) -> None:
    """A ``label`` binding rewrites the label every frame. If it dropped the
    suffix the id would CHANGE mid-interaction, which is worse than a collision:
    ImGui would discard the edit in progress rather than merely misroute it."""
    try:
        device = create_device(backend=backend)
    except Exception as exc:                                   # noqa: BLE001
        pytest.skip(f"backend {backend!r} unavailable: {exc}")

    panel = Panel(device, title="Viewer", scale=1.0)
    panel.button(lambda: "Train", lambda: None)
    panel.button(lambda: "Train", lambda: None)     # same TEXT, must differ in id
    panel.input_text("name.ply", lambda s: None)
    panel.drag_int("iters", 3000, lambda v: None, minimum=1, maximum=9)

    panel.refresh()
    first = [label for _, label in _interactive_labels(panel.window)]
    panel.refresh()
    panel.refresh()
    assert [label for _, label in _interactive_labels(panel.window)] == first

    ids = [label.split("##")[-1] for label in first]
    assert len(set(ids)) == len(ids), f"ids collided: {first}"
    # The visible half is untouched — the suffix must not leak into the caption.
    assert [label.split("##")[0] for label in first][:2] == ["Train", "Train"]
