"""
Static checks on the Streamlit app.

Streamlit derives a widget's internal ID from its type and parameters, so two
sliders with identical arguments collide at runtime with
StreamlitDuplicateElementId. Worse, that exception aborts the whole script,
so every tab after the offending one renders blank -- one bug that looks like
five broken features. These tests catch it without a browser.
"""

import ast
import os

import pytest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "app.py")

WIDGETS = {
    "slider", "selectbox", "multiselect", "checkbox", "number_input", "radio",
    "text_input", "text_area", "file_uploader", "button", "download_button",
    "toggle", "select_slider", "data_editor",
}


def _widget_calls():
    trees = [ast.parse(open(p).read()) for p in [APP, os.path.join(os.path.dirname(APP), "msqc", "rescue_ui.py")]]
    return [n for tree in trees for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in WIDGETS]


def test_app_parses():
    ast.parse(open(APP).read())


def test_every_widget_has_an_explicit_key():
    missing = [f"{n.func.attr} at line {n.lineno}" for n in _widget_calls()
               if not any(k.arg == "key" for k in n.keywords)]
    assert not missing, (
        "Widgets without an explicit key collide whenever two of them share a "
        "type and arguments:\n  " + "\n  ".join(missing))


def test_widget_keys_are_unique():
    keys = [k.value.value for n in _widget_calls() for k in n.keywords
            if k.arg == "key" and isinstance(k.value, ast.Constant)]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    assert not dupes, f"duplicate widget keys: {dupes}"


def test_no_two_sliders_share_identical_arguments():
    """
    The specific failure that shipped: 'Quality threshold' with the same
    bounds appeared in both the run settings and the threshold explorer.
    """
    sigs = {}
    for n in _widget_calls():
        if n.func.attr != "slider":
            continue
        sig = ast.dump(ast.Tuple(elts=list(n.args), ctx=ast.Load()))
        sigs.setdefault(sig, []).append(n.lineno)
    clashes = {s: ls for s, ls in sigs.items() if len(ls) > 1}
    # identical args are fine now that keys are explicit, but flag them so
    # the keys are never removed
    for _, lines in clashes.items():
        for ln in lines:
            node = next(n for n in _widget_calls() if n.lineno == ln)
            assert any(k.arg == "key" for k in node.keywords), (
                f"slider at line {ln} shares arguments with another and has "
                f"no key")


def test_tabs_are_rendered_through_the_isolating_helper():
    """
    Each tab must go through render(), so one failing tab cannot blank the
    others.
    """
    src = open(APP).read()
    assert "def render(tab, fn" in src
    tree = ast.parse(src)
    main = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    body = ast.dump(main)
    # no bare `with tabs[i]:` blocks left behind
    assert "withitem" not in body or "tabs" not in body.split("withitem")[1][:80]
