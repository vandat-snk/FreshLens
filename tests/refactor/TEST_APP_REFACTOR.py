"""G4 regression contract for FreshLens modular Streamlit UI (Nhi + TV3).

This checks structure and session-state logic without starting a browser or loading
CNN weights. Run TV3 AppTest and an interactive UI smoke test separately.
"""
from __future__ import annotations

import ast
from pathlib import Path

from freshlens_ai.inference import (
    clear_result,
    image_identity,
    result_for_image,
    store_result,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]
APP = PROJECT_DIR / "APP_CNN_V2.py"
DIAGNOSIS = PROJECT_DIR / "ui" / "diagnosis.py"
OTHER_PAGES = [
    PROJECT_DIR / "ui" / "evaluation.py",
    PROJECT_DIR / "ui" / "explanation.py",
    PROJECT_DIR / "ui" / "about.py",
]
RUNNER = PROJECT_DIR / "RUN_APP_V2.cmd"


def _parse(path: Path) -> ast.Module:
    if not path.is_file():
        raise AssertionError(f"Missing UI module: {path}")
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(path))
    compile(tree, str(path), "exec")
    return tree


def _calls(tree: ast.AST, dotted: str):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == dotted:
            yield node


def _require_calls(tree: ast.AST, names: tuple[str, ...], path: Path) -> None:
    for name in names:
        if not any(_calls(tree, name)):
            raise AssertionError(f"{path.name}: missing call {name}(...)")


def _require_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    found = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(found) == 1, f"Function {name} must exist exactly once"
    return found[0]


def _check_no_legacy_imports(path: Path, tree: ast.AST) -> None:
    forbidden = {"cnn_data", "cnn_model", "cnn_metrics", "open_set"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in forbidden, f"Legacy import in {path}"
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            assert (node.module or "").split(".")[0] not in forbidden, f"Legacy import in {path}"
        elif isinstance(node, ast.Call) and ast.unparse(node.func) == "sys.path.insert":
            raise AssertionError(f"Legacy sys.path.insert in {path}")


def _verify_result_identity() -> None:
    state = {}
    img_a, img_b = b"image-a", b"image-b"
    outcome = {"fruit": "apple", "condition": "fresh"}
    assert image_identity(img_a) != image_identity(img_b)
    assert result_for_image(state, img_a) is None
    store_result(state, img_a, outcome)
    assert result_for_image(state, img_a) == outcome
    assert result_for_image(state, img_b) is None
    clear_result(state)
    assert result_for_image(state, img_a) is None
    print("[OK] Session state binds prediction to exact input bytes")


def _verify_app(tree: ast.Module) -> None:
    _require_calls(tree, ("st.set_page_config", "render_diagnosis_page",
                          "render_evaluation_page", "render_explanation_page",
                          "render_about_page", "FreshLensPredictor",
                          "load_quality_config", "quality_config_hash",
                          "clear_result", "st.sidebar.checkbox"), APP)
    entry = next(_calls(tree, "render_diagnosis_page"))
    kwargs = {k.arg: k.value for k in entry.keywords}
    assert isinstance(kwargs.get("predictor"), ast.Name), "Pass runtime predictor to diagnosis"
    assert isinstance(kwargs.get("check_quality"), ast.Name), "Pass quality policy to diagnosis"
    assert kwargs["check_quality"].id == "quality_enabled", "Quality policy must follow checkbox"
    assert any(isinstance(x, ast.Constant) and x.value == "FRESHLENS_QUALITY_CONFIG" for x in ast.walk(tree)), \
        "APP must honor custom quality config environment variable"
    for module in ["ui.about", "ui.diagnosis", "ui.evaluation", "ui.explanation"]:
        assert any(isinstance(n, ast.ImportFrom) and n.module == module for n in tree.body), \
            f"Missing modular UI import: {module}"
    print("[OK] APP uses four modular pages and TV3 quality policy")


def _verify_diagnosis(tree: ast.Module) -> None:
    _require_calls(tree, ("st.file_uploader", "st.camera_input", "result_for_image",
                          "store_result", "clear_result", "predictor.predict"), DIAGNOSIS)
    render = _require_function(tree, "render_diagnosis_page")
    renderer_args = [x.arg for x in render.args.args]
    assert "check_quality" in renderer_args, "UI renderer must receive quality policy"
    result_panel = _require_function(tree, "_render_result_panel")
    assert "check_quality" in [x.arg for x in result_panel.args.args]
    analyze = _require_function(tree, "_analyze_current_image")
    assert "check_quality" in [x.arg for x in analyze.args.args]
    predict_calls = list(_calls(analyze, "predictor.predict"))
    assert len(predict_calls) == 1, "Expect exactly one predictor call in analyze handler"
    kwargs = {k.arg: k.value for k in predict_calls[0].keywords}
    assert isinstance(kwargs.get("check_quality"), ast.Name) and kwargs["check_quality"].id == "check_quality", \
        "Diagnosis must pass the checkbox policy to runtime.predict()"
    assert any(_calls(result_panel, "_analyze_current_image")), "Analysis button must use handler"
    assert any(_calls(render, "_render_result_panel")), "Diagnosis page must show results"
    print("[OK] Diagnosis connects upload/camera, analysis and quality policy")


def main() -> int:
    _verify_result_identity()
    trees = {path: _parse(path) for path in [APP, DIAGNOSIS, *OTHER_PAGES]}
    for path, tree in trees.items():
        _check_no_legacy_imports(path, tree)
    print("[OK] All five UI modules compile; no legacy imports/path hacks")
    _verify_app(trees[APP])
    _verify_diagnosis(trees[DIAGNOSIS])
    assert RUNNER.is_file(), "Missing RUN_APP_V2.cmd"
    runner = RUNNER.read_text(encoding="utf-8-sig")
    assert "streamlit run APP_CNN_V2.py" in runner, "Runner targets wrong UI script"
    print("[OK] Windows launcher runs the modular app")
    print("[PASS] Stage G4 modular UI regression contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
