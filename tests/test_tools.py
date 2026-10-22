import json

import nbformat
import pytest


def text_of(blocks):
    return "\n".join(b["text"] for b in blocks if b["type"] == "text")


def test_run_python_records_notebook_and_figures(toolbox):
    blocks, is_error = toolbox.dispatch(
        "run_python",
        {
            "purpose": "plot",
            "code": "import matplotlib.pyplot as plt\n"
            "plt.hist([1, 2, 2, 3])\nplt.show()\nprint('done')",
        },
    )
    assert not is_error
    assert "done" in text_of(blocks)
    assert any(b["type"] == "image" for b in blocks)
    assert (toolbox.figures / "fig_01.png").exists()
    nb = nbformat.read(toolbox.recorder.path, as_version=4)
    assert any(c.cell_type == "code" and "plt.hist" in c.source for c in nb.cells)


def test_run_python_error_is_flagged(toolbox):
    blocks, is_error = toolbox.dispatch("run_python", {"purpose": "fail", "code": "undefined_name"})
    assert is_error
    assert "NameError" in text_of(blocks)


def test_paths_cannot_escape_workdir(toolbox):
    blocks, is_error = toolbox.dispatch("read_file", {"path": "../../etc/passwd"})
    assert is_error
    assert "outside the working directory" in text_of(blocks)


def test_write_then_read_and_list(toolbox):
    toolbox.dispatch("write_file", {"path": "outputs/notes.txt", "content": "hello"})
    blocks, _ = toolbox.dispatch("read_file", {"path": "outputs/notes.txt"})
    assert text_of(blocks) == "hello"
    blocks, _ = toolbox.dispatch("list_files", {"path": "outputs"})
    assert "outputs/notes.txt" in text_of(blocks)


def test_log_experiment_builds_leaderboard(toolbox):
    toolbox.dispatch(
        "log_experiment", {"name": "baseline", "model": "Dummy", "metrics": {"auc": 0.5}}
    )
    blocks, _ = toolbox.dispatch(
        "log_experiment", {"name": "lgbm", "model": "LGBMClassifier", "metrics": {"auc": 0.84}}
    )
    board = text_of(blocks)
    assert "| baseline | Dummy | 0.5000 |" in board
    assert "| lgbm | LGBMClassifier | 0.8400 |" in board
    lines = toolbox.experiments_path.read_text().splitlines()
    assert json.loads(lines[1])["metrics"]["auc"] == 0.84


def test_checkpoint_in_auto_mode_continues(toolbox):
    blocks, _ = toolbox.dispatch("checkpoint", {"summary": "EDA done", "question": "Go on?"})
    assert "Auto mode" in text_of(blocks)


def test_checkpoint_asks_user(toolbox):
    toolbox.auto = False
    toolbox.ask_user = lambda prompt: "drop the id column"
    blocks, _ = toolbox.dispatch("checkpoint", {"summary": "s", "question": "q"})
    assert "drop the id column" in text_of(blocks)


def test_save_memory_and_finish(toolbox):
    toolbox.dispatch("save_memory", {"lesson": "TotalCharges is   a string with blanks"})
    assert "TotalCharges is a string with blanks" in toolbox.memory_path.read_text()
    toolbox.dispatch("finish", {"report": "# Report"})
    assert toolbox.finished
    assert (toolbox.outputs / "report.md").read_text() == "# Report"


@pytest.mark.parametrize("spec", ["xgboost", "catboost==1.2", "a; rm -rf /"])
def test_install_package_rejects_shell_injection(toolbox, spec, monkeypatch):
    monkeypatch.setattr(
        toolbox.kernel, "execute", lambda code: type("R", (), {"text": "ok", "ok": True})()
    )
    _, is_error = toolbox.dispatch("install_package", {"package": spec})
    assert is_error == (";" in spec)


def test_unknown_tool(toolbox):
    _, is_error = toolbox.dispatch("nope", {})
    assert is_error
