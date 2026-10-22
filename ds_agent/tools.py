"""Tools the model can call, and the code that runs them."""

from __future__ import annotations

import base64
import json
import re
from collections.abc import Callable
from datetime import date
from pathlib import Path

from .kernel import PythonKernel
from .notebook import NotebookRecorder

MAX_TEXT = 6000
MAX_IMAGES_PER_CALL = 3
PACKAGE_RE = re.compile(r"^[A-Za-z0-9_.\-\[\]=<>,]+$")

TOOL_SPECS = [
    {
        "name": "run_python",
        "description": (
            "Run Python code in a persistent Jupyter kernel. Variables, imports and loaded data "
            "stay in memory between calls. The working directory contains data/ (inputs) and "
            "outputs/ (save models, figures and files here). Printed output, the value of the "
            "last expression and matplotlib figures are returned to you. Keep printed output "
            "short: print shapes, heads, summaries, not whole tables."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "purpose": {
                    "type": "string",
                    "description": "One sentence on what this step does (a notebook heading).",
                },
                "code": {"type": "string", "description": "Python code to execute."},
            },
            "required": ["purpose", "code"],
        },
    },
    {
        "name": "list_files",
        "description": "List files under a directory of the working directory, with sizes.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "default": "."}},
        },
    },
    {
        "name": "read_file",
        "description": "Read the start of a text file (CSV header, README, data dictionary).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_chars": {"type": "integer", "default": 4000},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write a text file inside the working directory, e.g. outputs/train.py.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
    {
        "name": "install_package",
        "description": "pip install a package into the kernel, e.g. 'xgboost' or 'catboost==1.2'.",
        "input_schema": {
            "type": "object",
            "properties": {"package": {"type": "string"}},
            "required": ["package"],
        },
    },
    {
        "name": "log_experiment",
        "description": (
            "Record one trained model with its validation metrics. Call this for every model you "
            "evaluate, including the baseline. Returns the leaderboard of all experiments so far."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Short id, e.g. 'lgbm_tuned'."},
                "model": {"type": "string", "description": "Model class or description."},
                "features": {"type": "string", "description": "Feature set used."},
                "params": {"type": "object"},
                "metrics": {
                    "type": "object",
                    "description": "Validation metrics, e.g. {'roc_auc': 0.84, 'cv_std': 0.01}.",
                },
                "notes": {"type": "string"},
            },
            "required": ["name", "model", "metrics"],
        },
    },
    {
        "name": "checkpoint",
        "description": (
            "Show the user a short summary of progress and ask whether to continue or change "
            "direction. Use after EDA and after the baseline. Returns the user's answer."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "question": {"type": "string"},
            },
            "required": ["summary", "question"],
        },
    },
    {
        "name": "save_memory",
        "description": (
            "Save one reusable lesson for future tasks on other datasets (a pitfall, a technique "
            "that worked). Not task results."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"lesson": {"type": "string"}},
            "required": ["lesson"],
        },
    },
    {
        "name": "finish",
        "description": (
            "End the task. Pass the final report in Markdown: task, data, approach, leaderboard, "
            "chosen model and why, its metrics, key drivers, limitations, next steps. Reference "
            "figures as outputs/figures/<file>.png."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"report": {"type": "string"}},
            "required": ["report"],
        },
    },
]


def truncate(text: str, limit: int = MAX_TEXT) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n... [{len(text) - limit} characters truncated] ...\n{text[-half:]}"


def text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def image_block(png_b64: str) -> dict:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": png_b64},
    }


class Toolbox:
    def __init__(
        self,
        workdir: str | Path,
        kernel: PythonKernel,
        recorder: NotebookRecorder,
        memory_path: str | Path,
        ask_user: Callable[[str], str] | None = None,
        auto: bool = False,
    ):
        self.workdir = Path(workdir).resolve()
        self.kernel = kernel
        self.recorder = recorder
        self.memory_path = Path(memory_path)
        self.ask_user = ask_user or input
        self.auto = auto
        self.outputs = self.workdir / "outputs"
        self.figures = self.outputs / "figures"
        self.figures.mkdir(parents=True, exist_ok=True)
        self.experiments_path = self.outputs / "experiments.jsonl"
        self.finished = False
        self.report: str | None = None
        self._figure_count = 0

    def dispatch(self, name: str, args: dict) -> tuple[list[dict], bool]:
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return [text_block(f"Unknown tool: {name}")], True
        try:
            return handler(**args)
        except Exception as exc:  # noqa: BLE001 - tool failures go back to the model
            return [text_block(f"{type(exc).__name__}: {exc}")], True

    def _resolve(self, path: str) -> Path:
        full = (self.workdir / path).resolve()
        if full != self.workdir and self.workdir not in full.parents:
            raise ValueError(f"path {path!r} is outside the working directory")
        return full

    def _tool_run_python(self, code: str, purpose: str = "") -> tuple[list[dict], bool]:
        result = self.kernel.execute(code)
        if purpose:
            self.recorder.add_markdown(f"### {purpose}")
        self.recorder.add_code(code, result.outputs)

        parts = []
        if result.text.strip():
            parts.append(result.text)
        if result.error:
            parts.append(result.error)
        saved = []
        for png in result.images:
            self._figure_count += 1
            path = self.figures / f"fig_{self._figure_count:02d}.png"
            path.write_bytes(base64.b64decode(png))
            saved.append(str(path.relative_to(self.workdir)))
        if saved:
            parts.append("Figures saved: " + ", ".join(saved))
        if not parts:
            parts.append("(no output)")
        blocks = [text_block(truncate("\n".join(parts)))]
        blocks += [image_block(png) for png in result.images[:MAX_IMAGES_PER_CALL]]
        return blocks, not result.ok

    def _tool_list_files(self, path: str = ".") -> tuple[list[dict], bool]:
        root = self._resolve(path)
        lines = []
        for p in sorted(root.rglob("*")):
            if p.is_file() and ".ipynb_checkpoints" not in p.parts:
                lines.append(f"{p.relative_to(self.workdir)}  ({p.stat().st_size:,} bytes)")
        return [text_block("\n".join(lines[:200]) or "(empty)")], False

    def _tool_read_file(self, path: str, max_chars: int = 4000) -> tuple[list[dict], bool]:
        with open(self._resolve(path), encoding="utf-8", errors="replace") as f:
            return [text_block(f.read(max_chars))], False

    def _tool_write_file(self, path: str, content: str) -> tuple[list[dict], bool]:
        full = self._resolve(path)
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")
        return [text_block(f"Wrote {full.relative_to(self.workdir)} ({len(content)} chars)")], False

    def _tool_install_package(self, package: str) -> tuple[list[dict], bool]:
        if not PACKAGE_RE.match(package):
            return [text_block(f"Refusing suspicious package spec: {package!r}")], True
        result = self.kernel.execute(f"%pip install -q {package}")
        return [text_block(truncate(result.text or "installed"))], not result.ok

    def _tool_log_experiment(
        self,
        name: str,
        model: str,
        metrics: dict,
        features: str = "",
        params: dict | None = None,
        notes: str = "",
    ) -> tuple[list[dict], bool]:
        record = {
            "name": name,
            "model": model,
            "features": features,
            "params": params or {},
            "metrics": metrics,
            "notes": notes,
        }
        with open(self.experiments_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
        return [text_block("Leaderboard:\n" + self.leaderboard())], False

    def experiments(self) -> list[dict]:
        if not self.experiments_path.exists():
            return []
        lines = self.experiments_path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def leaderboard(self) -> str:
        rows = self.experiments()
        if not rows:
            return "(no experiments)"
        keys = sorted({k for r in rows for k in r["metrics"]})
        header = "| name | model | " + " | ".join(keys) + " |"
        sep = "|" + "---|" * (len(keys) + 2)
        body = []
        for r in rows:
            vals = []
            for k in keys:
                v = r["metrics"].get(k, "")
                vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
            body.append(f"| {r['name']} | {r['model']} | " + " | ".join(vals) + " |")
        return "\n".join([header, sep, *body])

    def _tool_checkpoint(self, summary: str, question: str) -> tuple[list[dict], bool]:
        self.recorder.add_markdown(f"**Checkpoint.** {summary}")
        if self.auto:
            return [
                text_block("Auto mode: the user is away. Continue with your best judgement.")
            ], False
        answer = self.ask_user(f"\n--- CHECKPOINT ---\n{summary}\n\n{question}\n> ").strip()
        return [text_block(f"User answer: {answer or 'continue'}")], False

    def _tool_save_memory(self, lesson: str) -> tuple[list[dict], bool]:
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        lesson = " ".join(lesson.split())
        with open(self.memory_path, "a", encoding="utf-8") as f:
            f.write(f"- {lesson} [added: {date.today().isoformat()}]\n")
        return [text_block("Saved to memory.")], False

    def _tool_finish(self, report: str) -> tuple[list[dict], bool]:
        self.report = report
        (self.outputs / "report.md").write_text(report, encoding="utf-8")
        self.recorder.add_markdown(report)
        self.finished = True
        return [text_block("Report saved to outputs/report.md. Task finished.")], False
