"""Command line entry point: ds-agent DATA... --task "..." """

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

from .agent import Agent
from .kernel import PythonKernel
from .notebook import NotebookRecorder
from .prompts import build_system_prompt
from .tools import Toolbox

DEFAULT_MODEL = os.environ.get("DS_AGENT_MODEL", "claude-sonnet-5-5")
DEFAULT_MEMORY = Path(os.environ.get("DS_AGENT_MEMORY", Path.home() / ".ds_agent" / "memory.md"))

COLORS = {"thought": "\033[36m", "tool": "\033[33m", "result": "\033[90m", "error": "\033[31m"}


def print_event(kind: str, text: str) -> None:
    prefix = {"thought": "agent", "tool": "  ->", "result": "  ok", "error": "  !!"}[kind]
    color = COLORS[kind] if sys.stdout.isatty() else ""
    reset = "\033[0m" if color else ""
    print(f"{color}{prefix} {text}{reset}", flush=True)


def build_task(task: str, data_files: list[str], target: str | None) -> str:
    lines = [f"Task: {task}", "", "Data files (in data/):"]
    lines += [f"- data/{name}" for name in data_files]
    if target:
        lines.append(f"\nTarget column: {target}")
    lines.append("\nStart by looking at the files.")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="ds-agent", description=__doc__)
    p.add_argument("data", nargs="+", help="data files or directories to work on")
    p.add_argument("--task", required=True, help="what to predict or find out, in plain words")
    p.add_argument("--target", help="target column, if you already know it")
    p.add_argument("--workdir", help="run directory (default: runs/<timestamp>)")
    p.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"Claude model (default: {DEFAULT_MODEL})"
    )
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--auto", action="store_true", help="do not stop at checkpoints")
    p.add_argument("--memory", type=Path, default=DEFAULT_MEMORY, help="memory file")
    return p.parse_args(argv)


def prepare_workdir(workdir: Path, data: list[str]) -> list[str]:
    data_dir = workdir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for item in data:
        src = Path(item)
        if src.is_dir():
            shutil.copytree(src, data_dir / src.name, dirs_exist_ok=True)
            names += [
                str(p.relative_to(data_dir))
                for p in (data_dir / src.name).rglob("*")
                if p.is_file()
            ]
        elif src.is_file():
            shutil.copy2(src, data_dir / src.name)
            names.append(src.name)
        else:
            raise SystemExit(f"not found: {item}")
    return names


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY first.")
    import anthropic

    workdir = Path(args.workdir or f"runs/{time.strftime('%Y%m%d-%H%M%S')}").resolve()
    files = prepare_workdir(workdir, args.data)
    memory = args.memory.read_text(encoding="utf-8") if args.memory.exists() else ""
    recorder = NotebookRecorder(workdir / "outputs" / "analysis.ipynb", title=args.task)

    print(f"Run directory: {workdir}\nModel: {args.model}\n")
    with PythonKernel(workdir) as kernel:
        toolbox = Toolbox(workdir, kernel, recorder, args.memory, auto=args.auto)
        agent = Agent(
            anthropic.Anthropic(max_retries=5),
            args.model,
            toolbox,
            build_system_prompt(memory),
            max_steps=args.max_steps,
            on_event=print_event,
            call_log=workdir / "outputs" / "llm_calls.jsonl",
        )
        result = agent.run(build_task(args.task, files, args.target))

    u = result.usage
    print(
        f"\nSteps: {result.steps}. Tokens: input {u.input_tokens:,}, output {u.output_tokens:,}, "
        f"cache read {u.cache_read_tokens:,}, cache write {u.cache_write_tokens:,}.\n"
        f"Model calls: {u.calls}, latency mean {u.mean_latency_s:.1f} s, "
        f"max {u.max_latency_s:.1f} s, total {u.latency_s:.0f} s "
        f"(per call: outputs/llm_calls.jsonl)."
    )
    print(f"Leaderboard:\n{toolbox.leaderboard()}\n")
    if result.finished:
        print(f"Report:   {workdir / 'outputs' / 'report.md'}")
        print(f"Notebook: {workdir / 'outputs' / 'analysis.ipynb'}")
        return 0
    print("The agent stopped before calling finish (step limit). Partial results are in outputs/.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
