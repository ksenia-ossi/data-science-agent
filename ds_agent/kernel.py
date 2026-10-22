"""A persistent Jupyter kernel, so variables survive between agent steps like in a notebook."""

from __future__ import annotations

import queue
import re
from dataclasses import dataclass, field
from pathlib import Path

import nbformat
from jupyter_client import KernelManager

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

SETUP_CODE = """
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.rcParams["figure.dpi"] = 80
"""


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


@dataclass
class ExecResult:
    text: str = ""
    images: list[str] = field(default_factory=list)
    error: str | None = None
    outputs: list = field(default_factory=list)
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None and not self.timed_out


class PythonKernel:
    def __init__(self, workdir: str | Path, kernel_name: str = "python3"):
        self.workdir = Path(workdir)
        self.kernel_name = kernel_name
        self.km: KernelManager | None = None
        self.kc = None

    def start(self) -> None:
        self.km = KernelManager(kernel_name=self.kernel_name)
        self.km.start_kernel(cwd=str(self.workdir))
        self.kc = self.km.client()
        self.kc.start_channels()
        self.kc.wait_for_ready(timeout=60)
        self.execute(SETUP_CODE)

    def shutdown(self) -> None:
        if self.kc is not None:
            self.kc.stop_channels()
        if self.km is not None:
            self.km.shutdown_kernel(now=True)
        self.km = self.kc = None

    def __enter__(self) -> PythonKernel:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()

    def execute(self, code: str, timeout: float = 600) -> ExecResult:
        assert self.kc is not None, "kernel is not started"
        msg_id = self.kc.execute(code)
        result = ExecResult()
        chunks: list[str] = []
        while True:
            try:
                msg = self.kc.get_iopub_msg(timeout=timeout)
            except queue.Empty:
                result.timed_out = True
                chunks.append(f"\n[execution interrupted after {timeout:.0f}s]")
                if not self._interrupt(msg_id):
                    chunks.append(" [kernel restarted: all variables were lost]")
                break
            if msg["parent_header"].get("msg_id") != msg_id:
                continue
            kind, content = msg["msg_type"], msg["content"]
            if kind == "status" and content["execution_state"] == "idle":
                break
            if kind == "stream":
                chunks.append(content["text"])
                result.outputs.append(
                    nbformat.v4.new_output("stream", name=content["name"], text=content["text"])
                )
            elif kind in ("execute_result", "display_data"):
                data = content["data"]
                if "image/png" in data:
                    result.images.append(data["image/png"])
                elif "text/plain" in data:
                    chunks.append(data["text/plain"] + "\n")
                if kind == "execute_result":
                    out = nbformat.v4.new_output(
                        "execute_result", data=data, execution_count=content["execution_count"]
                    )
                else:
                    out = nbformat.v4.new_output("display_data", data=data)
                result.outputs.append(out)
            elif kind == "error":
                tb = strip_ansi("\n".join(content["traceback"]))
                result.error = tb
                result.outputs.append(
                    nbformat.v4.new_output(
                        "error",
                        ename=content["ename"],
                        evalue=content["evalue"],
                        traceback=content["traceback"],
                    )
                )
        result.text = strip_ansi("".join(chunks))
        return result

    def _interrupt(self, msg_id: str, grace: float = 15) -> bool:
        """Interrupt a running cell and wait until the kernel is idle; restart it if stuck."""
        self.km.interrupt_kernel()
        while True:
            try:
                msg = self.kc.get_iopub_msg(timeout=grace)
            except queue.Empty:
                self.km.restart_kernel(now=True)
                self.kc.wait_for_ready(timeout=60)
                self.execute(SETUP_CODE)
                return False
            if (
                msg["parent_header"].get("msg_id") == msg_id
                and msg["msg_type"] == "status"
                and msg["content"]["execution_state"] == "idle"
            ):
                self.kc.get_shell_msg(timeout=grace)
                return True
