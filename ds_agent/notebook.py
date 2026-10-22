"""Records every step the agent runs into a Jupyter notebook you can open and rerun."""

from __future__ import annotations

from pathlib import Path

import nbformat


class NotebookRecorder:
    def __init__(self, path: str | Path, title: str):
        self.path = Path(path)
        self.nb = nbformat.v4.new_notebook()
        self.nb.cells.append(nbformat.v4.new_markdown_cell(f"# {title}"))
        self._count = 0

    def add_markdown(self, text: str) -> None:
        self.nb.cells.append(nbformat.v4.new_markdown_cell(text))
        self.save()

    def add_code(self, code: str, outputs: list) -> None:
        self._count += 1
        cell = nbformat.v4.new_code_cell(code, execution_count=self._count)
        for out in outputs:
            if "execution_count" in out:
                out["execution_count"] = self._count
        cell.outputs = outputs
        self.nb.cells.append(cell)
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(self.nb, self.path)
