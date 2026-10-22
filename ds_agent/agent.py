"""The agent loop: ask the model, run the tools it calls, feed results back, repeat."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .tools import TOOL_SPECS, Toolbox, text_block

WRAP_UP_AT = 4
OMITTED_IMAGE = "[figure omitted from context to save tokens; it is saved under outputs/figures]"


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def add(self, usage) -> None:
        self.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0


@dataclass
class RunResult:
    finished: bool
    steps: int
    report: str | None
    usage: Usage = field(default_factory=Usage)


def _block_to_dict(block) -> dict:
    if isinstance(block, dict):
        return block
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    return block.model_dump(exclude_none=True)


class Agent:
    def __init__(
        self,
        client,
        model: str,
        toolbox: Toolbox,
        system_prompt: str,
        max_steps: int = 40,
        max_tokens: int = 8000,
        keep_images: int = 2,
        on_event: Callable[[str, str], None] | None = None,
    ):
        self.client = client
        self.model = model
        self.toolbox = toolbox
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.keep_images = keep_images
        self.on_event = on_event or (lambda kind, text: None)
        self.messages: list[dict] = []
        self.usage = Usage()
        self.tools = [dict(t) for t in TOOL_SPECS]
        self.tools[-1]["cache_control"] = {"type": "ephemeral"}

    def run(self, task: str) -> RunResult:
        self.messages = [{"role": "user", "content": [text_block(task)]}]
        nudged = False
        step = 0
        while step < self.max_steps and not self.toolbox.finished:
            step += 1
            self._prepare_context()
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": self.system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                tools=self.tools,
                messages=self.messages,
            )
            self.usage.add(response.usage)
            content = [_block_to_dict(b) for b in response.content]
            self.messages.append({"role": "assistant", "content": content})

            tool_uses = [b for b in content if b["type"] == "tool_use"]
            for b in content:
                if b["type"] == "text" and b["text"].strip():
                    self.on_event("thought", b["text"].strip())

            if not tool_uses:
                if nudged:
                    break
                nudged = True
                self.messages.append(
                    {
                        "role": "user",
                        "content": [
                            text_block(
                                "Continue with the tools. Call finish when the task is done."
                            )
                        ],
                    }
                )
                continue

            results = []
            for use in tool_uses:
                self.on_event("tool", _describe(use))
                blocks, is_error = self.toolbox.dispatch(use["name"], use["input"])
                self.on_event("error" if is_error else "result", _first_text(blocks))
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": use["id"],
                        "content": blocks,
                        "is_error": is_error,
                    }
                )
            left = self.max_steps - step
            if 0 < left <= WRAP_UP_AT and not self.toolbox.finished:
                results.append(
                    text_block(f"Only {left} steps left. Save the model and call finish now.")
                )
            self.messages.append({"role": "user", "content": results})

        return RunResult(self.toolbox.finished, step, self.toolbox.report, self.usage)

    def _prepare_context(self) -> None:
        """Drop old figures from the context and move the prompt-cache marker to the last block."""
        image_results = []
        for msg in self.messages:
            if msg["role"] != "user":
                continue
            for block in msg["content"]:
                block.pop("cache_control", None)
                if block.get("type") == "tool_result" and any(
                    b.get("type") == "image" for b in block["content"]
                ):
                    image_results.append(block)
        for block in (
            image_results[: -self.keep_images or None] if self.keep_images else image_results
        ):
            block["content"] = [
                b if b.get("type") != "image" else text_block(OMITTED_IMAGE)
                for b in block["content"]
            ]
        self.messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}


def _describe(use: dict) -> str:
    args = use["input"]
    if use["name"] == "run_python":
        return f"run_python: {args.get('purpose', '')}"
    if use["name"] in ("read_file", "write_file", "list_files"):
        return f"{use['name']}: {args.get('path', '.')}"
    if use["name"] == "log_experiment":
        return f"log_experiment: {args.get('name')} {args.get('metrics')}"
    if use["name"] == "install_package":
        return f"install_package: {args.get('package')}"
    return use["name"]


def _first_text(blocks: list[dict], limit: int = 400) -> str:
    for b in blocks:
        if b.get("type") == "text":
            text = b["text"].strip()
            return text if len(text) <= limit else text[:limit] + " ..."
    return ""
