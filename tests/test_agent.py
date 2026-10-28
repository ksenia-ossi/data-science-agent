import copy
import json
from types import SimpleNamespace

from ds_agent.agent import OMITTED_IMAGE, Agent


def tool_use(id_, name, args):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=args)


def text(t):
    return SimpleNamespace(type="text", text=t)


class FakeClient:
    """Plays back scripted model responses and records what the agent sent."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        content = self.responses.pop(0)
        stop = "tool_use" if any(b.type == "tool_use" for b in content) else "end_turn"
        usage = SimpleNamespace(input_tokens=10, output_tokens=5)
        return SimpleNamespace(content=content, stop_reason=stop, usage=usage)


def make_agent(toolbox, responses, **kwargs):
    client = FakeClient(responses)
    return Agent(client, "fake-model", toolbox, "system", **kwargs), client


def test_agent_runs_tools_until_finish(toolbox):
    agent, client = make_agent(
        toolbox,
        [
            [text("Load data."), tool_use("t1", "run_python", {"purpose": "x", "code": "x = 2"})],
            [tool_use("t2", "run_python", {"purpose": "y", "code": "print(x + 1)"})],
            [tool_use("t3", "finish", {"report": "# Done"})],
        ],
    )
    result = agent.run("do it")
    assert result.finished
    assert result.steps == 3
    assert result.report == "# Done"
    assert result.usage.input_tokens == 30
    second_call_results = client.calls[2]["messages"][-1]["content"]
    assert second_call_results[0]["tool_use_id"] == "t2"
    assert "3" in second_call_results[0]["content"][0]["text"]


def test_agent_nudges_once_then_stops(toolbox):
    agent, client = make_agent(toolbox, [[text("I think we are done.")], [text("Really done.")]])
    result = agent.run("do it")
    assert not result.finished
    assert len(client.calls) == 2


def test_agent_respects_step_limit_and_warns(toolbox):
    responses = [[tool_use(f"t{i}", "list_files", {})] for i in range(5)]
    agent, client = make_agent(toolbox, responses, max_steps=5)
    result = agent.run("do it")
    assert not result.finished
    assert result.steps == 5
    warnings = [
        b["text"] for b in client.calls[-1]["messages"][-1]["content"] if b["type"] == "text"
    ]
    assert any("steps left" in w for w in warnings)


def test_old_images_are_pruned_and_cache_marker_moves(toolbox):
    plot = {"purpose": "p", "code": "import matplotlib.pyplot as plt\nplt.plot([1]); plt.show()"}
    responses = [[tool_use(f"t{i}", "run_python", plot)] for i in range(4)]
    responses.append([tool_use("end", "finish", {"report": "r"})])
    agent, client = make_agent(toolbox, responses, keep_images=2)
    agent.run("plots")
    messages = client.calls[-1]["messages"]
    results = [
        b
        for m in messages
        if m["role"] == "user"
        for b in m["content"]
        if b.get("type") == "tool_result"
    ]
    with_images = [r for r in results if any(c["type"] == "image" for c in r["content"])]
    assert len(with_images) == 2
    assert any(c.get("text") == OMITTED_IMAGE for c in results[0]["content"])
    markers = [b for m in messages for b in m["content"] if "cache_control" in b]
    assert len(markers) == 1


def test_agent_logs_latency_and_tokens_per_call(toolbox, tmp_path):
    log = tmp_path / "llm_calls.jsonl"
    agent, _ = make_agent(
        toolbox,
        [
            [tool_use("t1", "run_python", {"purpose": "x", "code": "x = 1"})],
            [tool_use("t2", "finish", {"report": "# Done"})],
        ],
        call_log=log,
    )
    result = agent.run("do it")
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["step"] for r in rows] == [1, 2]
    assert all(r["input_tokens"] == 10 and r["output_tokens"] == 5 for r in rows)
    assert all(r["latency_s"] >= 0 for r in rows)
    assert result.usage.calls == 2
    assert result.usage.max_latency_s >= result.usage.mean_latency_s
