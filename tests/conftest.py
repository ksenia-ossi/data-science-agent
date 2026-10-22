import pytest

from ds_agent.kernel import PythonKernel
from ds_agent.notebook import NotebookRecorder
from ds_agent.tools import Toolbox


@pytest.fixture(scope="session")
def kernel(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("kernel")
    with PythonKernel(workdir) as k:
        yield k


@pytest.fixture
def toolbox(tmp_path):
    with PythonKernel(tmp_path) as k:
        recorder = NotebookRecorder(tmp_path / "outputs" / "analysis.ipynb", title="test")
        yield Toolbox(tmp_path, k, recorder, tmp_path / "memory.md", auto=True)
