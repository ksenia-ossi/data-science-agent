def test_state_persists_between_calls(kernel):
    kernel.execute("x = 21")
    result = kernel.execute("print(x * 2)")
    assert result.ok
    assert result.text.strip() == "42"


def test_last_expression_is_returned(kernel):
    assert "3" in kernel.execute("1 + 2").text


def test_error_is_captured(kernel):
    result = kernel.execute("1 / 0")
    assert not result.ok
    assert "ZeroDivisionError" in result.error


def test_matplotlib_figure_is_captured(kernel):
    result = kernel.execute("import matplotlib.pyplot as plt\nplt.plot([1, 2, 3])\nplt.show()")
    assert result.ok
    assert len(result.images) == 1


def test_timeout_interrupts(kernel):
    result = kernel.execute("import time\ntime.sleep(30)", timeout=2)
    assert result.timed_out
    assert kernel.execute("print('alive')").text.strip() == "alive"
