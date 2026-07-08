import signal
import sys

from stark.evaluation.isolated import _run_worker_process, _worker_signal_log


def test_worker_sigkill_is_reported_without_killing_parent(tmp_path):
    script = tmp_path / "kill_self.py"
    script.write_text(
        "import os, signal\nos.kill(os.getpid(), signal.SIGKILL)\n",
        encoding="utf-8",
    )

    result = _run_worker_process([sys.executable, str(script)], cwd=tmp_path, timeout_seconds=10)

    assert result.returncode == -signal.SIGKILL
    assert not result.timed_out
    assert _worker_signal_log(result.returncode) == "isolated_worker_signal=SIGKILL"


def test_worker_timeout_kills_detached_group(tmp_path):
    script = tmp_path / "sleep_forever.py"
    script.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")

    result = _run_worker_process([sys.executable, str(script)], cwd=tmp_path, timeout_seconds=1)

    assert result.timed_out
    assert result.returncode != 0
