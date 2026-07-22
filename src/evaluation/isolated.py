from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..models import EvaluationResult, StarkConfig, TaskSpec
from .base import Evaluator, _failure_result


class IsolatedEvaluator(Evaluator):
    def __init__(self, inner: Evaluator, timeout_seconds: int = 900):
        self.inner = inner
        self.timeout_seconds = int(timeout_seconds or 900)
        self.repo_root = Path(__file__).resolve().parents[2]

    def evaluate(self, task: TaskSpec, code: str, config: StarkConfig) -> EvaluationResult:
        if getattr(config, "evaluator_isolation", "off") != "candidate_subprocess":
            return self.inner.evaluate(task, code, config)
        payload = {
            "task": _task_payload(task),
            "config": asdict(config),
            "code": code,
        }
        temp_path = Path(tempfile.mkdtemp(prefix="kw_eval_"))
        trace_path = _make_trace_dir(task)
        try:
            input_path = temp_path / "input.json"
            output_path = temp_path / "output.json"
            stdout_path = temp_path / "worker_stdout.log"
            stderr_path = temp_path / "worker_stderr.log"
            input_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            if trace_path is not None:
                shutil.copy2(input_path, trace_path / "input.json")
            command = [
                sys.executable,
                "-m",
                "src.evaluation.worker",
                str(input_path),
                str(output_path),
            ]
            if getattr(config, "verbose", False):
                trace_label = f" trace_dir={trace_path}" if trace_path is not None else ""
                print(
                    f"[evaluator] isolated_worker_start timeout={self.timeout_seconds}s "
                    f"detached_session=true task={task.name}{trace_label}",
                    flush=True,
                )
            completed = _run_worker_process(
                command,
                cwd=self.repo_root,
                timeout_seconds=self.timeout_seconds,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )
            _mirror_eval_artifacts(temp_path, trace_path)
            common_logs = [
                "isolated_worker_detached_session=true",
                f"isolated_worker_session_id={completed.session_id}",
                f"isolated_worker_exit_code={completed.returncode}",
                _worker_signal_log(completed.returncode),
                _tail_log("isolated_worker_stdout_tail", completed.stdout),
                _tail_log("isolated_worker_stderr_tail", completed.stderr),
            ]
            if trace_path is not None:
                common_logs.append(f"isolated_worker_trace_dir={trace_path}")
            if completed.timed_out:
                return _isolated_failure(
                    "isolated_evaluator_timeout",
                    [
                        "isolated_worker_timed_out=true",
                        f"isolated_worker_timeout_seconds={self.timeout_seconds}",
                        *common_logs,
                    ],
                )
            if completed.returncode != 0:
                return _isolated_failure("isolated_evaluator_crash", common_logs)
            if not output_path.exists():
                return _isolated_failure(
                    "isolated_evaluator_missing_output",
                    ["isolated_worker_output_missing=true", *common_logs],
                )
            try:
                data = json.loads(output_path.read_text(encoding="utf-8"))
                result = _evaluation_from_payload(data)
            except Exception as exc:
                return _isolated_failure(
                    "isolated_evaluator_bad_output",
                    [f"isolated_worker_bad_output={exc}", *common_logs],
                )
            if getattr(config, "verbose", False):
                print(
                    f"[evaluator] isolated_worker_done exit_code={completed.returncode} "
                    f"detached_session=true task={task.name}",
                    flush=True,
                )
            result.logs.append("isolated_evaluator=candidate_subprocess")
            result.logs.append("isolated_worker_detached_session=true")
            result.logs.append(f"isolated_worker_session_id={completed.session_id}")
            if trace_path is not None:
                result.logs.append(f"isolated_worker_trace_dir={trace_path}")
            if completed.stderr.strip():
                result.logs.append(_tail_log("isolated_worker_stderr_tail", completed.stderr))
            return result
        finally:
            shutil.rmtree(temp_path, ignore_errors=True)


@dataclass
class _WorkerProcessResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    session_id: int | None = None


def _run_worker_process(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: int,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
) -> _WorkerProcessResult:
    owned_log_dir: tempfile.TemporaryDirectory[str] | None = None
    if stdout_path is None or stderr_path is None:
        owned_log_dir = tempfile.TemporaryDirectory(prefix="kw_worker_logs_")
        log_dir = Path(owned_log_dir.name)
        stdout_path = stdout_path or log_dir / "worker_stdout.log"
        stderr_path = stderr_path or log_dir / "worker_stderr.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open("w", encoding="utf-8") as stderr_file:
            process = subprocess.Popen(
                command,
                cwd=str(cwd),
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                start_new_session=_supports_detached_session(),
                close_fds=True,
            )
            session_id = process.pid if _supports_detached_session() else None
            timed_out = False
            try:
                returncode = int(process.wait(timeout=timeout_seconds) or 0)
            except subprocess.TimeoutExpired:
                timed_out = True
                _terminate_worker_session(process, session_id=session_id)
                try:
                    returncode = int(process.wait(timeout=5) or 0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    try:
                        returncode = int(process.wait(timeout=5) or -signal.SIGKILL)
                    except subprocess.TimeoutExpired:
                        returncode = -signal.SIGKILL
            finally:
                if session_id is not None:
                    _terminate_lingering_session_members(session_id)
        return _WorkerProcessResult(
            returncode=returncode,
            stdout=_read_text(stdout_path),
            stderr=_read_text(stderr_path),
            timed_out=timed_out,
            session_id=session_id,
        )
    finally:
        if owned_log_dir is not None:
            owned_log_dir.cleanup()

def _supports_detached_session() -> bool:
    return os.name != "nt"


def _terminate_worker_session(process: subprocess.Popen, *, session_id: int | None) -> None:
    if process.poll() is not None:
        return
    if session_id is not None:
        _send_session_signal(session_id, signal.SIGTERM)
        try:
            process.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            _send_session_signal(session_id, signal.SIGKILL)
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def _terminate_lingering_session_members(session_id: int) -> None:
    members = [pid for pid in _session_member_pids(session_id) if pid != os.getpid()]
    if not members:
        return
    _send_signal_to_pids(members, signal.SIGTERM)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        remaining = [pid for pid in _session_member_pids(session_id) if pid != os.getpid()]
        if not remaining:
            return
        time.sleep(0.1)
    remaining = [pid for pid in _session_member_pids(session_id) if pid != os.getpid()]
    _send_signal_to_pids(remaining, signal.SIGKILL)


def _send_session_signal(session_id: int, sig: signal.Signals) -> None:
    pids = [pid for pid in _session_member_pids(session_id) if pid != os.getpid()]
    if pids:
        _send_signal_to_pids(pids, sig)
        return
    try:
        os.killpg(session_id, sig)
    except ProcessLookupError:
        return


def _send_signal_to_pids(pids: list[int], sig: signal.Signals) -> None:
    for pid in sorted(set(pids), reverse=True):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass


def _session_member_pids(session_id: int) -> list[int]:
    if session_id is None or os.name == "nt":
        return []
    try:
        result = subprocess.run(
            ["ps", "-s", str(session_id), "-o", "pid="],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=2,
            check=False,
        )
    except Exception:
        return []
    pids: list[int] = []
    for raw in result.stdout.split():
        try:
            pids.append(int(raw))
        except ValueError:
            continue
    return pids


def _worker_signal_log(returncode: int) -> str:
    if returncode >= 0:
        return "isolated_worker_signal=none"
    signal_number = -returncode
    try:
        signal_name = signal.Signals(signal_number).name
    except ValueError:
        signal_name = f"SIG{signal_number}"
    return f"isolated_worker_signal={signal_name}"


def _make_trace_dir(task: TaskSpec) -> Path | None:
    root = os.environ.get("KERNELWEAVER_EVAL_TRACE_DIR")
    if not root:
        return None
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", task.name).strip("_")[:80] or "task"
    trace_dir = Path(root) / f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{uuid.uuid4().hex[:8]}_{safe_name}"
    trace_dir.mkdir(parents=True, exist_ok=False)
    return trace_dir


def _mirror_eval_artifacts(temp_path: Path, trace_path: Path | None) -> None:
    if trace_path is None:
        return
    trace_path.mkdir(parents=True, exist_ok=True)
    for name in ["output.json", "worker_stdout.log", "worker_stderr.log"]:
        source = temp_path / name
        if source.exists():
            shutil.copy2(source, trace_path / name)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _task_payload(task: TaskSpec) -> dict[str, Any]:
    return {
        "name": task.name,
        "description": task.description,
        "source_code": task.source_code,
        "reference_code": task.reference_code,
        "function_name": task.function_name,
        "reference_function_name": task.reference_function_name,
        "tags": list(task.tags),
        "source_origin": task.source_origin,
        "benchmark_family": task.benchmark_family,
        "entry_kind": task.entry_kind,
        "level": task.level,
        "problem_id": task.problem_id,
        "backend": task.backend,
        "source_root": task.source_root,
    }


def _evaluation_from_payload(payload: dict[str, Any]) -> EvaluationResult:
    return EvaluationResult(
        compile_ok=bool(payload.get("compile_ok", False)),
        correct=bool(payload.get("correct", False)),
        runtime=payload.get("runtime"),
        score=float(payload.get("score", float("inf"))),
        logs=list(payload.get("logs") or []),
        failure_type=payload.get("failure_type"),
        failure_stage=str(payload.get("failure_stage", "none")),
        reference_runtime=payload.get("reference_runtime"),
        speedup=payload.get("speedup"),
        reference_runtimes=dict(payload.get("reference_runtimes") or {}),
        speedups=dict(payload.get("speedups") or {}),
        primary_reference=payload.get("primary_reference"),
    )


def _isolated_failure(failure_type: str, logs: list[str]) -> EvaluationResult:
    return _failure_result("runtime", failure_type, *[log for log in logs if log])


def _tail_log(label: str, value: Any, limit: int = 2000) -> str:
    text = "" if value is None else str(value)
    text = text.strip()
    if len(text) > limit:
        text = text[-limit:]
    return f"{label}={text}"