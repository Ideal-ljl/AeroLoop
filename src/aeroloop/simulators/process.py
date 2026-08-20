from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


def scene_directory(env_root: str | Path, env_name: str) -> Path:
    root = Path(env_root).expanduser().resolve()
    scene = (root / env_name).resolve()
    if scene != root and root not in scene.parents:
        raise ValueError(f"environment name escapes env_root: {env_name!r}")
    if not scene.is_dir():
        raise FileNotFoundError(f"simulator scene directory not found: {scene}")
    return scene


def find_file(root: Path, filename: str) -> Path:
    matches = list(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"{filename} not found below {root}")
    if len(matches) > 1:
        raise RuntimeError(f"multiple {filename} files below {root}: {matches}")
    return matches[0]


def set_json_value(path: Path, key: str, value: Any) -> None:
    row = json.loads(path.read_text(encoding="utf-8"))
    if row.get(key) == value:
        return
    row[key] = value
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def wait_for_port(host: str, port: int, timeout_s: float, process=None) -> None:
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"simulator process exited early with code {process.returncode}")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            try:
                sock.connect((host, int(port)))
                return
            except OSError:
                pass
        time.sleep(0.5)
    raise TimeoutError(f"timed out waiting for simulator on {host}:{port}")


def launch_process(
    command: Sequence[str],
    *,
    cwd: str | Path,
    environment: Mapping[str, str] | None = None,
    log_path: str | Path | None = None,
):
    log_handle = None
    stdout: Any = subprocess.DEVNULL
    if log_path:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = path.open("ab")
        stdout = log_handle
    process = subprocess.Popen(
        list(command),
        cwd=str(cwd),
        env={**os.environ, **dict(environment or {})},
        stdout=stdout,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    process._uav_eval_log_handle = log_handle  # type: ignore[attr-defined]
    return process


def stop_process(process, timeout_s: float = 8.0) -> None:
    if process is None:
        return
    if process.poll() is None:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=timeout_s)
        except Exception:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except Exception:
                process.kill()
    handle = getattr(process, "_uav_eval_log_handle", None)
    if handle is not None:
        handle.close()
