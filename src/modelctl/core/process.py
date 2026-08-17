#!/usr/bin/env python3
"""core/process.py — 引擎无关的进程生命周期：后台启动、PID、停止、健康检查。"""

from __future__ import annotations

import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from modelctl.core.envfile import PROJECT_ROOT


def log_dir() -> Path:
    d = Path(os.environ.get("LOG_DIR") or PROJECT_ROOT.parent / "logs")
    d.mkdir(parents=True, exist_ok=True)
    return d


def pid_file(name: str) -> Path:
    return log_dir() / f"{name}.pid"


def launch_log(name: str) -> Path | None:
    logs = sorted(log_dir().glob(f"launch-{name}-*.log"))
    return logs[-1] if logs else None


def start_detached(name: str, command: list[str], extra_env: dict[str, str]) -> int:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir() / f"launch-{name}-{stamp}.log"
    env = {**os.environ, **extra_env}
    fp = open(log_path, "a", encoding="utf-8")
    kwargs: dict = {"stdout": fp, "stderr": subprocess.STDOUT, "env": env, "stdin": subprocess.DEVNULL}
    kwargs["start_new_session"] = True  # nohup 语义：SSH 断开不影响
    proc = subprocess.Popen(command, **kwargs)
    pid_file(name).write_text(str(proc.pid), encoding="utf-8")
    return proc.pid


def is_running(name: str) -> bool:
    pf = pid_file(name)
    if not pf.is_file():
        return False
    try:
        pid = int(pf.read_text(encoding="utf-8").strip())
    except ValueError:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def stop_instance(name: str, port: int, patterns: list[str]) -> bool:
    """先 PID 优雅终止，再按端口/进程名兜底。返回是否有进程被终止。"""
    stopped = False
    pf = pid_file(name)
    if pf.is_file():
        try:
            pid = int(pf.read_text(encoding="utf-8").strip())
            os.killpg(pid, signal.SIGTERM)  # type: ignore[attr-defined]  # POSIX-only，Windows 类型桩无此 API
            deadline = time.time() + 10
            while time.time() < deadline:
                try:
                    os.kill(pid, 0)
                    time.sleep(0.5)
                except OSError:
                    break
            else:
                os.killpg(pid, signal.SIGKILL)  # type: ignore[attr-defined]  # POSIX-only，Windows 类型桩无此 API
            stopped = True
        except (ValueError, OSError):
            pass
        pf.unlink(missing_ok=True)
    subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True)
    for pat in patterns:
        subprocess.run(["pkill", "-f", pat], capture_output=True)
    return stopped


def wait_health(url: str, timeout: float, api_key: str | None = None) -> bool:
    deadline = time.time() + timeout
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=5) as resp:
                if 200 <= resp.status < 300:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)
    return False


def tail_file(path: Path, lines: int) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(content[-lines:])
