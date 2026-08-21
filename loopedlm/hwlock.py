"""Code-enforced reservation of the shared GPU.

Discipline does not work: a benchmark of mine outlived its 12-minute booking and
then survived a `timeout` kill, pinning 15.4 GiB of the card for half an hour
while the board showed it as free -- which silently corrupted my own throughput
measurements and blocked another session.  So the claim is no longer something a
human remembers to do around a script; it is a context manager the script cannot
run without, and it keeps itself honest:

  * refuses to start (or waits) when somebody else holds the card;
  * a daemon heartbeat extends the booking while the process is alive, so a run
    that takes longer than predicted never appears free;
  * releases on normal exit, on exception, and on SIGINT/SIGTERM.

If the board CLI is missing entirely the lock degrades to a warning, so the code
still runs on a machine without the board.
"""
from __future__ import annotations

import atexit
import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

BOARD_CLI = Path(os.environ.get("BOARD_CLI", r"C:\ml\gpu_board\gpu_board.py"))
OWNER = os.environ.get("BOARD_OWNER", "looped-lm (chat-1)")
HEARTBEAT_S = 120


def _call(args, timeout=60):
    if not BOARD_CLI.exists():
        return None
    try:
        return subprocess.run([sys.executable, str(BOARD_CLI), *args],
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None


def release(owner: str = OWNER) -> None:
    _call(["release", "--owner", owner], timeout=30)


@contextmanager
def hold_gpu(note: str, minutes: int = 30, owner: str = OWNER, exclusive: bool = True,
             wait: bool = True, wait_timeout: int = 10800, poll: int = 60,
             vram: float = 0.0, ram: float = 0.0, cpu: float = 0.0):
    """Hold the GPU for the duration of the block, extending the booking as needed."""
    args = ["--owner", owner, "--minutes", str(int(minutes)), "--note", note]
    if exclusive:
        args.append("--exclusive")
    else:
        for flag, val in (("--vram", vram), ("--ram", ram), ("--cpu", cpu)):
            if val:
                args += [flag, str(val)]

    if BOARD_CLI.exists():
        cmd = (["wait", *args, "--timeout", str(wait_timeout), "--poll", str(poll)]
               if wait else ["claim", *args])
        r = _call(cmd, timeout=wait_timeout + 120 if wait else 60)
        if r is None:
            print(f"[hwlock] board did not answer; proceeding without a booking", flush=True)
        else:
            line = (r.stdout or r.stderr).strip().splitlines()[-1:] or [""]
            print(f"[hwlock] {line[0]}", flush=True)
            if r.returncode != 0:
                raise SystemExit(f"[hwlock] refused the GPU: {line[0]}")
    else:
        print(f"[hwlock] no board at {BOARD_CLI}; proceeding without a booking", flush=True)

    stop = threading.Event()

    def beat():
        while not stop.wait(HEARTBEAT_S):
            _call(["extend", "--owner", owner, "--minutes", "10", "--note", note]
                  + (["--exclusive"] if exclusive else []), timeout=30)

    hb = threading.Thread(target=beat, daemon=True)
    hb.start()

    released = threading.Event()

    def cleanup(*_):
        if not released.is_set():
            released.set()
            stop.set()
            release(owner)

    atexit.register(cleanup)
    prev = {}
    for sig in (signal.SIGINT, signal.SIGTERM) + ((signal.SIGBREAK,) if hasattr(signal, "SIGBREAK") else ()):
        try:
            prev[sig] = signal.signal(sig, lambda s, f: (cleanup(), sys.exit(130)))
        except (ValueError, OSError):
            pass
    try:
        yield
    finally:
        cleanup()
        for sig, h in prev.items():
            try:
                signal.signal(sig, h)
            except (ValueError, OSError):
                pass
        print(f"[hwlock] released", flush=True)
