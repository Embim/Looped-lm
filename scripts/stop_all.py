"""Stop every training/benchmark process of this project and release the board.

Needed because killing the queue parent does not kill the child run: Windows has
no process group to signal, and a command line written with backslashes does not
match a pattern written with forward slashes -- an orphaned run of mine kept the
card at 97% and silently halved a benchmark I was reading at the same time.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PATTERNS = (r"scripts\train.py", "scripts/train.py", "run_experiments", "bench_", "loopedlm")
BOARD = Path(r"C:\ml\gpu_board\gpu_board.py")
OWNER = "looped-lm (chat-1)"
# PowerShell writes its output in the console code page, so a command line holding
# the (Cyrillic) project path raised UnicodeDecodeError and this script reported
# "0 processes" while a run was still on the GPU.  Force UTF-8 out, be permissive in.
PS = ("[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; "
      "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
      "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    import json
    r = subprocess.run(["powershell", "-NoProfile", "-Command", PS],
                       capture_output=True, encoding="utf-8", errors="replace")
    out = (r.stdout or "").strip()
    if not out:
        raise SystemExit(f"could not list processes: {(r.stderr or '')[:200]}")
    procs = json.loads(out)
    if isinstance(procs, dict):
        procs = [procs]
    me = str(Path(__file__).resolve().parents[1]).lower()
    # Same discipline as ${VAR:?} in a shell rm path: if the discriminator is
    # empty or implausibly short, every python process would match and get
    # killed.  Refuse instead of widening the blast radius.
    if len(me) < 12:
        raise SystemExit(f"refusing to match on a too-short project path: {me!r}")
    killed = 0
    for p in procs:
        cmd = (p.get("CommandLine") or "")
        low = cmd.lower()
        if "stop_all" in low or "gpu_board.py agent" in low:
            continue
        if me in low or any(x.lower() in low for x in PATTERNS):
            print(f"{'would kill' if a.dry_run else 'killing'} {p['ProcessId']}: {cmd[:100]}")
            if not a.dry_run:
                subprocess.run(["taskkill", "/F", "/PID", str(p["ProcessId"])],
                               capture_output=True)
            killed += 1
    print(f"{killed} process(es)")
    if not a.dry_run and BOARD.exists():
        subprocess.run([sys.executable, str(BOARD), "release", "--owner", OWNER])


if __name__ == "__main__":
    main()
