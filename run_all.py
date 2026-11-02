"""Run the six training scripts one after another, then evaluate every finished run.

Start it detached, never inside a notebook cell (see README.md):
    cd /workspace/blur_imbalance && mkdir -p logs && nohup python -u run_all.py --batch 16 > logs/launcher.txt 2>&1 &
Watch it:
    tail -n 40 /workspace/blur_imbalance/logs/train_log.txt

Everything (this runner and all six scripts) is written to logs/train_log.txt.

Hang protection: every training script prints a progress line at least every 30 s. If the log stops
growing for --stall-minutes, the script is killed together with its data-loader workers and retried
with fewer workers (8 -> 2 -> 0). A crash is retried the same way. After the last attempt the method
is marked FAILED and the runner moves on. Finished runs (runs/<name>/done.json) are skipped, so
starting run_all.py again resumes where it stopped.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = HERE / "logs" / "train_log.txt"

# (method, script, extra arguments); "{blur_csv}" is replaced by --blur-csv
SCRIPTS = [
    ("baseline", "train_1_baseline.py", []),
    ("coreset", "train_2_coreset.py", ["--report-csv", "{blur_csv}"]),
    ("undersampling", "train_3_undersampling.py", []),
    ("smote", "train_4_smote.py", []),
    ("repeat_factor", "train_5_repeat_factor.py", []),
    ("focal_loss", "train_6_focal_loss.py", []),
]


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [run_all] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def kill_tree(proc: subprocess.Popen) -> None:
    """Kill a script and every process it started (its data-loader workers)."""
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGKILL)  # started with start_new_session=True -> own process group
        except ProcessLookupError:
            pass
    else:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    proc.wait()


def run_watched(cmd: list[str], stall_s: float) -> str:
    """Run cmd with its output appended to LOG. Returns 'ok', 'failed' or 'stalled'."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    with open(LOG, "ab") as out:
        proc = subprocess.Popen(cmd, cwd=HERE, stdout=out, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    last_size, last_change = -1, time.time()
    while proc.poll() is None:
        time.sleep(10)
        size = LOG.stat().st_size
        if size != last_size:
            last_size, last_change = size, time.time()
        elif time.time() - last_change > stall_s:
            log(f"no new output for {stall_s / 60:.0f} min -> killing pid {proc.pid} and its workers")
            kill_tree(proc)
            return "stalled"
    return "ok" if proc.returncode == 0 else "failed"


def main():
    base = Path("/workspace") if Path("/workspace/rawdata").exists() else Path.home() / "Desktop"
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", nargs="+", type=int, default=[0], help="e.g. --seeds 0 1 2 for three repeats")
    p.add_argument("--only", nargs="+", choices=[m for m, _, _ in SCRIPTS], help="run only these methods")
    p.add_argument("--data", default=str(base / "rawdata" / "data.yaml"))
    p.add_argument("--test2", default=None, help="second test-set yaml (default: data_moretest.yaml next to --data)")
    p.add_argument("--blur-csv", default=str(base / "blur_labels.csv"))
    p.add_argument("--model", default="yolo11x.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=8, help="8 for a 24 GB GPU, 16 for 48 GB+")
    p.add_argument("--workers", type=int, default=8, help="data-loader workers for the first attempt")
    p.add_argument("--device", default="0")
    p.add_argument("--stall-minutes", type=float, default=20, help="kill a script after this long without output")
    p.add_argument("--no-eval", action="store_true", help="skip the evaluation after training")
    args = p.parse_args()

    LOG.parent.mkdir(parents=True, exist_ok=True)
    project = HERE / "runs"
    stall_s = args.stall_minutes * 60
    worker_plan = list(dict.fromkeys(w for w in (args.workers, 2, 0) if w <= args.workers))
    shared = ["--data", args.data, "--model", args.model, "--epochs", str(args.epochs), "--imgsz", str(args.imgsz),
              "--batch", str(args.batch), "--device", args.device, "--project", str(project)]
    jobs = [job for job in SCRIPTS if not args.only or job[0] in args.only]
    log(f"start: {len(jobs)} methods x seeds {args.seeds} | {args.model} imgsz={args.imgsz} batch={args.batch} "
        f"epochs={args.epochs} | data {args.data} | python {sys.version.split()[0]}")

    results = {}
    for seed in args.seeds:
        for method, script, extra in jobs:
            name = f"{method}_s{seed}"
            if (project / name / "done.json").exists():
                log(f"{name}: already finished, skipping")
                results[name] = "finished earlier"
                continue
            status = "failed"
            for attempt, workers in enumerate(worker_plan, 1):
                log(f"=== {name}: attempt {attempt}/{len(worker_plan)}, {workers} data-loader workers")
                start = time.time()
                cmd = [sys.executable, "-u", script, *shared, "--seed", str(seed), "--workers", str(workers),
                       *(a.format(blur_csv=args.blur_csv) for a in extra)]
                status = run_watched(cmd, stall_s)
                if status == "ok" and not (project / name / "done.json").exists():
                    status = "failed"
                log(f"=== {name}: {status.upper()} after {(time.time() - start) / 3600:.2f} h")
                if status == "ok":
                    break
                time.sleep(15)  # give the GPU memory of the dead process time to be released
            results[name] = status

    if not args.no_eval:
        test2 = args.test2 or str(Path(args.data).parent / "data_moretest.yaml")
        for data, out in ((args.data, "test1_rawdata"), (test2, "test2_moretest")):
            if not Path(data).exists():
                log(f"skipping evaluation on {data}: file not found")
                continue
            log(f"=== evaluating all finished runs on {data}")
            status = run_watched([sys.executable, "-u", "evaluate.py", "--data", data, "--blur-csv", args.blur_csv,
                                  "--project", str(project), "--out", str(HERE / "results" / out),
                                  "--imgsz", str(args.imgsz), "--batch", str(args.batch), "--device", args.device],
                                 stall_s)
            log(f"=== evaluation on {Path(data).name}: {status.upper()}")

    log("summary: " + ", ".join(f"{k}={v}" for k, v in results.items()))
    log("all done")


if __name__ == "__main__":
    main()
