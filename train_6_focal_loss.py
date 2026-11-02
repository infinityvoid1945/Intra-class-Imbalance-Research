"""[6/6] Focal loss (Lin et al., ICCV 2017): the loss-reweighting baseline.

Unlike methods 2-5 the data is untouched; the BCE classification term of YOLO's loss is replaced by
sigmoid focal loss, which down-weights easy examples so hard ones (such as blurred balls) count for
relatively more. It needs no notion of "blurred", but it modifies the loss, which the coreset
method avoids.

gamma=1.5, alpha=0.25 match ultralytics' own FocalLoss defaults. Focal loss is numerically much
smaller than BCE, so the training cls loss is not on the baseline's scale; validation losses are
computed with standard BCE on the EMA model and stay comparable. The focal criterion is attached
only to the live model after the EMA copy is made, and checkpoints store the EMA, so best.pt is a
plain YOLO checkpoint.

    python train_6_focal_loss.py --batch 16

Output: runs/focal_loss_s<seed>/
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

METHOD = "focal_loss"
HERE = Path(__file__).resolve().parent


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def fmt_time(seconds: float) -> str:
    s = int(max(seconds, 0))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def parse_args():
    base = Path("/workspace") if Path("/workspace/rawdata").exists() else Path.home() / "Desktop"
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(base / "rawdata" / "data.yaml"))
    p.add_argument("--model", default="yolo11x.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=8, help="8 for a 24 GB GPU, 16 for 48 GB+; same for every method")
    p.add_argument("--workers", type=int, default=8, help="data-loader processes")
    p.add_argument("--device", default="0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--project", default=str(HERE / "runs"))
    p.add_argument("--overwrite", action="store_true", help="retrain even if this run already finished")
    p.add_argument("--fl-gamma", type=float, default=1.5)
    p.add_argument("--fl-alpha", type=float, default=0.25, help="negative value disables alpha balancing")
    return p.parse_args()


def fresh_run_dir(args, name: str) -> None:
    run_dir = Path(args.project) / name
    if (run_dir / "done.json").exists() and not args.overwrite:
        log(f"{run_dir} already finished; pass --overwrite to train it again")
        sys.exit(0)
    shutil.rmtree(run_dir, ignore_errors=True)  # leftovers of an interrupted attempt


def add_progress_callbacks(model, tag: str, every_s: float = 30.0) -> None:
    """Print a progress line at least every `every_s` seconds and one line per finished epoch.

    Ultralytics hides its own progress bar on RunPod (it treats RUNPOD_POD_ID as a non-interactive
    console and prints only when an epoch ends), which makes a long epoch look frozen.
    """
    st = {"start": None, "epoch_start": time.time(), "last": 0.0, "batch": 0, "logged_epoch": -1}

    def on_train_start(trainer):
        st["start"] = time.time()
        log(f"[{tag}] training started: {trainer.epochs} epochs x {len(trainer.train_loader)} batches per epoch")

    def on_train_epoch_start(trainer):
        st["epoch_start"] = st["last"] = time.time()
        st["batch"] = 0
        log(f"[{tag}] epoch {trainer.epoch + 1}/{trainer.epochs} started")

    def on_train_batch_end(trainer):
        st["batch"] += 1
        now, nb, b = time.time(), len(trainer.train_loader), st["batch"]
        if now - st["last"] < every_s and b < nb:
            return
        st["last"] = now
        rate = b / max(now - st["epoch_start"], 1e-9)
        losses = ""
        if trainer.tloss is not None:
            values = trainer.tloss.tolist() if trainer.tloss.dim() else [trainer.tloss.item()]
            losses = " ".join(f"{n.replace('_loss', '')}={v:.3f}" for n, v in zip(trainer.loss_names, values))
        log(f"[{tag}] epoch {trainer.epoch + 1}/{trainer.epochs} batch {b}/{nb} ({100 * b // nb}%) {losses} "
            f"| {rate:.2f} batch/s | epoch ETA {fmt_time((nb - b) / rate)}")

    def on_fit_epoch_end(trainer):
        m = trainer.metrics or {}
        maps = (f"mAP50={float(m.get('metrics/mAP50(B)', float('nan'))):.3f} "
                f"mAP50-95={float(m.get('metrics/mAP50-95(B)', float('nan'))):.3f}")
        if st["logged_epoch"] == trainer.epoch:  # second call for the same epoch = final validation of best.pt
            log(f"[{tag}] final validation of best.pt: {maps}")
            return
        st["logged_epoch"] = trainer.epoch
        done, now = trainer.epoch + 1, time.time()
        spent = now - (st["start"] or st["epoch_start"])
        log(f"[{tag}] epoch {done}/{trainer.epochs} finished in {fmt_time(now - st['epoch_start'])} | val {maps} "
            f"| total {fmt_time(spent)}, about {fmt_time(spent / done * (trainer.epochs - done))} left")

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_train_batch_end", on_train_batch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)


class FocalBCE(torch.nn.Module):
    """Element-wise sigmoid focal loss that also accepts the soft IoU-aware targets of YOLO's
    task-aligned assigner. Drop-in replacement for v8DetectionLoss.bce."""

    def __init__(self, gamma: float, alpha: float):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred, target = pred.float(), target.float()
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        p = pred.sigmoid()
        p_t = target * p + (1 - target) * (1 - p)
        loss = loss * (1 - p_t).pow(self.gamma)
        if self.alpha >= 0:
            loss = loss * (target * self.alpha + (1 - target) * (1 - self.alpha))
        return loss


def focal_loss_callback(gamma: float, alpha: float):
    """on_pretrain_routine_end: runs after the EMA copy exists, so only the live model gets focal loss."""

    def attach(trainer):
        from ultralytics.utils.loss import v8DetectionLoss
        from ultralytics.utils.torch_utils import unwrap_model

        model = unwrap_model(trainer.model)
        if getattr(model, "end2end", False):
            raise RuntimeError("focal-loss swap expects the standard (non end-to-end) YOLO11 head")
        criterion = v8DetectionLoss(model)
        criterion.bce = FocalBCE(gamma, alpha)
        model.criterion = criterion
        log(f"[focal] classification BCE replaced by focal loss (gamma={gamma}, alpha={alpha})")

    return attach


def main():
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    args = parse_args()
    name = f"{METHOD}_s{args.seed}"
    fresh_run_dir(args, name)

    from ultralytics import YOLO

    log(f"[{name}] {args.model} imgsz={args.imgsz} batch={args.batch} epochs={args.epochs} "
        f"workers={args.workers} data={args.data}")
    model = YOLO(args.model)
    add_progress_callbacks(model, name)
    model.add_callback("on_pretrain_routine_end", focal_loss_callback(args.fl_gamma, args.fl_alpha))
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        seed=args.seed,
        deterministic=True,
        optimizer="SGD",  # pinned: 'auto' picks SGD/AdamW from the dataset size, which other methods change
        patience=args.epochs,  # no early stopping: every method trains for the same number of epochs
        project=args.project,
        name=name,
        exist_ok=True,
    )
    save_dir = Path(model.trainer.save_dir)
    info = {"fl_gamma": args.fl_gamma, "fl_alpha": args.fl_alpha}
    (save_dir / "done.json").write_text(json.dumps({"method": METHOD, "args": vars(args), **info}, indent=2))
    log(f"[{name}] finished -> {save_dir}")


if __name__ == "__main__":
    main()
