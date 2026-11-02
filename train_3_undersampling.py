"""[3/6] Random undersampling (RUS) of the majority (sharp-ball) mode.

Training images fall into three groups:
    blur    - holds at least one blurred ball            -> all kept
    sharp   - holds balls, none of them blurred          -> randomly undersampled
    no ball - background for the ball class              -> all kept
Sharp-only images are dropped at random until #sharp : #blur = --rus-ratio. The other classes on the
dropped images are lost too, which is inherent to image-level undersampling, and the epoch is
shorter than the baseline's because the dataset is smaller.

"Blurred" = the blurriest --blur-quantile of the training balls by Laplacian variance (the same
label-free signal the coreset method uses), or the manual labels with --blur-csv (run name gets
"_oracle").

    python train_3_undersampling.py --batch 16

Output: runs/undersampling_s<seed>/ and work/undersampling_s<seed>/train.txt (kept images)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

METHOD = "undersampling"
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
    p.add_argument("--ball-class", type=int, default=1)
    p.add_argument("--margin", type=float, default=0.2, help="context margin per side, fraction of box width/height")
    p.add_argument("--blur-quantile", type=float, default=0.2, help="blurriest fraction of balls = minority")
    p.add_argument("--blur-csv", default=None, help="use manual blur labels instead of the Laplacian proxy")
    p.add_argument("--blur-positive", default="1,2", help="blur_label values that count as blurred")
    p.add_argument("--rus-ratio", type=float, default=1.0, help="kept sharp-only images per blur-holding image")
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


# ----------------------------------------------------------------------------- data
def list_images(folder: str) -> list[str]:
    from ultralytics.data.utils import IMG_FORMATS

    return sorted(os.path.normpath(str(f)) for f in Path(folder).rglob("*.*") if f.suffix[1:].lower() in IMG_FORMATS)


def label_path(img: str) -> str:
    from ultralytics.data.utils import img2label_paths

    return img2label_paths([img])[0]


def imread(path: str) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cannot read {path}")
    return img


def parse_box(parts: list[str]) -> np.ndarray:
    v = np.array(parts[1:], dtype=np.float64)
    if len(v) == 4:
        return v
    xs, ys = v[0::2], v[1::2]  # polygon label -> enclosing box
    return np.array([(xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2, np.ptp(xs), np.ptp(ys)])


def context_box(box: np.ndarray, w: int, h: int, margin: float) -> tuple[int, int, int, int]:
    """Normalised xywh box -> pixel xyxy padded by `margin` x box size on every side, clipped."""
    xc, yc, bw, bh = box[0] * w, box[1] * h, box[2] * w, box[3] * h
    hw, hh = bw * (0.5 + margin), bh * (0.5 + margin)
    x1, y1 = max(0, math.floor(xc - hw)), max(0, math.floor(yc - hh))
    x2, y2 = min(w, math.ceil(xc + hw)), min(h, math.ceil(yc + hh))
    return x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)


def collect_balls(images: list[str], ball_class: int, margin: float):
    """Every ball box in the training labels: image index, line index and Laplacian variance of its crop."""
    img_idx, line_idx, lap = [], [], []
    for i, img_path in enumerate(images):
        lp = Path(label_path(img_path))
        if lp.exists():
            rows = [(li, ln.split()) for li, ln in enumerate(lp.read_text().splitlines())]
            rows = [(li, p) for li, p in rows if len(p) >= 5 and int(float(p[0])) == ball_class]
            if rows:
                img = imread(img_path)
                H, W = img.shape[:2]
                for li, parts in rows:
                    x1, y1, x2, y2 = context_box(parse_box(parts), W, H, margin)
                    gray = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                    img_idx.append(i)
                    line_idx.append(li)
                    lap.append(cv2.Laplacian(gray, cv2.CV_64F).var())
        if (i + 1) % 200 == 0 or i + 1 == len(images):
            log(f"[balls] {i + 1}/{len(images)} images read, {len(img_idx)} balls")
    return np.asarray(img_idx, np.int64), np.asarray(line_idx, np.int64), np.asarray(lap, np.float64)


def manual_blur_labels(csv_path: str, images: list[str], img_idx: np.ndarray, line_idx: np.ndarray) -> np.ndarray:
    """blur_label per ball from blur_labels.csv, matched on (label file name, line index); '' if missing."""
    table = {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            table[(Path(r["label_file"].replace("\\", "/")).name, int(r["instance_idx"]))] = r["blur_label"].strip()
    names = [Path(label_path(p)).name for p in images]
    return np.array([table.get((names[i], int(li)), "") for i, li in zip(img_idx, line_idx)])


def blurred_mask(args, images, img_idx, line_idx, lap) -> np.ndarray:
    if args.blur_csv:
        labels = manual_blur_labels(args.blur_csv, images, img_idx, line_idx)
        coverage = float(np.mean(labels != ""))
        if coverage < 0.9:
            sys.exit(f"--blur-csv covers only {coverage:.0%} of the training balls; wrong file?")
        mask = np.isin(labels, args.blur_positive.split(","))
        log(f"[minority] manual labels: {mask.sum()}/{len(mask)} training balls are blurred ({mask.mean():.1%})")
    else:
        thr = np.quantile(lap, args.blur_quantile)
        mask = lap <= thr
        log(f"[minority] Laplacian variance <= {thr:.1f}: {mask.sum()}/{len(mask)} balls count as blurred")
    return mask


def main():
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    args = parse_args()
    name = f"{METHOD}{'_oracle' if args.blur_csv else ''}_s{args.seed}"
    fresh_run_dir(args, name)

    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset

    cfg = check_det_dataset(args.data, autodownload=False)
    images = list_images(cfg["train"])
    log(f"[{name}] {len(images)} training images in {cfg['train']}")
    img_idx, line_idx, lap = collect_balls(images, args.ball_class, args.margin)
    blurred = blurred_mask(args, images, img_idx, line_idx, lap)

    has_blur = np.zeros(len(images), bool)
    has_sharp = np.zeros(len(images), bool)
    has_blur[img_idx[blurred]] = True
    has_sharp[img_idx[~blurred]] = True
    blur_imgs = np.flatnonzero(has_blur)
    sharp_only = np.flatnonzero(has_sharp & ~has_blur)
    no_ball = np.flatnonzero(~has_sharp & ~has_blur)
    n_keep = min(len(sharp_only), round(args.rus_ratio * len(blur_imgs)))
    kept_sharp = np.random.default_rng(args.seed).choice(sharp_only, size=n_keep, replace=False)
    kept = np.sort(np.concatenate([blur_imgs, kept_sharp, no_ball]))
    log(f"[{name}] blur {len(blur_imgs)} | sharp-only {len(sharp_only)} -> {n_keep} | no ball {len(no_ball)} "
        f"=> training on {len(kept)}/{len(images)} images")

    work = HERE / "work" / name
    work.mkdir(parents=True, exist_ok=True)
    train_txt = work / "train.txt"
    train_txt.write_text("\n".join(images[i] for i in kept) + "\n")
    data_yaml = work / "data.yaml"
    data = {"train": str(train_txt), "val": cfg["val"], "nc": cfg["nc"], "names": cfg["names"]}
    if cfg.get("test"):
        data["test"] = cfg["test"]
    data_yaml.write_text(yaml.safe_dump(data, sort_keys=False))
    info = {"n_images": len(images), "n_kept": int(len(kept)), "n_blur_images": int(len(blur_imgs)),
            "n_sharp_only": int(len(sharp_only)), "n_sharp_only_kept": int(n_keep), "n_no_ball": int(len(no_ball)),
            "minority": "manual labels" if args.blur_csv else f"Laplacian q={args.blur_quantile}"}

    log(f"[{name}] {args.model} imgsz={args.imgsz} batch={args.batch} epochs={args.epochs} workers={args.workers}")
    model = YOLO(args.model)
    add_progress_callbacks(model, name)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        seed=args.seed,
        deterministic=True,
        optimizer="SGD",  # pinned: 'auto' picks SGD/AdamW from the dataset size, which this method changes
        patience=args.epochs,  # no early stopping: every method trains for the same number of epochs
        project=args.project,
        name=name,
        exist_ok=True,
    )
    save_dir = Path(model.trainer.save_dir)
    (save_dir / "done.json").write_text(json.dumps({"method": METHOD, "args": vars(args), **info}, indent=2))
    log(f"[{name}] finished -> {save_dir}")


if __name__ == "__main__":
    main()
