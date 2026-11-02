"""[5/6] Repeat-factor sampling (RFS; Gupta et al., LVIS, CVPR 2019) over the two ball modes.

RFS gives every image the repeat factor r_i = max over its categories c of r_c, with
    r_c = max(1, sqrt(t / f_c)),   f_c = fraction of training images containing category c.
Here the categories are "sharp ball" and "blurred ball"; the other classes are not rebalanced, so
that, like the other methods, this one acts on the ball modes only. Images without a ball keep r = 1.

LVIS's t = 0.001 is meant for >1000 categories and would never fire for two modes, so t defaults to
f_sharp: sharp-only images stay at 1, blur-holding images get sqrt(f_sharp / f_blur). r_i is used
as a WeightedRandomSampler weight with the epoch kept at len(dataset) draws (same number of
iterations as the baseline).

"Blurred" = the blurriest --blur-quantile of the training balls by Laplacian variance, or the
manual labels with --blur-csv (run name gets "_oracle").

    python train_5_repeat_factor.py --batch 16

Output: runs/repeat_factor_s<seed>/
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
import torch

METHOD = "repeat_factor"
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
    p.add_argument("--rfs-t", type=float, default=None, help="RFS threshold t (default: frequency of the sharp mode)")
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
def path_key(p) -> str:
    """Normalised absolute path, used to match image lists against ultralytics' dataset.im_files."""
    return os.path.normcase(os.path.abspath(str(p)))


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


def weighted_trainer(image_weights: dict[str, float]):
    """DetectionTrainer whose training loader draws images with a WeightedRandomSampler.

    Unlisted images get weight 1. The epoch stays len(dataset) draws, so the run takes exactly as
    many iterations as the baseline. Single GPU only.
    """
    from torch.utils.data import WeightedRandomSampler
    from ultralytics.data.build import InfiniteDataLoader, seed_worker
    from ultralytics.models.yolo.detect import DetectionTrainer

    class WeightedSamplingTrainer(DetectionTrainer):
        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            if mode != "train":
                return super().get_dataloader(dataset_path, batch_size, rank, mode)
            if self.world_size > 1:
                raise RuntimeError("weighted sampling is implemented for single-GPU training only")
            dataset = self.build_dataset(dataset_path, mode, batch_size)
            keys = [path_key(f) for f in dataset.im_files]
            matched = sum(k in image_weights for k in keys)
            if matched != len(image_weights):
                raise RuntimeError(f"only {matched}/{len(image_weights)} weighted images found in the train set")
            weights = torch.tensor([image_weights.get(k, 1.0) for k in keys], dtype=torch.double)
            sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True,
                                            generator=torch.Generator().manual_seed(self.args.seed))
            nw = min(os.cpu_count() // max(torch.cuda.device_count(), 1), self.args.workers)
            log(f"[sampler] WeightedRandomSampler over {len(dataset)} images, {matched} reweighted, {nw} workers")
            return InfiniteDataLoader(
                dataset=dataset,
                batch_size=min(batch_size, len(dataset)),
                sampler=sampler,
                num_workers=nw,
                prefetch_factor=4 if nw > 0 else None,
                pin_memory=torch.cuda.is_available(),
                collate_fn=getattr(dataset, "collate_fn", None),
                worker_init_fn=seed_worker,
                generator=torch.Generator().manual_seed(6148914691236517205),  # as in ultralytics' own loader
            )

    return WeightedSamplingTrainer


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
    f_blur, f_sharp = float(has_blur.mean()), float(has_sharp.mean())
    t = args.rfs_t if args.rfs_t is not None else f_sharp

    def repeat(f: float) -> float:
        return max(1.0, math.sqrt(t / f)) if f > 0 else 1.0

    r_blur, r_sharp = repeat(f_blur), repeat(f_sharp)
    r_img = np.ones(len(images))
    r_img[has_sharp] = np.maximum(r_img[has_sharp], r_sharp)
    r_img[has_blur] = np.maximum(r_img[has_blur], r_blur)
    log(f"[{name}] f_blur={f_blur:.3f} f_sharp={f_sharp:.3f} t={t:.3f} -> r_blur={r_blur:.3f} r_sharp={r_sharp:.3f}; "
        f"share of draws for blur-holding images {has_blur.mean():.1%} -> {r_img[has_blur].sum() / r_img.sum():.1%}")
    weights = {path_key(images[i]): float(r_img[i]) for i in np.flatnonzero(r_img != 1.0)}
    info = {"t": t, "f_blur": f_blur, "f_sharp": f_sharp, "r_blur": r_blur, "r_sharp": r_sharp,
            "n_images_upweighted": len(weights),
            "minority": "manual labels" if args.blur_csv else f"Laplacian q={args.blur_quantile}"}

    log(f"[{name}] {args.model} imgsz={args.imgsz} batch={args.batch} epochs={args.epochs} workers={args.workers}")
    model = YOLO(args.model)
    add_progress_callbacks(model, name)
    model.train(
        data=args.data,
        trainer=weighted_trainer(weights),
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
    (save_dir / "done.json").write_text(json.dumps({"method": METHOD, "args": vars(args), **info}, indent=2))
    log(f"[{name}] finished -> {save_dir}")


if __name__ == "__main__":
    main()
