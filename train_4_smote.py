"""[4/6] SMOTE adapted to object detection (image-space interpolation of blurred balls).

Classic SMOTE (Chawla et al., 2002) synthesises minority samples between a minority sample and one
of its k nearest minority neighbours: x_new = x_a + lam * (x_b - x_a). A detector has no feature
vector that could be interpolated without changing its architecture, so it is done on the image:

  * minority    = blurred training balls (Laplacian proxy, or manual labels with --blur-csv)
  * neighbours  = k nearest minority balls in ResNet-18 embedding space, excluding the same
                  source frame (Roboflow augmented copies would be near-duplicates)
  * synthesis   = copy ball a's image; inside a's context crop, blend in ball b's crop (resized to
                  a's crop) with weight lam ~ U(0,1) under a feathered mask so no rectangular seam
                  appears that the detector could learn as a shortcut
  * label       = a's label file unchanged (b's ball lines up with a's box through the crop)

The synthetic images are trained on together with the full original set, so the epoch is longer
than the baseline's. --smote-percent 100 makes one synthetic image per minority ball, as in the
original paper.

    python train_4_smote.py --batch 16

Output: runs/smote_s<seed>/ and work/smote_s<seed>/train_synthetic/
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
import torch.nn.functional as F
import yaml

METHOD = "smote"
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
    p.add_argument("--smote-percent", type=float, default=100, help="synthetic images as %% of the minority count")
    p.add_argument("--smote-k", type=int, default=5, help="nearest minority neighbours to interpolate towards")
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


def imwrite(path: Path, img: np.ndarray) -> None:
    cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(str(path))


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
    """Every ball box in the training labels with its Laplacian variance and context crop."""
    img_idx, line_idx, boxes, lap, crops = [], [], [], [], []
    for i, img_path in enumerate(images):
        lp = Path(label_path(img_path))
        if lp.exists():
            rows = [(li, ln.split()) for li, ln in enumerate(lp.read_text().splitlines())]
            rows = [(li, p) for li, p in rows if len(p) >= 5 and int(float(p[0])) == ball_class]
            if rows:
                img = imread(img_path)
                H, W = img.shape[:2]
                for li, parts in rows:
                    box = parse_box(parts)
                    x1, y1, x2, y2 = context_box(box, W, H, margin)
                    crop = img[y1:y2, x1:x2].copy()
                    img_idx.append(i)
                    line_idx.append(li)
                    boxes.append(box)
                    lap.append(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
                    crops.append(crop)
        if (i + 1) % 200 == 0 or i + 1 == len(images):
            log(f"[balls] {i + 1}/{len(images)} images read, {len(img_idx)} balls")
    return (np.asarray(img_idx, np.int64), np.asarray(line_idx, np.int64),
            np.asarray(boxes, np.float64).reshape(-1, 4), np.asarray(lap, np.float64), crops)


@torch.no_grad()
def resnet18_embed(crops: list[np.ndarray], device: torch.device, batch: int = 128) -> np.ndarray:
    """Frozen ImageNet ResNet-18 with the classifier removed: BGR crop -> 512-d unit vector."""
    from torchvision.models import ResNet18_Weights, resnet18

    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()
    net = net.eval().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    out = []
    for s in range(0, len(crops), batch):
        x = np.stack([cv2.resize(c, (224, 224), interpolation=cv2.INTER_LINEAR)[..., ::-1] for c in crops[s:s + batch]])
        x = torch.from_numpy(x.copy()).to(device).permute(0, 3, 1, 2).float().div_(255)
        out.append(F.normalize(net((x - mean) / std), dim=1).cpu().numpy())
        log(f"[embed] {min(s + batch, len(crops))}/{len(crops)} crops")
    del net
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(out).astype(np.float32)


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


def feather_mask(crop_hw: tuple[int, int], inner_xyxy: tuple[int, int, int, int], sigma: float) -> np.ndarray:
    """1 on the tight ball box, falling smoothly to ~0 at the edge of the context crop."""
    mask = np.zeros(crop_hw, np.float32)
    x1, y1, x2, y2 = inner_xyxy
    mask[y1:y2, x1:x2] = 1.0
    return cv2.GaussianBlur(mask, (0, 0), sigmaX=max(sigma, 0.5))


def main():
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    args = parse_args()
    name = f"{METHOD}{'_oracle' if args.blur_csv else ''}_s{args.seed}"
    fresh_run_dir(args, name)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    cfg = check_det_dataset(args.data, autodownload=False)
    images = list_images(cfg["train"])
    log(f"[{name}] {len(images)} training images in {cfg['train']}")
    img_idx, line_idx, boxes, lap, crops = collect_balls(images, args.ball_class, args.margin)
    minority = np.flatnonzero(blurred_mask(args, images, img_idx, line_idx, lap))
    m = len(minority)
    if m < 2:
        sys.exit("need at least two minority balls for SMOTE")

    # k nearest minority neighbours (cosine == euclidean on unit vectors), from other source frames only
    emb = torch.from_numpy(resnet18_embed([crops[i] for i in minority], device))
    del crops
    sim = emb @ emb.T
    src = np.array([Path(images[img_idx[i]]).stem.split(".rf.")[0] for i in minority])
    sim[torch.from_numpy(src[:, None] == src[None, :])] = -float("inf")  # also removes each ball itself
    k = min(args.smote_k, m - 1)
    nn_sim, nn_idx = sim.topk(k, dim=1)
    valid, nn_idx = torch.isfinite(nn_sim).numpy(), nn_idx.numpy()

    n_syn = round(m * args.smote_percent / 100)
    bases = np.concatenate([rng.permutation(m) for _ in range(-(-n_syn // m))])[:n_syn]
    work = HERE / "work" / name
    syn = work / "train_synthetic"
    shutil.rmtree(syn, ignore_errors=True)
    (syn / "images").mkdir(parents=True)
    (syn / "labels").mkdir(parents=True)

    written = skipped = 0
    for t, ai in enumerate(bases):
        choices = nn_idx[ai][valid[ai]]
        if len(choices) == 0:
            skipped += 1
            continue
        a, b = minority[ai], minority[rng.choice(choices)]
        lam = rng.uniform()
        img_a_path = images[img_idx[a]]
        img_a, img_b = imread(img_a_path), imread(images[img_idx[b]])
        H, W = img_a.shape[:2]
        x1, y1, x2, y2 = context_box(boxes[a], W, H, args.margin)
        tx1, ty1, tx2, ty2 = context_box(boxes[a], W, H, 0.0)
        bx1, by1, bx2, by2 = context_box(boxes[b], img_b.shape[1], img_b.shape[0], args.margin)
        patch_b = cv2.resize(img_b[by1:by2, bx1:bx2], (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
        sigma = 0.5 * args.margin * max(tx2 - tx1, ty2 - ty1)
        wmap = lam * feather_mask((y2 - y1, x2 - x1), (tx1 - x1, ty1 - y1, tx2 - x1, ty2 - y1), sigma)[..., None]
        region = img_a[y1:y2, x1:x2].astype(np.float32)
        img_a[y1:y2, x1:x2] = np.clip(region * (1 - wmap) + patch_b.astype(np.float32) * wmap + 0.5, 0, 255).astype(np.uint8)
        stem = f"smote_{t:06d}_{Path(img_a_path).stem}"
        imwrite(syn / "images" / f"{stem}.jpg", img_a)
        shutil.copyfile(label_path(img_a_path), syn / "labels" / f"{stem}.txt")
        written += 1
        if written % 50 == 0:
            log(f"[smote] {written}/{n_syn} synthetic images written")
    log(f"[{name}] {m} minority balls, k={k}: {written} synthetic images in {syn}"
        + (f" ({skipped} skipped: no neighbour from another frame)" if skipped else ""))

    data_yaml = work / "data.yaml"
    data = {"train": [str(cfg["train"]), str(syn / "images")], "val": cfg["val"], "nc": cfg["nc"], "names": cfg["names"]}
    if cfg.get("test"):
        data["test"] = cfg["test"]
    data_yaml.write_text(yaml.safe_dump(data, sort_keys=False))
    info = {"n_minority": int(m), "n_synthetic": written, "smote_k": k, "n_images": len(images),
            "minority": "manual labels" if args.blur_csv else f"Laplacian q={args.blur_quantile}"}
    if device.type == "cuda":
        torch.cuda.empty_cache()

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
