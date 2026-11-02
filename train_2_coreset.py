"""[2/6] Greedy k-center coreset-boosted sampling (the proposed method).

  1. crop every training ball with a 20% context margin on each side
  2. frozen ImageNet ResNet-18 (classifier removed) -> 512-d L2-normalised embedding, plus the
     Laplacian variance of the crop as an explicit blur score: x_i = [e_i, beta * z_i]
  3. greedy k-center (Gonzalez, 1985): start from a random point, repeatedly add the point farthest
     from its nearest selected center; 2-approximation of min_{|S|=k} max_i min_{j in S} ||x_i - x_j||
  4. every training image that holds a selected ball is drawn alpha times as often by a
     WeightedRandomSampler; the full dataset is still used and the epoch length is unchanged

No blur labels are used for the selection. blur_labels.csv (if found) is only read to report how
many of the selected balls are blurred.

    python train_2_coreset.py --batch 16
    python train_2_coreset.py --batch 16 --selection random   # ablation: same k and alpha, random balls

Output: runs/coreset_s<seed>/ and work/coreset_s<seed>/selected_balls.csv
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

METHOD = "coreset"
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
    p.add_argument("--coreset-frac", type=float, default=0.10, help="k as a fraction of the training balls")
    p.add_argument("--alpha", type=float, default=3.0, help="sampling weight of images that hold a coreset ball")
    p.add_argument("--blur-weight", type=float, default=1.0,
                   help="variance of the blur axis relative to the total variance of the 512 embedding dims")
    p.add_argument("--selection", choices=["kcenter", "random"], default="kcenter")
    p.add_argument("--report-csv", default=str(base / "blur_labels.csv"), help="manual blur labels, reporting only")
    p.add_argument("--blur-positive", default="1,2", help="blur_label values that count as blurred")
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


# ----------------------------------------------------------------------------- coreset
@torch.no_grad()
def greedy_k_center(x: np.ndarray, k: int, seed: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Farthest-point traversal. Time O(nkd), extra memory O(n). Returns the selected indices in
    selection order and the covering radius max_i min_{j in S} ||x_i - x_j|| after each addition."""
    pts = torch.from_numpy(x).to(device)
    first = int(np.random.default_rng(seed).integers(len(pts)))
    selected = [first]
    min_dist = torch.linalg.vector_norm(pts - pts[first], dim=1)
    radius = [min_dist.max().item()]
    while len(selected) < k and radius[-1] > 0:
        nxt = int(torch.argmax(min_dist))
        selected.append(nxt)
        min_dist = torch.minimum(min_dist, torch.linalg.vector_norm(pts - pts[nxt], dim=1))
        radius.append(min_dist.max().item())
    return np.asarray(selected), np.asarray(radius)


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
    name = f"{METHOD}{'_random' if args.selection == 'random' else ''}_s{args.seed}"
    fresh_run_dir(args, name)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    cfg = check_det_dataset(args.data, autodownload=False)
    images = list_images(cfg["train"])
    log(f"[{name}] {len(images)} training images in {cfg['train']}")
    img_idx, line_idx, boxes, lap, crops = collect_balls(images, args.ball_class, args.margin)
    n = len(img_idx)

    # features: [embedding, beta * z_blur]; beta scales the blur axis relative to the embedding's own spread
    emb = resnet18_embed(crops, device)
    z = -np.log1p(lap)  # higher = blurrier
    z = (z - z.mean()) / (z.std() + 1e-12)
    beta = args.blur_weight * float(np.sqrt(emb.var(axis=0).sum()))
    x = np.hstack([emb, beta * z[:, None]]).astype(np.float32)

    k = max(1, round(args.coreset_frac * n))
    if args.selection == "kcenter":
        sel, radius = greedy_k_center(x, k, args.seed, device)
        log(f"[{name}] greedy k-center: n={n} balls, d={x.shape[1]}, k={len(sel)}, beta={beta:.4f}, "
            f"covering radius {radius[0]:.3f} -> {radius[-1]:.3f}")
    else:
        sel = np.random.default_rng(args.seed).choice(n, size=k, replace=False)
        radius = np.full(len(sel), np.nan)
        log(f"[{name}] random selection (ablation): n={n} balls, k={k}")

    in_s = np.zeros(n, bool)
    in_s[sel] = True
    info = {"n_balls": n, "k": int(len(sel)), "alpha": args.alpha, "beta": beta, "selection": args.selection}
    manual = np.full(n, "")
    if args.report_csv and Path(args.report_csv).exists():
        manual = manual_blur_labels(args.report_csv, images, img_idx, line_idx)
        if np.mean(manual != "") >= 0.9:
            blur = np.isin(manual, args.blur_positive.split(","))
            info.update(blurred_share_all=float(blur.mean()), blurred_share_coreset=float(blur[in_s].mean()))
            log(f"[{name}] manually-labelled blurred balls: {blur.mean():.1%} of all, {blur[in_s].mean():.1%} "
                f"of the selected (x{blur[in_s].mean() / max(blur.mean(), 1e-12):.2f})")

    coreset_images = sorted(set(img_idx[sel].tolist()))
    weights = {path_key(images[i]): args.alpha for i in coreset_images}
    N, m = len(images), len(coreset_images)
    share = args.alpha * m / (args.alpha * m + N - m)
    info.update(n_images=N, n_upweighted_images=m, expected_draw_share=share)
    log(f"[{name}] {m}/{N} images upweighted x{args.alpha}: their share of draws {m / N:.1%} -> {share:.1%}")

    work = HERE / "work" / name
    work.mkdir(parents=True, exist_ok=True)
    with open(work / "selected_balls.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["order", "image", "line_idx", "x", "y", "w", "h", "laplacian_var", "manual_blur_label", "radius_after"])
        for order, (i, r) in enumerate(zip(sel, radius)):
            w.writerow([order, images[img_idx[i]], line_idx[i], *boxes[i], lap[i], manual[i], r])
    del crops, emb, x
    if device.type == "cuda":
        torch.cuda.empty_cache()

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
    (save_dir / "done.json").write_text(json.dumps({"method": METHOD, "args": vars(args), **info}, indent=2, default=str))
    log(f"[{name}] finished -> {save_dir}")


if __name__ == "__main__":
    main()
