"""Score every finished run on one test set: mAP, ball AP, and ball recall split into blurred / sharp.

    python evaluate.py --data /workspace/rawdata/data.yaml            # test 1: rawdata/test
    python evaluate.py --data /workspace/rawdata/data_moretest.yaml   # test 2: held-out game_A balls

run_all.py runs both after training. Blur strata come from blur_labels.csv (1,2 = blurred, 0 = sharp,
anything else or missing = other). A ball counts as found if a ball prediction with conf >= --conf
overlaps it with IoU >= --iou (one-to-one matching); 95% Wilson intervals are given. On the moretest
set only balls are labelled, so its mAP equals the ball AP.

Output: <out>/summary.csv (one row per run) and <out>/summary_by_method.csv (mean and std over seeds).
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def list_images(folder: str) -> list[str]:
    from ultralytics.data.utils import IMG_FORMATS

    return sorted(os.path.normpath(str(f)) for f in Path(folder).rglob("*.*") if f.suffix[1:].lower() in IMG_FORMATS)


def label_path(img: str) -> str:
    from ultralytics.data.utils import img2label_paths

    return img2label_paths([img])[0]


def parse_box(parts: list[str]) -> np.ndarray:
    v = np.array(parts[1:], dtype=np.float64)
    if len(v) == 4:
        return v
    xs, ys = v[0::2], v[1::2]  # polygon label -> enclosing box
    return np.array([(xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2, np.ptp(xs), np.ptp(ys)])


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def xywh2xyxy(b: np.ndarray) -> np.ndarray:
    return np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)


def match(gt: np.ndarray, pred: np.ndarray, conf: np.ndarray, thr: float) -> np.ndarray:
    """Greedy one-to-one matching, most confident prediction first. Returns a hit flag per GT box."""
    hit = np.zeros(len(gt), bool)
    if len(pred) == 0:
        return hit
    lt = np.maximum(pred[:, None, :2], gt[None, :, :2])
    rb = np.minimum(pred[:, None, 2:], gt[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area_p = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    area_g = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    iou = inter / (area_p[:, None] + area_g[None, :] - inter + 1e-12)
    for p in np.argsort(-conf):
        cand = np.flatnonzero(~hit & (iou[p] >= thr))
        if len(cand):
            hit[cand[np.argmax(iou[p, cand])]] = True
    return hit


def main():
    sys.stdout.reconfigure(line_buffering=True)
    base = Path("/workspace") if Path("/workspace/rawdata").exists() else Path.home() / "Desktop"
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(base / "rawdata" / "data.yaml"), help="its 'test' split is scored")
    p.add_argument("--blur-csv", default=str(base / "blur_labels.csv"))
    p.add_argument("--blur-positive", default="1,2", help="blur_label values that count as blurred")
    p.add_argument("--project", default=str(HERE / "runs"), help="folder holding the training runs")
    p.add_argument("--out", default=None, help="default: results/<data yaml name>")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--conf", type=float, default=0.25, help="operating point for the recall numbers")
    p.add_argument("--iou", type=float, default=0.5, help="IoU needed for a ball to count as found")
    p.add_argument("--ball-class", type=int, default=1)
    args = p.parse_args()

    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset

    runs = sorted(d for d in Path(args.project).glob("*")
                  if (d / "done.json").exists() and (d / "weights" / "best.pt").exists())
    if not runs:
        sys.exit(f"no finished runs (done.json + weights/best.pt) in {args.project}")
    out = Path(args.out) if args.out else HERE / "results" / Path(args.data).stem
    out.mkdir(parents=True, exist_ok=True)

    # ground-truth balls of the test split
    cfg = check_det_dataset(args.data, autodownload=False)
    images = list_images(cfg["test"])
    b_img, b_line, b_box = [], [], []
    for i, img in enumerate(images):
        lp = Path(label_path(img))
        if not lp.exists():
            continue
        for li, line in enumerate(lp.read_text().splitlines()):
            parts = line.split()
            if len(parts) >= 5 and int(float(parts[0])) == args.ball_class:
                b_img.append(i)
                b_line.append(li)
                b_box.append(parse_box(parts))
    b_box = np.asarray(b_box, np.float64).reshape(-1, 4)

    # blur strata from the manual labels, matched on (label file name, line index)
    table = {}
    if args.blur_csv and Path(args.blur_csv).exists():
        with open(args.blur_csv, newline="") as f:
            for r in csv.DictReader(f):
                table[(Path(r["label_file"].replace("\\", "/")).name, int(r["instance_idx"]))] = r["blur_label"].strip()
    names = [Path(label_path(img)).name for img in images]
    labels = np.array([table.get((names[i], li), "") for i, li in zip(b_img, b_line)])
    strata = np.where(np.isin(labels, args.blur_positive.split(",")), "blurred",
                      np.where(labels == "0", "sharp", "other"))
    log(f"[eval] {args.data}: {len(images)} test images, {len(b_img)} balls "
        f"(sharp {np.sum(strata == 'sharp')}, blurred {np.sum(strata == 'blurred')}, other/unlabelled "
        f"{np.sum(strata == 'other')}); {len(runs)} runs")
    by_img: dict[int, list[int]] = {}
    for j, i in enumerate(b_img):
        by_img.setdefault(i, []).append(j)
    img_ids = list(by_img)

    rows = []
    for run in runs:
        t0 = time.time()
        model = YOLO(str(run / "weights" / "best.pt"))
        m = model.val(data=args.data, split="test", imgsz=args.imgsz, batch=args.batch, device=args.device,
                      plots=False, verbose=False, project=str(out), name=f"{run.name}_val", exist_ok=True)
        ap_classes = list(m.box.ap_class_index)
        ball_ap50 = ball_ap = float("nan")
        if args.ball_class in ap_classes:
            _, _, ball_ap50, ball_ap = m.box.class_result(ap_classes.index(args.ball_class))

        # a list source is run by ultralytics as ONE batch, so feed fixed-size chunks
        hits = np.zeros(len(b_img), bool)
        for s in range(0, len(img_ids), args.batch):
            ids = img_ids[s:s + args.batch]
            results = model.predict([images[i] for i in ids], imgsz=args.imgsz, conf=args.conf, device=args.device,
                                    classes=[args.ball_class], verbose=False, save=False,
                                    project=str(out), name=f"{run.name}_predict", exist_ok=True)
            for i, r in zip(ids, results):
                js = np.asarray(by_img[i])
                hits[js] = match(xywh2xyxy(b_box[js]), r.boxes.xyxyn.cpu().numpy(), r.boxes.conf.cpu().numpy(), args.iou)

        row = {"run": run.name, "method": re.sub(r"_s\d+$", "", run.name), "mAP50": float(m.box.map50),
               "mAP50-95": float(m.box.map), "ball_AP50": float(ball_ap50), "ball_AP50-95": float(ball_ap)}
        for s, mask in (("all", np.ones(len(b_img), bool)), ("sharp", strata == "sharp"), ("blurred", strata == "blurred")):
            k, n = int(hits[mask].sum()), int(mask.sum())
            lo, hi = wilson(k, n)
            row.update({f"recall_{s}": k / n if n else float("nan"), f"recall_{s}_ci95": f"[{lo:.3f}, {hi:.3f}]",
                        f"n_{s}": n})
        rows.append(row)
        log(f"[eval] {run.name}: ball AP50 {row['ball_AP50']:.3f}, recall sharp {row['recall_sharp']:.3f}, "
            f"recall blurred {row['recall_blurred']:.3f} ({time.time() - t0:.0f} s)")

    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    keys = ["mAP50", "mAP50-95", "ball_AP50", "ball_AP50-95", "recall_all", "recall_sharp", "recall_blurred"]
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["method"], []).append(r)
    agg = []
    for method, g in groups.items():
        a = {"method": method, "n_seeds": len(g)}
        for k in keys:
            v = np.array([r[k] for r in g], float)
            a[k] = float(v.mean())
            a[f"{k}_std"] = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
        agg.append(a)
    with open(out / "summary_by_method.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0]))
        w.writeheader()
        w.writerows(agg)

    print(f"\nResults on {args.data} (recall at conf >= {args.conf}, IoU >= {args.iou})", flush=True)
    print(f"{'run':<26}{'mAP50':>7}{'mAP':>7}{'ballAP50':>9}{'ballAP':>8}  {'recall sharp':<24}{'recall blurred':<24}")
    for r in rows:
        print(f"{r['run']:<26}{r['mAP50']:7.3f}{r['mAP50-95']:7.3f}{r['ball_AP50']:9.3f}{r['ball_AP50-95']:8.3f}  "
              f"{r['recall_sharp']:.3f} {r['recall_sharp_ci95']:<18}{r['recall_blurred']:.3f} {r['recall_blurred_ci95']}")
    print(f"\nwritten: {out / 'summary.csv'} and {out / 'summary_by_method.csv'}", flush=True)


if __name__ == "__main__":
    main()
