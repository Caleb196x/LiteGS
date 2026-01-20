#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Offline image screener for 3DGS (run before COLMAP).

Features
- Per-image metrics:
  - Resolution: long-side, megapixels
  - Sharpness: variance of Laplacian
  - Exposure: over/under-exposed ratios from luma
  - Contrast: luma standard deviation
  - EXIF risk (optional): ISO, exposure time
- Optional dynamic-region ratio between consecutive frames (ORB+Homography align + residual)
- Outputs:
  <output>/images/        kept images (copy or symlink)
  <output>/rejected/      rejected images (copy or symlink)
  <output>/report.json    detailed metrics + summary
  <output>/report.csv     tabular metrics
  <output>/kept_list.txt  kept image paths (absolute or relative)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ExifTags


# -------------------------
# Config
# -------------------------

@dataclass
class FilterConfig:
    # Resolution
    min_long_side_px: int = 1600
    min_total_mp: float = 2.0

    # Sharpness
    min_laplacian_var: float = 120.0  # 80~150 typical

    # Exposure / contrast
    overexposed_thresh: int = 250
    underexposed_thresh: int = 5
    max_overexposed_ratio: float = 0.02
    max_underexposed_ratio: float = 0.05
    min_luma_std: float = 20.0

    # EXIF risk gates (optional if EXIF exists)
    max_iso: int = 800
    min_shutter_1_over_sec: float = 60.0  # shutter >= 1/60s

    # Dynamic check (optional, sequential images recommended)
    enable_dynamic_check: bool = False
    max_dynamic_ratio: float = 0.20

    # IO
    mode: str = "symlink"  # symlink or copy
    recursive: bool = True
    exif: bool = True
    write_absolute_list: bool = False

    # Speed
    num_workers: int = 0  # 0 => auto (cpu count - 1), 1 => single process

    # Supported extensions
    exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp")


@dataclass
class ImageMetrics:
    path: str
    width: int
    height: int
    total_mp: float
    laplacian_var: float
    overexposed_ratio: float
    underexposed_ratio: float
    luma_std: float
    iso: Optional[int] = None
    shutter_sec: Optional[float] = None
    dynamic_ratio: Optional[float] = None
    passed_precheck: bool = False
    reject_reasons: List[str] = field(default_factory=list)


@dataclass
class Summary:
    total: int
    kept: int
    rejected: int
    reject_reason_counts: Dict[str, int]
    config: Dict[str, Any]


# -------------------------
# Utilities
# -------------------------

def _safe_imread(path: Path) -> np.ndarray:
    """
    Robust read supporting unicode paths by decoding bytes.
    Returns BGR uint8 image.
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def _to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _luma_Y(bgr: np.ndarray) -> np.ndarray:
    # Use Y from YCrCb (close enough and fast)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    Y = ycrcb[:, :, 0]
    return Y


def variance_of_laplacian(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def exposure_stats(Y: np.ndarray, over_thresh: int, under_thresh: int) -> Tuple[float, float, float]:
    total = Y.size
    over = float(np.count_nonzero(Y >= over_thresh)) / total
    under = float(np.count_nonzero(Y <= under_thresh)) / total
    std = float(np.std(Y))
    return over, under, std


def _rational_to_float(x: Any) -> Optional[float]:
    # PIL may give (num, den) tuples or Rational objects
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, tuple) and len(x) == 2:
            num, den = x
            den = float(den)
            if den == 0:
                return None
            return float(num) / den
        # Some PIL versions: IFDRational supports float()
        return float(x)
    except Exception:
        return None


def read_exif_iso_shutter(path: Path) -> Tuple[Optional[int], Optional[float]]:
    """
    Returns (ISO, shutter_seconds). None if unavailable.
    """
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            if not exif:
                return None, None

            # Map tag IDs to names (once)
            # ISO: ISOSpeedRatings (34855)
            # ExposureTime: (33434) seconds as rational
            iso = None
            shutter_sec = None

            iso_val = exif.get(34855, None)
            if iso_val is not None:
                # Sometimes ISO is list-like
                if isinstance(iso_val, (list, tuple)) and len(iso_val) > 0:
                    iso_val = iso_val[0]
                try:
                    iso = int(iso_val)
                except Exception:
                    iso = None

            exp_time = exif.get(33434, None)
            shutter_sec = _rational_to_float(exp_time)

            return iso, shutter_sec
    except Exception:
        return None, None


# -------------------------
# Dynamic region estimator (optional)
# -------------------------

def compute_dynamic_ratio(prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> float:
    """
    Estimate how much of the image is not explained by global motion
    by aligning prev -> curr with homography (ORB+RANSAC), then measuring residual.

    Returns ratio in [0, 1].
    """
    prev_gray = _to_gray(prev_bgr)
    curr_gray = _to_gray(curr_bgr)

    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)

    if des1 is None or des2 is None or len(kp1) < 50 or len(kp2) < 50:
        # Not enough features -> fall back to raw diff (crude)
        diff = cv2.absdiff(prev_gray, curr_gray)
        _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        return float(np.count_nonzero(mask)) / mask.size

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)[:500]

    if len(matches) < 30:
        diff = cv2.absdiff(prev_gray, curr_gray)
        _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        return float(np.count_nonzero(mask)) / mask.size

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    H, inliers = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=3.0)
    if H is None:
        diff = cv2.absdiff(prev_gray, curr_gray)
        _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        return float(np.count_nonzero(mask)) / mask.size

    aligned = cv2.warpPerspective(prev_gray, H, (curr_gray.shape[1], curr_gray.shape[0]))
    diff = cv2.absdiff(aligned, curr_gray)

    # Adaptive-ish threshold: base + a bit of noise
    thr = 20 + 0.5 * float(np.std(diff))
    thr = float(np.clip(thr, 15, 40))

    _, mask = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)

    # Clean small noise blobs
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return float(np.count_nonzero(mask)) / mask.size


# -------------------------
# Precheck
# -------------------------

def precheck_one(path: Path, cfg: FilterConfig) -> ImageMetrics:
    bgr = _safe_imread(path)
    h, w = bgr.shape[:2]
    long_side = max(w, h)
    total_mp = (w * h) / 1e6

    gray = _to_gray(bgr)
    lap_var = variance_of_laplacian(gray)

    Y = _luma_Y(bgr)
    over, under, ystd = exposure_stats(Y, cfg.overexposed_thresh, cfg.underexposed_thresh)

    iso = None
    shutter = None
    if cfg.exif:
        iso, shutter = read_exif_iso_shutter(path)

    reasons: List[str] = []

    # Resolution gates
    if long_side < cfg.min_long_side_px:
        reasons.append("resolution_long_side_too_small")
    if total_mp < cfg.min_total_mp:
        reasons.append("resolution_total_mp_too_small")

    # Sharpness
    if lap_var < cfg.min_laplacian_var:
        reasons.append("blurry_low_laplacian_var")

    # Exposure/contrast
    if over > cfg.max_overexposed_ratio:
        reasons.append("overexposed_too_much")
    if under > cfg.max_underexposed_ratio:
        reasons.append("underexposed_too_much")
    if ystd < cfg.min_luma_std:
        reasons.append("low_contrast_or_flat_lighting")

    # EXIF risk (only if present)
    if iso is not None and iso > cfg.max_iso:
        reasons.append("high_iso_noise_risk")
    if shutter is not None and shutter > 0:
        if (1.0 / shutter) < cfg.min_shutter_1_over_sec:
            reasons.append("slow_shutter_motion_blur_risk")

    m = ImageMetrics(
        path=str(path),
        width=w,
        height=h,
        total_mp=float(total_mp),
        laplacian_var=float(lap_var),
        overexposed_ratio=float(over),
        underexposed_ratio=float(under),
        luma_std=float(ystd),
        iso=iso,
        shutter_sec=shutter,
        passed_precheck=(len(reasons) == 0),
        reject_reasons=reasons,
    )
    return m


def list_images(input_dir: Path, cfg: FilterConfig) -> List[Path]:
    if cfg.recursive:
        paths = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in cfg.exts]
    else:
        paths = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in cfg.exts]
    # Stable order = helpful for dynamic check + reproducibility
    paths.sort(key=lambda p: str(p).lower())
    return paths


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    _ensure_dir(dst.parent)
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        try:
            os.symlink(src.resolve(), dst)
        except FileExistsError:
            pass
    else:
        raise ValueError(f"Unknown mode: {mode}")


def write_reports(
    out_dir: Path,
    kept: List[ImageMetrics],
    rejected: List[ImageMetrics],
    cfg: FilterConfig,
    input_dir: Path,
) -> None:
    reason_counts: Dict[str, int] = {}
    for m in rejected:
        for r in m.reject_reasons:
            reason_counts[r] = reason_counts.get(r, 0) + 1

    summary = Summary(
        total=len(kept) + len(rejected),
        kept=len(kept),
        rejected=len(rejected),
        reject_reason_counts=reason_counts,
        config=asdict(cfg),
    )

    report = {
        "summary": asdict(summary),
        "kept": [asdict(m) for m in kept],
        "rejected": [asdict(m) for m in rejected],
    }

    # JSON
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # CSV (flatten)
    csv_path = out_dir / "report.csv"
    fields = [
        "path", "width", "height", "total_mp",
        "laplacian_var", "overexposed_ratio", "underexposed_ratio", "luma_std",
        "iso", "shutter_sec", "dynamic_ratio",
        "passed_precheck", "reject_reasons",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for m in kept + rejected:
            row = asdict(m)
            row["reject_reasons"] = ";".join(m.reject_reasons)
            w.writerow({k: row.get(k, "") for k in fields})

    # kept list for COLMAP
    list_path = out_dir / "kept_list.txt"
    with list_path.open("w", encoding="utf-8") as f:
        for m in kept:
            p = Path(m.path)
            if cfg.write_absolute_list:
                f.write(str(p.resolve()) + "\n")
            else:
                # relative to input_dir when possible
                try:
                    f.write(str(p.relative_to(input_dir)) + "\n")
                except Exception:
                    f.write(str(p) + "\n")


def print_summary(kept: List[ImageMetrics], rejected: List[ImageMetrics]) -> None:
    print("\n=== Screening summary ===")
    total = len(kept) + len(rejected)
    print(f"Total: {total}")
    print(f"Kept: {len(kept)}")
    print(f"Rejected: {len(rejected)}")

    reason_counts: Dict[str, int] = {}
    for m in rejected:
        for r in m.reject_reasons:
            reason_counts[r] = reason_counts.get(r, 0) + 1

    if reason_counts:
        print("\nTop reject reasons:")
        for k, v in sorted(reason_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]:
            print(f"  - {k}: {v}")


# -------------------------
# Main pipeline
# -------------------------

def run_filter(input_dir: Path, out_dir: Path, cfg: FilterConfig) -> Tuple[List[ImageMetrics], List[ImageMetrics]]:
    _ensure_dir(out_dir)
    kept_dir = out_dir / "images"
    rej_dir = out_dir / "rejected"
    _ensure_dir(kept_dir)
    _ensure_dir(rej_dir)

    paths = list_images(input_dir, cfg)
    if not paths:
        raise RuntimeError(f"No images found under: {input_dir}")

    # Precheck (no multiprocessing if dynamic_check is on, because we need sequential prev frame anyway)
    metrics: List[ImageMetrics] = []
    if cfg.enable_dynamic_check:
        prev_bgr = None
        prev_path = None
        for p in paths:
            m = precheck_one(p, cfg)
            # dynamic ratio computed regardless of precheck pass, because it can become a reject reason
            if prev_bgr is not None:
                try:
                    curr_bgr = _safe_imread(p)
                    dyn = compute_dynamic_ratio(prev_bgr, curr_bgr)
                    m.dynamic_ratio = float(dyn)
                    if dyn > cfg.max_dynamic_ratio:
                        m.reject_reasons.append("dynamic_objects_too_much")
                        m.passed_precheck = False
                    prev_bgr = curr_bgr
                except Exception:
                    # if dynamic calc fails, just keep pipeline moving
                    pass
            else:
                try:
                    prev_bgr = _safe_imread(p)
                except Exception:
                    prev_bgr = None
            prev_path = p
            # update passed flag after potential dynamic reason
            m.passed_precheck = (len(m.reject_reasons) == 0)
            metrics.append(m)
    else:
        # optionally parallelize
        if cfg.num_workers is None:
            cfg.num_workers = 0
        if cfg.num_workers == 1:
            metrics = [precheck_one(p, cfg) for p in paths]
        else:
            # Lazy import to avoid overhead when not needed
            import multiprocessing as mp

            workers = cfg.num_workers
            if workers <= 0:
                workers = max(1, (os.cpu_count() or 2) - 1)

            with mp.Pool(processes=workers) as pool:
                metrics = pool.starmap(precheck_one, [(p, cfg) for p in paths])

    kept: List[ImageMetrics] = []
    rejected: List[ImageMetrics] = []
    for m in metrics:
        src = Path(m.path)
        # Keep folder structure relative to input_dir (nice for datasets with subfolders)
        try:
            rel = src.relative_to(input_dir)
        except Exception:
            rel = Path(src.name)

        if m.passed_precheck:
            kept.append(m)
            dst = kept_dir / rel
        else:
            rejected.append(m)
            dst = rej_dir / rel

        _link_or_copy(src, dst, cfg.mode)

    write_reports(out_dir, kept, rejected, cfg, input_dir)
    return kept, rejected


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Offline high-quality image screener for 3DGS (run before COLMAP).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, type=str, help="Input image directory")
    p.add_argument("--output", required=True, type=str, help="Output directory (filtered dataset + reports)")
    p.add_argument("--mode", choices=["symlink", "copy"], default="symlink", help="How to populate output folders")
    p.add_argument("--no-recursive", action="store_true", help="Do not scan subfolders")
    p.add_argument("--no-exif", action="store_true", help="Do not read EXIF ISO/shutter")
    p.add_argument("--abs-list", action="store_true", help="Write kept_list.txt with absolute paths")

    # Threshold overrides
    p.add_argument("--min-long-side", type=int, default=None)
    p.add_argument("--min-mp", type=float, default=None)
    p.add_argument("--min-lapvar", type=float, default=None)
    p.add_argument("--max-over", type=float, default=None)
    p.add_argument("--max-under", type=float, default=None)
    p.add_argument("--min-luma-std", type=float, default=None)
    p.add_argument("--max-iso", type=int, default=None)
    p.add_argument("--min-shutter-inv", type=float, default=None)

    # Dynamic check
    p.add_argument("--dynamic", action="store_true", help="Enable dynamic-region check (sequential images recommended)")
    p.add_argument("--max-dynamic", type=float, default=None)

    # Speed
    p.add_argument("--workers", type=int, default=0, help="0=auto, 1=single process, N=multiprocessing")

    # Config file
    p.add_argument("--config", type=str, default=None, help="Optional JSON config file to override defaults")

    return p


def load_config(args: argparse.Namespace) -> FilterConfig:
    cfg = FilterConfig()

    # JSON config override
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    # CLI overrides
    cfg.mode = args.mode
    cfg.recursive = not args.no_recursive
    cfg.exif = not args.no_exif
    cfg.write_absolute_list = bool(args.abs_list)
    cfg.enable_dynamic_check = bool(args.dynamic)
    cfg.num_workers = int(args.workers)

    if args.min_long_side is not None:
        cfg.min_long_side_px = int(args.min_long_side)
    if args.min_mp is not None:
        cfg.min_total_mp = float(args.min_mp)
    if args.min_lapvar is not None:
        cfg.min_laplacian_var = float(args.min_lapvar)
    if args.max_over is not None:
        cfg.max_overexposed_ratio = float(args.max_over)
    if args.max_under is not None:
        cfg.max_underexposed_ratio = float(args.max_under)
    if args.min_luma_std is not None:
        cfg.min_luma_std = float(args.min_luma_std)
    if args.max_iso is not None:
        cfg.max_iso = int(args.max_iso)
    if args.min_shutter_inv is not None:
        cfg.min_shutter_1_over_sec = float(args.min_shutter_inv)
    if args.max_dynamic is not None:
        cfg.max_dynamic_ratio = float(args.max_dynamic)

    return cfg


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    out_dir = Path(args.output).expanduser().resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[ERROR] input is not a directory: {input_dir}", file=sys.stderr)
        return 2

    cfg = load_config(args)

    try:
        kept, rejected = run_filter(input_dir, out_dir, cfg)
        print_summary(kept, rejected)
        print(f"\nOutputs written to: {out_dir}")
        print(f"COLMAP should use image_dir: {out_dir / 'images'}")
        return 0
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
