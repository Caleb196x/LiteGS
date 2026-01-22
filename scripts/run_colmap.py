"""
Automate COLMAP to produce sparse/0 from a folder of frames for LiteGS.

Example (PowerShell/CMD):
python run_colmap.py ^
  --images F:/GS/GSServer/tmp/job123/images ^
  --out F:/GS/GSServer/tmp/job123 ^
  --matcher sequential ^
  --single_camera

Layout produced:
<out>/
  database.db
  images/            # input frames
  sparse/0/          # COLMAP output (bin + txt)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path


def ensure_colmap() -> str:
    """
    Resolve a COLMAP executable. Prefer the bundled tools/colmap if present,
    otherwise fall back to PATH.
    """
    # prefer bundled
    here = Path(__file__).resolve().parents[1]  # .../litegs
    bundled = here / "tools" / "colmap"
    candidates = [
        bundled / "COLMAP.bat",
        bundled / "bin" / "colmap.exe",
        bundled / "bin" / "colmap",
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    colmap = shutil.which("colmap")
    if not colmap:
        raise FileNotFoundError("colmap not found (checked tools/colmap and PATH)")
    return colmap

def ensure_glomap() -> str:
    """
    Resolve a GLOMAP executable. Prefer the bundled tools/glomap if present,
    otherwise fall back to PATH.
    """
    # prefer bundled
    here = Path(__file__).resolve().parents[1]  # .../litegs
    bundled = here / "tools" / "glomap"
    candidates = [
        bundled / "bin" / "glomap.exe",
        bundled / "bin" / "glomap",
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    glomap = shutil.which("glomap")
    if not glomap:
        raise FileNotFoundError("glomap not found (checked tools/colmap and PATH)")
    return glomap

def run_cmd(cmd: list[str], cwd: Path | None = None, label: str | None = None) -> float:
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}Running: {' '.join(cmd)}")
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"{prefix}Command failed with exit code {proc.returncode} after {elapsed:.2f}s")
    print(f"[{prefix}]: Done [{elapsed:.2f}s]")
    return elapsed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run COLMAP to generate sparse/0")
    p.add_argument("--images", required=True, help="Folder containing extracted frames")
    p.add_argument("--out", required=True, help="Output root (database.db and sparse/0 will be written here)")
    p.add_argument("--matcher", choices=["exhaustive", "sequential"], default="sequential", help="Matching strategy")
    p.add_argument("--single_camera", action="store_true", help="Treat all frames as one camera (video-friendly)")
    p.add_argument("--threads", type=int, default=16, help="Mapper thread count")
    p.add_argument("--camera_model", default="OPENCV", help="COLMAP camera model (e.g., PINHOLE, SIMPLE_RADIAL)")
    p.add_argument("--use_gpu", type=int, choices=[0, 1], default=1,
                   help="Use GPU for SIFT (default: 1)")
    return p.parse_args()


def main() -> int:
    total_start = time.perf_counter()
    args = parse_args()
    colmap = ensure_colmap()
    glomap = ensure_glomap()

    images = Path(args.images).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()
    if not images.exists():
        print(f"Images folder not found: {images}", file=sys.stderr)
        return 1
    out_root.mkdir(parents=True, exist_ok=True)
    sparse_dir = out_root / "sparse"
    sparse_dir.mkdir(exist_ok=True)

    db_path = out_root / "database.db"

    feat_cmd = [
        colmap,
        "feature_extractor",
        "--database_path",
        str(db_path),
        "--image_path",
        str(images),
    ]
    if args.single_camera:
        feat_cmd += ["--ImageReader.single_camera", "1"]
    if args.camera_model:
        feat_cmd += ["--ImageReader.camera_model", args.camera_model]
    run_cmd(feat_cmd, label="feature_extractor")

    if args.matcher == "sequential":
        match_cmd = [colmap, "sequential_matcher", 
                     "--database_path", str(db_path)]
    else:
        match_cmd = [colmap, "exhaustive_matcher", 
                     "--database_path", str(db_path)]
    run_cmd(match_cmd, label="matcher")

    mapper_cmd = [
        glomap,
        "mapper",
        "--database_path",
        str(db_path),
        "--image_path",
        str(images),
        "--output_path",
        str(sparse_dir),
        # "--Mapper.num_threads",
        # str(args.threads),
        # "--Mapper.ba_global_function_tolerance",
        # "0.000001",
    ]
    run_cmd(mapper_cmd, label="mapper")

    model_dir = sparse_dir / "0"
    if model_dir.exists():
        converter_cmd = [
            colmap,
            "model_converter",
            "--input_path",
            str(model_dir),
            "--output_path",
            str(model_dir),
            "--output_type",
            "TXT",
        ]
        try:
            run_cmd(converter_cmd, label="model_converter")
        except Exception as e:
            print(f"model_converter failed (bin is still usable): {e}", file=sys.stderr)

    print(f"Done. Sparse model at: {model_dir}")
    total_elapsed = time.perf_counter() - total_start
    print(f"Total time: {total_elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
