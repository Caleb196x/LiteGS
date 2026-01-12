"""
Inspect COLMAP point cloud data and print a quick summary.

Example:
  python scripts/view_colmap_pointcloud.py --colmap /path/to/colmap/out
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from litegs.io_manager import colmap as colmap_io
except ImportError:
    # Allow running directly from this file by adding repo root to sys.path.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from litegs.io_manager import colmap as colmap_io


def resolve_colmap_root(path: Path) -> Path:
    if (path / "sparse" / "0").exists():
        return path
    if path.name == "0" and path.parent.name == "sparse":
        return path.parent.parent
    if path.name == "sparse" and (path / "0").exists():
        return path.parent
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="View COLMAP point cloud summary")
    p.add_argument("--colmap", required=True, help="COLMAP output root (contains sparse/0)")
    p.add_argument("--sample", type=int, default=0, help="Print first N points (xyz rgb)")
    p.add_argument("--precision", type=int, default=5, help="Float precision in output")
    p.add_argument("--export_xyz", default="", help="Export XYZRGB to a text file")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = resolve_colmap_root(Path(args.colmap).expanduser().resolve())
    sparse_dir = root / "sparse" / "0"
    if not sparse_dir.exists():
        print(f"sparse/0 not found under: {root}", file=sys.stderr)
        return 1

    xyz, rgb = colmap_io.load_pointcloud(str(root))
    if xyz.size == 0:
        print("No points loaded.", file=sys.stderr)
        return 1

    np.set_printoptions(precision=args.precision, suppress=True)
    count = xyz.shape[0]
    bbox_min = xyz.min(axis=0)
    bbox_max = xyz.max(axis=0)

    ply_path = sparse_dir / "points3D.ply"

    print(f"COLMAP root: {root}")
    print(f"Point count: {count}")
    print(f"Bounds min: {bbox_min}")
    print(f"Bounds max: {bbox_max}")
    print(f"PLY path  : {ply_path}")

    if args.sample > 0:
        n = min(args.sample, count)
        sample_xyz = xyz[:n]
        sample_rgb = (rgb[:n] * 255.0).clip(0, 255).astype(np.uint8)
        print("Sample points (x y z r g b):")
        for i in range(n):
            x, y, z = sample_xyz[i]
            r, g, b = sample_rgb[i]
            print(f"{x:.{args.precision}f} {y:.{args.precision}f} {z:.{args.precision}f} {r} {g} {b}")

    if args.export_xyz:
        out_path = Path(args.export_xyz).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rgb_255 = (rgb * 255.0).clip(0, 255)
        data = np.hstack([xyz, rgb_255])
        np.savetxt(out_path, data, fmt="%.6f %.6f %.6f %.0f %.0f %.0f")
        print(f"Exported XYZRGB: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
