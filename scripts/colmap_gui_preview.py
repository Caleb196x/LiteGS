#!/usr/bin/env python3
"""
Launch COLMAP GUI to preview an existing reconstruction (database + sparse/0).

Example:
python scripts/colmap_gui_preview.py \
  --workspace /tmp/job123 \
  --images images \
  --sparse sparse/0

Notes:
- Expects COLMAP installed or bundled under thirdparty/litegs/tools/colmap.
- The GUI process stays attached; close the window to return.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import run_colmap  # type: ignore  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Open COLMAP GUI on an existing workspace")
    p.add_argument("--workspace", required=True, help="Folder containing database.db and sparse outputs")
    p.add_argument("--database", default="database.db", help="Database path relative to workspace (default: database.db)")
    p.add_argument("--images", default="images", help="Images folder relative to workspace (default: images)")
    p.add_argument("--sparse", default="sparse/0", help="Sparse model folder relative to workspace (default: sparse/0)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    workspace = Path(args.workspace).expanduser().resolve()
    db_path = (workspace / args.database).resolve()
    images = (workspace / args.images).resolve()
    sparse = (workspace / args.sparse).resolve()

    if not db_path.exists():
        print(f"[error] database not found: {db_path}", file=sys.stderr)
        return 1
    if not images.exists():
        print(f"[error] images folder not found: {images}", file=sys.stderr)
        return 1
    if not sparse.exists():
        print(f"[error] sparse model folder not found: {sparse}", file=sys.stderr)
        return 1

    colmap_bin = run_colmap.ensure_colmap()

    cmd = [
        colmap_bin,
        "gui",
        "--database_path",
        str(db_path),
        "--image_path",
        str(images),
        "--import_path",
        str(sparse),
    ]
    print(f"[colmap-gui] Running: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)
    proc.wait()
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
