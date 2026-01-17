#!/usr/bin/env python3
"""
End-to-end helper: frames -> COLMAP/GLOMAP -> LiteGS 3DGS reconstruction.

Example:
python scripts/frames_to_3dgs.py \
  --frames /data/frames \
  --workspace /tmp/job123 \
  --output_model /tmp/job123/output \
  --matcher sequential --single_camera \
  --sh_degree 3 --resolution -1

Outputs: <model_path>/point_cloud/finish/point_cloud.ply
Note: Training hyperparameters mirror example_train.py; source_path/images/model_path
are set automatically to the workspace layout produced here.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# Allow importing litegs and sibling scripts when run from repo root or this folder
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import run_colmap  # type: ignore  # noqa: E402
import litegs  # type: ignore  # noqa: E402
import litegs.config  # type: ignore  # noqa: E402
import litegs.arguments  # type: ignore  # noqa: E402
import litegs.training  # type: ignore  # noqa: E402


def stage_images(frames_dir: Path, staged_dir: Path, reuse: bool, link: bool) -> None:
    """
    Ensure images live under the workspace so COLMAP and LiteGS see consistent paths.
    Uses hardlinks when possible to avoid duplicate storage.
    """
    if reuse and staged_dir.exists():
        existing = sum(1 for _ in staged_dir.iterdir())
        if existing > 0:
            print(f"[stage] Reusing staged images at {staged_dir} ({existing} files)")
            return

    staged_dir.mkdir(parents=True, exist_ok=True)
    exts = {".jpg", ".jpeg", ".png"}
    copied = 0
    for src in sorted(frames_dir.iterdir()):
        if not src.is_file() or src.suffix.lower() not in exts:
            continue
        dst = staged_dir / src.name
        if dst.exists():
            continue
        if link:
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        else:
            shutil.copy2(src, dst)
        copied += 1

    total = sum(1 for _ in staged_dir.iterdir())
    if total == 0:
        raise RuntimeError(f"No images with extensions {sorted(exts)} found in {frames_dir}")
    print(f"[stage] Staged {copied} files (total {total}) into {staged_dir}")


def run_colmap_pipeline(images: Path, workspace: Path, args: argparse.Namespace) -> Path:
    """
    Run feature extraction + matching + mapping.
    Skips if sparse/0 already exists unless --rerun_colmap is set.
    """
    db_path = workspace / "database.db"
    sparse_dir = workspace / "sparse"
    model_dir = sparse_dir / "0"

    if model_dir.exists() and not args.rerun_colmap:
        print(f"[colmap] Found existing model at {model_dir}, skipping COLMAP stage.")
        return model_dir

    if args.rerun_colmap:
        if db_path.exists():
            db_path.unlink()
        if sparse_dir.exists():
            shutil.rmtree(sparse_dir)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    colmap_bin = run_colmap.ensure_colmap()
    glomap_bin = run_colmap.ensure_glomap()

    feat_cmd = [
        colmap_bin,
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
    # feat_cmd += ["--SiftExtraction.use_gpu", str(args.use_gpu)]
    # feat_cmd += ["--SiftExtraction.num_threads", str(args.threads)]
    run_colmap.run_cmd(feat_cmd, label="feature_extractor")

    match_cmd = [
        colmap_bin,
        "sequential_matcher" if args.matcher == "sequential" else "exhaustive_matcher",
        "--database_path",
        str(db_path),
    ]
    # match_cmd += ["--SiftMatching.use_gpu", str(args.use_gpu)]
    run_colmap.run_cmd(match_cmd, label="matcher")

    mapper_cmd = [
        glomap_bin,
        "mapper",
        "--database_path",
        str(db_path),
        "--image_path",
        str(images),
        "--output_path",
        str(sparse_dir),
    ]
    run_colmap.run_cmd(mapper_cmd, label="mapper")

    if model_dir.exists():
        converter_cmd = [
            colmap_bin,
            "model_converter",
            "--input_path",
            str(model_dir),
            "--output_path",
            str(model_dir),
            "--output_type",
            "TXT",
        ]
        try:
            run_colmap.run_cmd(converter_cmd, label="model_converter")
        except Exception as exc:  # noqa: BLE001
            print(f"[colmap] model_converter failed (binary model still usable): {exc}")

    return model_dir


def build_training_params(args: argparse.Namespace, workspace: Path, model_path: Path, image_dir: str):
    lp = litegs.arguments.ModelParams.extract(args)
    op = litegs.arguments.OptimizationParams.extract(args)
    pp = litegs.arguments.PipelineParams.extract(args)
    dp = litegs.arguments.DensifyParams.extract(args)

    lp.source_path = str(workspace)
    lp.images = image_dir
    lp.model_path = str(model_path)
    return lp, op, pp, dp


METRICS_PATTERN = re.compile(r"\b(SSIM|PSNR|LPIPS)\s*:?\s*([+-]?\d+(?:\.\d+)?)")


def parse_metrics_output(output: str) -> dict:
    metrics_by_label: dict[str, dict] = {}
    current_label = ""
    for line in output.splitlines():
        trimmed = line.strip()
        if not trimmed:
            continue
        label = parse_scene_label(trimmed)
        if label:
            current_label = label
            continue
        match = METRICS_PATTERN.search(trimmed)
        if not match or not current_label:
            continue
        metric_name = match.group(1)
        value = float(match.group(2))
        entry = metrics_by_label.setdefault(current_label, {})
        if metric_name == "SSIM":
            entry["ssim_metrics"] = value
        elif metric_name == "PSNR":
            entry["psnr_metrics"] = value
        elif metric_name == "LPIPS":
            entry["lpip_metrics"] = value

    if not metrics_by_label:
        raise RuntimeError("No metrics parsed from metrics output.")
    for label, entry in metrics_by_label.items():
        missing = {"ssim_metrics", "psnr_metrics", "lpip_metrics"} - entry.keys()
        if missing:
            raise RuntimeError(f"Incomplete metrics for {label}: missing {sorted(missing)}")

    result: dict[str, dict] = {}
    if "Trainingset" in metrics_by_label:
        result["train"] = metrics_by_label["Trainingset"]
    if "Testset" in metrics_by_label:
        result["test"] = metrics_by_label["Testset"]
    return result


def parse_scene_label(line: str) -> str:
    if "Scene:" not in line:
        return ""
    _, _, rest = line.partition("Scene:")
    rest = rest.strip()
    if not rest:
        return ""
    parts = rest.rsplit(None, 1)
    if len(parts) != 2:
        return ""
    label = parts[1]
    if label not in {"Trainingset", "Testset"}:
        return ""
    return label


def run_metrics(args: argparse.Namespace, workspace: Path, model_path: Path, image_dir_name: str) -> dict:
    metrics_script = ROOT / "example_metrics.py"
    cmd = [
        sys.executable,
        str(metrics_script),
        "-s",
        str(workspace),
        "-m",
        str(model_path),
        "-i",
        image_dir_name,
        "--sh_degree",
        str(args.sh_degree),
        "--resolution",
        str(args.resolution),
        "--cluster_size",
        str(args.cluster_size),
        "--eval",
    ]
    if args.white_background:
        cmd.append("--white_background")
    if args.learnable_viewproj:
        cmd.append("--learnable_viewproj")
    if args.input_color_type:
        cmd += ["--input_color_type", str(args.input_color_type)]

    print(f"[metrics] Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        if output.strip():
            print(output)
        raise RuntimeError(f"[metrics] example_metrics.py failed with exit code {proc.returncode}")

    metrics = parse_metrics_output(output)
    metrics_path = model_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[metrics] Saved metrics to {metrics_path}")
    return metrics


def parse_args() -> argparse.Namespace:
    lp_cdo, op_cdo, pp_cdo, dp_cdo = litegs.config.get_default_arg()
    parser = argparse.ArgumentParser(description="Frames -> COLMAP/GLOMAP -> LiteGS 3DGS reconstruction")
    parser.add_argument("--frames", required=True, help="Folder containing RGB frames (.jpg/.png)")
    parser.add_argument("--workspace", required=True, help="Working directory (staged images, database.db, sparse/)")
    parser.add_argument("--output_model", default=None, help="Where to save LiteGS outputs (default: <workspace>/output)")
    parser.add_argument("--image_dir_name", default="images", help="Subfolder under workspace to stage images")
    parser.add_argument("--link_images", action="store_true", help="Hardlink images instead of copying when possible")
    parser.add_argument("--reuse_images", action="store_true", help="Skip staging if images already exist in workspace")
    parser.add_argument("--skip_colmap", action="store_true", help="Assume sparse/0 already exists; do not run COLMAP")
    parser.add_argument("--rerun_colmap", action="store_true", help="Force rerun COLMAP even if sparse/0 exists")
    parser.add_argument("--matcher", choices=["exhaustive", "sequential"], default="sequential", help="COLMAP matcher")
    parser.add_argument("--single_camera", action="store_true", help="Treat frames as one camera (video-friendly)")
    parser.add_argument("--camera_model", default="PINHOLE", help="Camera model passed to COLMAP ImageReader")
    parser.add_argument("--use_gpu", type=int, choices=[0, 1], default=1, help="Whether COLMAP SIFT stages use GPU")
    parser.add_argument("--threads", type=int, default=16, help="Thread count for COLMAP SIFT stages")

    litegs.arguments.ModelParams.add_cmdline_arg(lp_cdo, parser)
    litegs.arguments.OptimizationParams.add_cmdline_arg(op_cdo, parser)
    litegs.arguments.PipelineParams.add_cmdline_arg(pp_cdo, parser)
    litegs.arguments.DensifyParams.add_cmdline_arg(dp_cdo, parser)

    parser.add_argument("--test_epochs", nargs="+", type=int, default=[], help="Evaluation epochs (see example_train.py)")
    parser.add_argument("--save_epochs", nargs="+", type=int, default=[], help="Save PLY at specific epochs")
    parser.add_argument("--checkpoint_epochs", nargs="+", type=int, default=[], help="Save checkpoints at epochs")
    parser.add_argument("--start_checkpoint", type=str, default=None, help="Resume from checkpoint")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    frames_dir = Path(args.frames).expanduser().resolve()
    workspace = Path(args.workspace).expanduser().resolve()
    image_dir_name = args.image_dir_name
    model_path = Path(args.output_model).expanduser().resolve() if args.output_model else workspace / "output"
    staged_images = workspace / image_dir_name

    if not frames_dir.exists():
        print(f"[error] Frames folder not found: {frames_dir}", file=sys.stderr)
        return 1

    workspace.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)

    stage_images(frames_dir, staged_images, reuse=args.reuse_images, link=args.link_images)

    if not args.skip_colmap:
        run_colmap_pipeline(staged_images, workspace, args)
    else:
        print("[colmap] Skipping COLMAP per --skip_colmap (ensure sparse/0 exists).")

    lp, op, pp, dp = build_training_params(args, workspace, model_path, image_dir_name)
    litegs.training.start(lp, op, pp, dp, args.test_epochs, args.save_epochs, args.checkpoint_epochs, args.start_checkpoint)
    metrics = run_metrics(args, workspace, model_path, image_dir_name)
    print(f"[metrics] Results: {metrics}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
