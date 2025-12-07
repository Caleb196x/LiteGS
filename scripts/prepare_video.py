"""
将单个视频转成 LiteGS 训练所需的帧目录结构。
输出目录形如：
<output_root>/
  images/
    frame_00001.png
    frame_00002.png
    ...

用法示例：
python prepare_video.py --video /path/to/video.mp4 --out data/myvideo --fps 10 --overwrite
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import shutil


def ensure_ffmpeg() -> str:
    """确保能找到 ffmpeg 可执行路径。"""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("未找到 ffmpeg，请先安装并加入 PATH。")
    return ffmpeg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将视频抽帧为 LiteGS 输入")
    parser.add_argument("--video", required=True, help="输入视频路径")
    parser.add_argument("--out", required=True, help="输出根目录（会创建 images 子目录）")
    parser.add_argument("--fps", type=int, default=10, help="抽帧帧率，默认 10")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的输出帧")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ffmpeg = ensure_ffmpeg()

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        print(f"输入视频不存在: {video_path}", file=sys.stderr)
        return 1

    out_root = Path(args.out).expanduser().resolve()
    images_dir = out_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if not args.overwrite:
        existing = list(images_dir.glob("frame_*.png"))
        if existing:
            print(f"输出目录已包含帧文件，使用 --overwrite 以覆盖: {images_dir}", file=sys.stderr)
            return 1

    pattern = images_dir / "frame_%05d.png"
    cmd = [
        ffmpeg,
        "-y" if args.overwrite else "-n",
        "-i",
        str(video_path),
        "-vf",
        f"fps={args.fps}",
        "-pix_fmt",
        "rgb24",
        str(pattern),
        "-hide_banner",
        "-loglevel",
        "error",
    ]

    print(f"抽帧命令: {' '.join(cmd)}")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        print("ffmpeg 抽帧失败:", file=sys.stderr)
        if proc.stdout:
            print(proc.stdout, file=sys.stderr)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        return proc.returncode

    frames = sorted(images_dir.glob("frame_*.png"))
    if not frames:
        print("未生成任何帧，请检查输入视频/参数。", file=sys.stderr)
        return 1

    print(f"完成，生成帧数: {len(frames)}，输出目录: {images_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
