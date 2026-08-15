"""Create the still-image videos used to train CsF.

Each input image is rendered with one blendshape activated. The resulting
video repeats that frame and includes a silent audio track so it can pass
through Hallo's existing preprocessing pipeline. CsF later replaces the
preprocessed audio embedding with zeros during training.
"""

import argparse
import subprocess
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def create_video(
    image_path: Path,
    output_path: Path,
    duration: float,
    fps: int,
    overwrite: bool,
) -> None:
    """Create one fixed-frame MP4 with a mono silent audio stream."""
    command = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-loop", "1",
        "-framerate", str(fps),
        "-i", str(image_path),
        "-f", "lavfi",
        "-i", "anullsrc=channel_layout=mono:sample_rate=16000",
        "-t", str(duration),
        "-r", str(fps),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    """Parse arguments and convert all rendered images in lexical order."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="directory containing the rendered blendshape images")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="destination directory for the static MP4 files")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="duration of each video in seconds (default: 10)")
    parser.add_argument("--fps", type=int, default=25,
                        help="video frame rate (default: 25)")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace existing MP4 files")
    args = parser.parse_args()

    image_paths = sorted(
        path for path in args.input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        parser.error(f"no PNG or JPEG images found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, image_path in enumerate(image_paths):
        output_path = args.output_dir / f"{index:04d}.mp4"
        create_video(image_path, output_path, args.duration, args.fps,
                     args.overwrite)
        print(f"Created {output_path} from {image_path.name}")


if __name__ == "__main__":
    main()
