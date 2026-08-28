#!/usr/bin/env python3
"""
Polished high-resolution Gource renderer.

Requirements:
  - gource >= 0.50
  - ffmpeg with libx264

Examples:
  python render_gource.py
  python render_gource.py --duration 180
  python render_gource.py --resolution 2560x1440 --duration 240
  python render_gource.py --resolution 3840x2160 --fps 60 --crf 18

The script inspects the complete Git history, derives a playback speed,
renders Gource to a raw PPM stream, and encodes it to MP4 with FFmpeg.
"""

import argparse
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def run(*args, cwd=None):
    return subprocess.check_output(
        args, cwd=cwd, text=True, stderr=subprocess.DEVNULL
    ).strip()


def require_program(name):
    if shutil.which(name) is None:
        sys.exit(f"Required program not found: {name}")


def repo_root():
    try:
        return Path(run("git", "rev-parse", "--show-toplevel")).resolve()
    except subprocess.CalledProcessError:
        sys.exit("Run this script from inside a Git repository.")


def git_metadata(repo):
    try:
        first = int(
            run("git", "log", "--reverse", "--format=%at", "-1", "--all", cwd=repo)
        )
        last = int(run("git", "log", "--format=%at", "-1", "--all", cwd=repo))
        commits = int(run("git", "rev-list", "--count", "--all", cwd=repo))
        emails = run("git", "log", "--all", "--format=%ae", cwd=repo).splitlines()
        authors = len(set(e for e in emails if e))
        branches = len(
            run(
                "git", "for-each-ref", "--format=%(refname)", "refs/heads/", cwd=repo
            ).splitlines()
        )
    except (subprocess.CalledProcessError, ValueError):
        sys.exit("Could not read Git history.")

    return first, last, commits, authors, branches


def main():
    parser = argparse.ArgumentParser(
        description="Render a polished high-resolution Gource video."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=180,
        help="Target video duration in seconds (default: 180).",
    )
    parser.add_argument(
        "--resolution",
        default="3840x2160",
        help="WIDTHxHEIGHT (default: 3840x2160 / 4K).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        choices=(25, 30, 60),
        default=60,
        help="Output frame rate (default: 60).",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="H.264 quality, lower is better (default: 18).",
    )
    parser.add_argument(
        "--output", default=None, help="Output MP4 filename (default: gource.mp4)."
    )
    args = parser.parse_args()

    for program in ("git", "gource", "ffmpeg"):
        require_program(program)

    repo = repo_root()

    try:
        width, height = map(int, args.resolution.lower().split("x"))
        if width <= 0 or height <= 0:
            raise ValueError
    except ValueError:
        sys.exit("--resolution must be WIDTHxHEIGHT, e.g. 3840x2160.")

    if args.duration <= 0:
        sys.exit("--duration must be greater than zero.")

    first, last, commits, authors, branches = git_metadata(repo)

    history_days = max((last - first) / 86400.0, 1.0)

    # Compress the complete repository lifetime into the requested runtime.
    seconds_per_day = args.duration / history_days

    # Keep very small/large values within a sensible Gource range.
    seconds_per_day = max(0.01, min(seconds_per_day, 2.0))

    first_date = datetime.fromtimestamp(first, UTC).strftime("%Y-%m-%d")
    last_date = datetime.fromtimestamp(last, UTC).strftime("%Y-%m-%d")

    output = Path(args.output) if args.output else repo / "gource.mp4"
    if not output.is_absolute():
        output = repo / output

    print()
    print("=" * 68)
    print(f"  GOURCE RENDER — {repo.name}")
    print("=" * 68)
    print(f"  Repository : {repo}")
    print(f"  History    : {first_date} → {last_date}")
    print(f"  Lifetime   : {history_days:,.0f} days")
    print(f"  Commits    : {commits:,}")
    print(f"  Authors    : {authors:,}")
    print(f"  Branches   : {branches:,}")
    print(f"  Resolution : {width}×{height}")
    print(f"  Frame rate : {args.fps} fps")
    print(f"  Target     : {args.duration:.0f} seconds")
    print(f"  Speed      : {seconds_per_day:.5f} seconds/day")
    print(f"  Output     : {output}")
    print("=" * 68)
    print()

    # IMPORTANT:
    # Gource does NOT have a generic "--bloom" switch.
    # Bloom is enabled by default; intensity/multiplier tune it.
    #
    # Also, --output-ppm-stream automatically stops at the end of the
    # repository unless overridden. This is the supported recording mode.
    gource_cmd = [
        "gource",
        str(repo),
        # Timeline
        "--seconds-per-day",
        f"{seconds_per_day:.8f}",
        "--auto-skip-seconds",
        "1",
        "--max-file-lag",
        "0.1",
        "--stop-at-end",
        # High-resolution rendering
        "--viewport",
        f"{width}x{height}",
        "--multi-sampling",
        # Camera / presentation
        "--camera-mode",
        "overview",
        "--disable-auto-rotate",
        # Appearance
        "--background-colour",
        "101820",
        "--bloom-intensity",
        "0.8",
        "--bloom-multiplier",
        "0.8",
        "--elasticity",
        "0.05",
        # Keep the tree readable
        "--file-idle-time",
        "0",
        "--dir-name-depth",
        "2",
        # Metadata
        "--title",
        repo.name,
        "--key",
        "--date-format",
        "%Y-%m-%d",
        "--font-size",
        "28",
        "--file-font-size",
        "16",
        "--dir-font-size",
        "18",
        "--user-font-size",
        "18",
        # Output
        "--output-framerate",
        str(args.fps),
        "--output-ppm-stream",
        "-",
    ]

    # Gource writes PPM frames to stdout.
    #
    # Explicitly specifying rgb24 makes the input format unambiguous for
    # modern FFmpeg builds. The size is also specified explicitly.
    ffmpeg_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-f",
        "image2pipe",
        "-vcodec",
        "ppm",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(args.fps),
        "-i",
        "-",
        # High-quality H.264.
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        str(args.crf),
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "0",
        # Make the MP4 streamable immediately.
        "-movflags",
        "+faststart",
        str(output),
    ]

    print("Rendering...")
    print()

    gource = None
    encoder = None

    try:
        gource = subprocess.Popen(
            gource_cmd,
            stdout=subprocess.PIPE,
            stderr=None,
        )

        encoder = subprocess.Popen(
            ffmpeg_cmd,
            stdin=gource.stdout,
        )

        # FFmpeg now owns the pipe.
        gource.stdout.close()

        ffmpeg_rc = encoder.wait()
        gource_rc = gource.wait()

    except KeyboardInterrupt:
        print("\nInterrupted; stopping processes...")
        for proc in (encoder, gource):
            if proc and proc.poll() is None:
                proc.terminate()
        output.unlink(missing_ok=True)
        sys.exit(130)

    if gource_rc != 0:
        output.unlink(missing_ok=True)
        sys.exit(
            f"Gource failed with exit code {gource_rc}. "
            "Run 'gource --help' to check options supported by your build."
        )

    if ffmpeg_rc != 0:
        output.unlink(missing_ok=True)
        sys.exit(f"FFmpeg failed with exit code {ffmpeg_rc}.")

    if not output.exists() or output.stat().st_size == 0:
        sys.exit("Rendering reported success, but no MP4 was produced.")

    print()
    print("=" * 68)
    print(f"  Finished: {output}")
    print(f"  Size    : {output.stat().st_size / 1024**2:.1f} MiB")
    print("=" * 68)


if __name__ == "__main__":
    main()
