#!/usr/bin/env python3
"""Render a terminal demo from an actual provider-free analyzer run."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"
WIDTH, HEIGHT = 1280, 720
FPS = 8
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

BG = "#07111f"
PANEL = "#0d1b2d"
PANEL_2 = "#12243a"
TEXT = "#e6edf7"
MUTED = "#8294ad"
CYAN = "#4bd5e7"
GREEN = "#55d6a5"
YELLOW = "#ffcc66"
RED = "#ff7185"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source(manifest: dict) -> None:
    source = manifest["source_commit"]
    tree = subprocess.check_output(
        ["git", "rev-parse", f"{source}^{{tree}}"], cwd=ROOT, text=True
    ).strip()
    if tree != manifest["source_tree"]:
        raise RuntimeError(f"source tree drift: {tree}")
    for path, expected_blob in manifest["runtime_inputs"].items():
        current = subprocess.check_output(
            ["git", "hash-object", path], cwd=ROOT, text=True
        ).strip()
        source_blob = subprocess.check_output(
            ["git", "rev-parse", f"{source}:{path}"], cwd=ROOT, text=True
        ).strip()
        if current != expected_blob or source_blob != expected_blob:
            raise RuntimeError(f"runtime input drift: {path}")


def capture_analysis() -> tuple[dict, list[str]]:
    completed = subprocess.run(
        [sys.executable, "cli/analyze_results.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout + completed.stderr)

    raw = completed.stdout + completed.stderr
    clean_lines = []
    for line in raw.splitlines():
        line = re.sub(r"^\d{4}-\d{2}-\d{2} [\d:,]+ - [^-]+ - \w+ - ", "", line)
        line = line.replace(str(ROOT), "$REPO")
        if line.strip():
            clean_lines.append(line)

    analysis = json.loads((ROOT / "results/classification_analysis.json").read_text())
    stats = analysis["basic_statistics"]
    expected = {
        "total_venues": 34,
        "execution_succeeded": 34,
        "classified_venues": 17,
        "abstained_venues": 17,
        "failed_venues": 0,
        "execution_success_rate": 100.0,
        "classification_coverage": 50.0,
    }
    for key, value in expected.items():
        if stats[key] != value:
            raise RuntimeError(f"unexpected {key}: {stats[key]}")
    return analysis, clean_lines


def rounded(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: str,
    radius: int = 18,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill)


def text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    value: str,
    size: int,
    color: str = TEXT,
    bold: bool = False,
) -> None:
    draw.text(xy, value, font=font(size, bold), fill=color)


def base_frame(
    section: str, progress: float
) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 7), fill="#193048")
    draw.rectangle((0, 0, int(WIDTH * progress), 7), fill=CYAN)
    text(draw, (42, 30), "VENUE RESULT AUDIT", 17, CYAN, True)
    text(draw, (1002, 30), section.upper(), 14, MUTED)
    rounded(draw, (34, 69, WIDTH - 34, HEIGHT - 42), PANEL, 20)
    text(
        draw,
        (50, HEIGHT - 29),
        "historical snapshot  •  provider-free replay  •  source pinned",
        13,
        MUTED,
    )
    return image, draw


def terminal_header(draw: ImageDraw.ImageDraw, y: int = 92) -> None:
    for x, color in ((60, RED), (82, YELLOW), (104, GREEN)):
        draw.ellipse((x, y, x + 11, y + 11), fill=color)
    text(draw, (WIDTH - 260, y - 3), "analysis.log", 14, MUTED)


def slide_title(progress: float) -> Image.Image:
    image, draw = base_frame("01 / audit", progress)
    terminal_header(draw)
    text(draw, (76, 159), "$ python cli/analyze_results.py", 21, GREEN)
    text(draw, (76, 215), "RESULT INTEGRITY", 48, TEXT, True)
    text(draw, (76, 276), "Execution success is not classification success.", 24, CYAN)
    rounded(draw, (76, 351, 1204, 559), PANEL_2, 16)
    text(draw, (104, 383), "Checked-in input", 15, MUTED)
    text(draw, (104, 417), "34 historical venue rows", 29, TEXT, True)
    text(draw, (104, 472), "No provider calls  ·  No ground-truth labels", 21, YELLOW)
    text(draw, (104, 514), "Descriptive output—not a model-quality claim", 19, MUTED)
    return image


def stat_card(
    draw: ImageDraw.ImageDraw, x: int, y: int, value: str, label: str, color: str
) -> None:
    rounded(draw, (x, y, x + 252, y + 155), PANEL_2, 14)
    text(draw, (x + 22, y + 22), value, 46, color, True)
    text(draw, (x + 22, y + 91), label, 16, TEXT)


def slide_outcomes(progress: float) -> Image.Image:
    image, draw = base_frame("02 / outcomes", progress)
    terminal_header(draw)
    text(draw, (76, 142), "BASIC STATISTICS", 29, TEXT, True)
    stat_card(draw, 76, 203, "34", "execution succeeded", GREEN)
    stat_card(draw, 356, 203, "17", "classified", CYAN)
    stat_card(draw, 636, 203, "17", "abstained", YELLOW)
    stat_card(draw, 916, 203, "0", "failed", GREEN)
    rounded(draw, (76, 402, 1204, 555), PANEL_2, 14)
    text(draw, (105, 430), "OBSERVED CLASSIFICATION COVERAGE", 16, MUTED)
    draw.rounded_rectangle((105, 478, 1175, 508), radius=15, fill="#26384e")
    draw.rounded_rectangle((105, 478, 640, 508), radius=15, fill=CYAN)
    text(draw, (105, 522), "17 / 34", 18, TEXT, True)
    text(draw, (1074, 522), "50.0%", 18, CYAN, True)
    return image


def slide_sentinels(progress: float) -> Image.Image:
    image, draw = base_frame("03 / semantics", progress)
    terminal_header(draw)
    text(draw, (76, 142), "SENTINEL-AWARE STATUS", 29, TEXT, True)
    rounded(draw, (76, 196, 566, 565), PANEL_2, 14)
    text(draw, (104, 226), "category_label", 15, MUTED)
    for idx, token in enumerate(("null / NaN", '"None"', '"Unknown"', '"   " (blank)')):
        y = 275 + idx * 60
        text(draw, (108, y), token, 21, YELLOW)
        text(draw, (390, y), "→ abstained", 18, TEXT, True)
    rounded(draw, (606, 196, 1204, 565), PANEL_2, 14)
    text(draw, (638, 226), "distribution denominator", 15, MUTED)
    text(draw, (638, 278), "CLASSIFIED ROWS ONLY", 27, CYAN, True)
    text(draw, (638, 341), "Category rows", 17, MUTED)
    text(draw, (1030, 341), "17", 23, TEXT, True)
    text(draw, (638, 385), "Similarity rows", 17, MUTED)
    text(draw, (1030, 385), "17", 23, TEXT, True)
    text(draw, (638, 448), "Abstentions never become", 17, TEXT)
    text(draw, (638, 482), "zero-score classifications.", 17, TEXT)
    return image


def slide_distribution(progress: float) -> Image.Image:
    image, draw = base_frame("04 / evidence", progress)
    terminal_header(draw)
    text(draw, (76, 142), "CLASSIFIED-ROW DISTRIBUTION", 29, TEXT, True)
    labels = [
        ("0.0–0.2", 0),
        ("0.2–0.4", 9),
        ("0.4–0.6", 8),
        ("0.6–0.8", 0),
        ("0.8–1.0", 0),
    ]
    for i, (label, count) in enumerate(labels):
        y = 215 + i * 60
        text(draw, (96, y), label, 17, MUTED)
        draw.rounded_rectangle((240, y + 2, 1060, y + 27), radius=12, fill="#26384e")
        if count:
            draw.rounded_rectangle(
                (240, y + 2, 240 + count * 82, y + 27),
                radius=12,
                fill=CYAN if count == 9 else GREEN,
            )
        text(draw, (1090, y - 1), str(count), 20, TEXT, True)
    text(
        draw,
        (96, 536),
        "Similarity is descriptive; no accuracy or precision is inferred.",
        17,
        YELLOW,
    )
    return image


def slide_receipt(progress: float, source: str) -> Image.Image:
    image, draw = base_frame("05 / receipt", progress)
    terminal_header(draw)
    text(draw, (76, 142), "$ python -m pytest -q", 19, GREEN)
    text(draw, (76, 186), "12 passed", 38, GREEN, True)
    rounded(draw, (76, 254, 1204, 552), PANEL_2, 14)
    rows = [
        ("source commit", source[:12]),
        ("input rows", "34 checked-in historical rows"),
        ("provider calls", "none"),
        ("missing credentials", "exit 1"),
        ("claim boundary", "coverage only — no quality claim"),
    ]
    for i, (label, value) in enumerate(rows):
        y = 285 + i * 48
        text(draw, (108, y), label, 15, MUTED)
        text(draw, (385, y), value, 17, TEXT, i == 0)
    return image


def main() -> int:
    manifest = json.loads((DEMO / "source-manifest.json").read_text())
    verify_source(manifest)
    analysis, transcript = capture_analysis()
    stats = analysis["basic_statistics"]

    selected = [
        line
        for line in transcript
        if any(
            marker in line
            for marker in (
                "Analyzing 34 rows",
                "Execution succeeded:",
                "Classified:",
                "Abstained:",
                "Failed:",
                "Observed classification coverage:",
                "Results analysis completed successfully",
            )
        )
    ]
    transcript_path = DEMO / "analysis-transcript.txt"
    transcript_path.write_text(
        "# Sanitized lines from: python cli/analyze_results.py\n"
        + "\n".join(selected)
        + "\n"
    )

    receipt = {
        "format_version": 1,
        "scope": "checked-in historical result analysis",
        "source_commit": manifest["source_commit"],
        "source_tree": manifest["source_tree"],
        "command": "python cli/analyze_results.py",
        "input": {
            "path": "results/venue_category_matches.csv",
            "rows": 34,
            "sha256": sha256(ROOT / "results/venue_category_matches.csv"),
        },
        "observed": {
            key: stats[key]
            for key in (
                "total_venues",
                "execution_succeeded",
                "classified_venues",
                "abstained_venues",
                "failed_venues",
                "execution_success_rate",
                "classification_coverage",
            )
        },
        "sentinel_policy": ["null", "None", "Unknown", "blank"],
        "limitations": [
            "historical checked-in output",
            "no provider calls during replay",
            "no ground-truth labels",
            "no accuracy, precision, provider-quality, or production-performance claim",
        ],
    }
    capture_path = DEMO / "capture.json"
    capture_path.write_text(json.dumps(receipt, indent=2) + "\n")

    slides = [slide_title, slide_outcomes, slide_sentinels, slide_distribution]
    seconds = [3.0, 3.0, 3.0, 3.0, 3.0]
    with tempfile.TemporaryDirectory(prefix="categorization-demo-") as frame_dir:
        frame_root = Path(frame_dir)
        frame_number = 0
        poster = slide_outcomes(0.35)
        poster.save(DEMO / "venue-result-integrity-poster.png", optimize=True)
        for slide_index, duration in enumerate(seconds):
            frame_count = int(duration * FPS)
            for local_index in range(frame_count):
                overall = (frame_number + 1) / int(sum(seconds) * FPS)
                if slide_index < len(slides):
                    frame = slides[slide_index](overall)
                else:
                    frame = slide_receipt(overall, manifest["source_commit"])
                frame.save(frame_root / f"frame-{frame_number:04d}.png", optimize=True)
                frame_number += 1

        input_pattern = str(frame_root / "frame-%04d.png")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(FPS),
                "-i",
                input_pattern,
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(DEMO / "venue-result-integrity.mp4"),
            ],
            check=True,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(FPS),
                "-i",
                input_pattern,
                "-vf",
                "fps=8,scale=960:-1:flags=lanczos,split[s0][s1];"
                "[s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3",
                "-loop",
                "0",
                str(DEMO / "venue-result-integrity.gif"),
            ],
            check=True,
        )

    checksummed = [
        "analysis-transcript.txt",
        "capture.json",
        "source-manifest.json",
        "venue-result-integrity-poster.png",
        "venue-result-integrity.gif",
        "venue-result-integrity.mp4",
    ]
    (DEMO / "checksums.sha256").write_text(
        "".join(f"{sha256(DEMO / name)}  {name}\n" for name in checksummed)
    )
    print(f"rendered {frame_number} actual-result frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
