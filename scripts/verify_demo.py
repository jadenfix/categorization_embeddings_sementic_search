#!/usr/bin/env python3
"""Verify source provenance, historical observations, and media integrity."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    manifest = json.loads((DEMO / "source-manifest.json").read_text())
    assert (
        subprocess.check_output(
            ["git", "rev-parse", f"{manifest['source_commit']}^{{tree}}"],
            cwd=ROOT,
            text=True,
        ).strip()
        == manifest["source_tree"]
    )
    for path, blob in manifest["runtime_inputs"].items():
        assert (
            subprocess.check_output(
                ["git", "hash-object", path], cwd=ROOT, text=True
            ).strip()
            == blob
        )
        assert (
            subprocess.check_output(
                ["git", "rev-parse", f"{manifest['source_commit']}:{path}"],
                cwd=ROOT,
                text=True,
            ).strip()
            == blob
        )

    receipt = json.loads((DEMO / "capture.json").read_text())
    rows = pd.read_csv(ROOT / receipt["input"]["path"])
    succeeded = (
        rows["status"].fillna("").astype(str).str.strip().str.casefold() == "success"
    )
    labels = rows["category_label"].fillna("").astype(str).str.strip().str.casefold()
    classified = succeeded & ~labels.isin({"", "none", "null", "unknown"})
    observed = receipt["observed"]
    assert len(rows) == observed["total_venues"] == 34
    assert int(succeeded.sum()) == observed["execution_succeeded"] == 34
    assert int(classified.sum()) == observed["classified_venues"] == 17
    assert int((succeeded & ~classified).sum()) == observed["abstained_venues"] == 17
    assert int((~succeeded).sum()) == observed["failed_venues"] == 0
    assert receipt["input"]["sha256"] == digest(ROOT / receipt["input"]["path"])

    for line in (DEMO / "checksums.sha256").read_text().splitlines():
        expected, filename = line.split("  ", 1)
        assert digest(DEMO / filename) == expected, filename

    gif = Image.open(DEMO / "venue-result-integrity.gif")
    assert gif.size == (960, 540)
    assert gif.n_frames == 120
    for frame in range(gif.n_frames):
        gif.seek(frame)
        gif.convert("RGB").load()

    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,nb_frames:format=duration",
                "-of",
                "json",
                str(DEMO / "venue-result-integrity.mp4"),
            ],
            text=True,
        )
    )
    stream = probe["streams"][0]
    assert stream["codec_name"] == "h264"
    assert (int(stream["width"]), int(stream["height"])) == (1280, 720)
    assert int(stream["nb_frames"]) == 120
    assert 14.9 <= float(probe["format"]["duration"]) <= 15.1
    print("demo verified: source, 34-row receipt, checksums, GIF and H.264 MP4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
