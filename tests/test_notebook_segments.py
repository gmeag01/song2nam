"""notebooks/02 세션 셀의 구간 입력(분:초-분:초)이 pipeline --segments 형식(초)으로 바뀌는지."""

import json
import re
from pathlib import Path

import pytest

from ampcap.pipeline import parse_segments

NB = Path(__file__).resolve().parents[1] / "notebooks" / "02_song_to_nam.ipynb"


def _run_cell(tmp_path, segments: str) -> dict:
    cells = json.loads(NB.read_text(encoding="utf-8"))["cells"]
    src = next("".join(c["source"]) for c in cells if "#@title 3" in "".join(c["source"]))
    src = re.sub(r'^segments = .*$', f"segments = {segments!r}", src, flags=re.M)
    g = {"SONG": str(tmp_path / "song.mp3")}
    exec(src, g)
    return g


@pytest.mark.parametrize("text, want", [
    ("", None),
    ("0:20-0:45, 2:10-2:22", [[20.0, 45.0], [130.0, 142.0]]),
    ("1:05.5-1:10", [[65.5, 70.0]]),
    ("15-30", [[15.0, 30.0]]),
])
def test_segments_to_seconds(tmp_path, text, want):
    seg = _run_cell(tmp_path, text)["SEGMENTS"]
    assert (parse_segments(seg) if seg else None) == want


@pytest.mark.parametrize("text", ["0:45-0:20", "0:20", "a-b"])
def test_bad_segments_rejected(tmp_path, text):
    with pytest.raises(ValueError):
        _run_cell(tmp_path, text)
