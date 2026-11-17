"""notebooks/02의 세션 폼 선택지가 session.py 스키마와 같고, 폼 기본값으로 만든 세션이 검증을 통과하는지."""

import ast
import json
import re
from pathlib import Path

from ampcap import session as S

NB = Path(__file__).resolve().parents[1] / "notebooks" / "02_song_to_nam.ipynb"


def _session_cell() -> str:
    cells = json.loads(NB.read_text(encoding="utf-8"))["cells"]
    return next("".join(c["source"]) for c in cells
                if c["cell_type"] == "code" and "#@title 3. 세션 정보" in "".join(c["source"]))


def test_form_options_match_schema():
    src = _session_cell()
    opts = {k: ast.literal_eval(v) for k, v in re.findall(r"^(\w+) = .* #@param (\[.*\])", src, re.M)}
    want = {"capture_target": S.TARGETS, "pickup": S.PICKUPS, "amp_family": S.FAMILIES,
            "amp_gain": S.GAINS, "cab_size": S.CABS, "boost": S.BOOSTS}
    for k, w in want.items():
        extra = [] if k == "capture_target" else ["unknown"]
        assert sorted(opts[k]) == sorted(w + extra), k


def test_form_defaults_validate(tmp_path):
    g = {"SONG": str(tmp_path / "song.mp3")}
    exec(_session_cell(), g)
    s = S.load_session(g["SESSION"])
    assert set(s["confidence"]) == {c for _, _, c in S.FIELDS}
    assert g["SEGMENTS"] is None
