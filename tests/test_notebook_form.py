"""notebooks/02의 세션 폼: 드롭다운 글자 ↔ 세션 값 표가 session.py 스키마와 맞고, 폼으로 만든 세션이 검증을 통과하는지."""

import ast
import json
import re
from pathlib import Path

from ampcap import session as S

NB = Path(__file__).resolve().parents[1] / "notebooks" / "02_song_to_nam.ipynb"

# 폼 변수 → (글자 표 이름, 스키마 선택지)
FORM = {
    "pickup": ("PICKUP", S.PICKUPS), "amp_family": ("FAMILY", S.FAMILIES), "amp_gain": ("GAIN", S.GAINS),
    "cab_size": ("CAB", S.CABS), "boost": ("BOOST", S.BOOSTS),
}


def _session_cell() -> str:
    cells = json.loads(NB.read_text(encoding="utf-8"))["cells"]
    return next("".join(c["source"]) for c in cells
                if c["cell_type"] == "code" and "#@title 3. 세션 정보" in "".join(c["source"]))


def _tables(src: str) -> dict[str, dict]:
    """PICKUP = dict(zip([...], [...])) 꼴의 표를 읽는다."""
    out = {}
    for node in ast.parse(src).body:
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and getattr(node.value.func, "id", "") == "dict"):
            keys, vals = (ast.literal_eval(a) for a in node.value.args[0].args)
            out[node.targets[0].id] = dict(zip(keys, vals))
    return out


def _run(src: str, tmp_path, **form) -> dict:
    for k, v in form.items():
        src, n = re.subn(rf"^{k} = .*$", f"{k} = {v!r}", src, flags=re.M)
        assert n == 1, k
    g = {"SONG": str(tmp_path / "song.mp3")}
    exec(src, g)
    return g


def test_dropdowns_match_schema():
    src = _session_cell()
    opts = {k: ast.literal_eval(v) for k, v in re.findall(r"^(\w+) = .* #@param (\[.*\])", src, re.M)}
    tables = _tables(src)
    for var, (table, schema) in FORM.items():
        assert opts[var] == list(tables[table]), var  # 드롭다운 = 표의 글자 (순서까지)
        assert sorted(tables[table].values()) == sorted(schema + ["unknown"]), var
    for var in ("pickup_source", "amp_source", "cab_source", "boost_source"):
        assert opts[var] == list(tables["SOURCE"]), var
    assert all(0.0 <= c <= 1.0 for c in tables["SOURCE"].values())


def test_defaults_validate(tmp_path):
    g = _run(_session_cell(), tmp_path)
    s = S.load_session(g["SESSION"])
    assert s["amp"]["family"] == "unknown" and g["SEGMENTS"] is None


def test_even_flow_choices(tmp_path):
    src = _session_cell()
    t = _tables(src)
    pick = lambda table, value: next(k for k, v in t[table].items() if v == value)  # noqa: E731
    g = _run(src, tmp_path, pickup=pick("PICKUP", "humbucker"), amp_family=pick("FAMILY", "unknown"),
             amp_name="JCM800", amp_gain=pick("GAIN", "mid"), cab_size=pick("CAB", "4x12"),
             boost=pick("BOOST", "tube_screamer"), cab_source=pick("SOURCE", 0.5), segments=" 20:45 ")
    s = S.load_session(g["SESSION"])
    assert s["amp"]["family"] == "british"  # 앰프 이름에서 계열을 찾음
    assert s["amp"]["gain"] == "mid" and s["confidence"]["cab"] == 0.5
    assert g["SEGMENTS"] == "20:45"
