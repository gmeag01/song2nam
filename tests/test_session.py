"""② 세션: 전부 unknown이면 전체 범위, 겹침·충돌 계산, 잘못된 값 거부, 세션 벡터."""

import numpy as np
import pytest

from ampcap import session as S
from ampcap.graybox.amp import INDEX, N_PARAMS


def test_all_unknown_gives_full_range():
    p = S.compute_prior({})
    assert np.allclose(p.lo, 0) and np.allclose(p.hi, 1)
    assert np.allclose(p.center, 0.5)
    assert p.warnings == []


def test_overlap_uses_intersection():
    p = S.compute_prior({"amp": {"family": "british", "gain": "high"}})
    i = INDEX["pre1_gain_db"]
    fam = S.MAPPING["amp.family"]["british"]["pre1_gain_db"]
    gain = S.MAPPING["amp.gain"]["high"]["pre1_gain_db"]
    assert p.lo[i] == pytest.approx(max(fam[0], gain[0]))
    assert p.hi[i] == pytest.approx(min(fam[1], gain[1]))
    assert p.warnings == []


def test_conflict_widens_to_union_with_warning():
    p = S.compute_prior({"amp": {"family": "fender", "gain": "high"}})
    i = INDEX["pre1_gain_db"]
    fam = S.MAPPING["amp.family"]["fender"]["pre1_gain_db"]
    gain = S.MAPPING["amp.gain"]["high"]["pre1_gain_db"]
    assert fam[1] < gain[0]  # 정말 겹치지 않는 경우인지
    assert p.lo[i] == pytest.approx(min(fam[0], gain[0]))
    assert p.hi[i] == pytest.approx(max(fam[1], gain[1]))
    assert any(w.startswith("range_conflict:pre1_gain_db") for w in p.warnings)


def test_low_confidence_widens_toward_full_range():
    hi_conf = S.compute_prior({"amp": {"family": "british"}})
    lo_conf = S.compute_prior({"amp": {"family": "british"}, "confidence": {"amp": 0.5}})
    zero = S.compute_prior({"amp": {"family": "british"}, "confidence": {"amp": 0.0}})
    i = INDEX["pre1_gain_db"]
    assert lo_conf.lo[i] <= hi_conf.lo[i] and lo_conf.hi[i] >= hi_conf.hi[i]
    assert zero.lo[i] == pytest.approx(0) and zero.hi[i] == pytest.approx(1)


def test_boost_none_fixes_drive():
    p = S.compute_prior({"boost": "none"})
    i = INDEX["boost_drive_db"]
    assert p.lo[i] == p.hi[i] == 0.0
    assert p.fixed[i]


@pytest.mark.parametrize("bad", [
    {"amp": {"family": "marshall"}},
    {"guitar": {"pickup": "humbuckers"}},
    {"cab": {"size": "8x10"}},
    {"boost": "fuzz"},
    {"capture_target": "bass"},
    {"confidence": {"amp": 1.5}},
    {"confidence": {"speaker": 1.0}},
    {"same_rig": "yes"},
])
def test_invalid_values_rejected(bad):
    with pytest.raises(S.SessionError):
        S.validate(bad)


def test_model_text_maps_to_family():
    s = S.validate({"amp": {"model": "Marshall JCM800 2203"}})
    assert s["amp"]["family"] == "british"
    s = S.validate({"amp": {"model": "Mesa Dual Rectifier"}})
    assert s["amp"]["family"] == "american_hg"
    s = S.validate({"amp": {"family": "vox", "model": "JCM800"}})  # 명시한 계열 우선
    assert s["amp"]["family"] == "vox"


def test_session_vector_layout():
    assert S.SESSION_DIM == 21
    v = S.session_vector({})
    assert v.shape == (21,)
    # 필드마다 unknown 칸만 1
    offs = np.cumsum([0] + [len(o) + 1 for _, o, _ in S.FIELDS])
    for (a, b) in zip(offs[:-1], offs[1:]):
        assert v[a:b].sum() == 1 and v[b - 1] == 1

    v = S.session_vector({"amp": {"family": "british", "gain": "high"}, "confidence": {"amp": 0.5}})
    fam_off = offs[1]
    assert v[fam_off + S.FAMILIES.index("british")] == pytest.approx(0.5)


def test_design_doc_example_validates():
    s = {
        "artist": "...", "song": "...", "capture_target": "rhythm", "same_rig": True,
        "guitar": {"pickup": "humbucker"},
        "amp": {"family": "british", "model": "JCM800", "gain": "high"},
        "cab": {"size": "4x12"}, "boost": "tube_screamer",
        "confidence": {"amp": 1.0, "boost": 0.5},
    }
    p = S.compute_prior(s)
    assert p.lo.shape == (N_PARAMS,) and np.all(p.lo <= p.hi)


def test_old_ts_boost_alias():
    # 예전 session.json의 "ts"는 tube_screamer로 읽는다
    assert S.validate({"boost": "ts"})["boost"] == "tube_screamer"
    assert np.array_equal(S.session_vector({"boost": "ts"}), S.session_vector({"boost": "tube_screamer"}))


def test_random_and_dropout_sessions_are_valid():
    rng = np.random.default_rng(0)
    for _ in range(50):
        s = S.random_session(rng, p_unknown=0.2)
        S.compute_prior(s)
        d = S.dropout_session(s, rng)
        assert S.session_vector(d).shape == (21,)
