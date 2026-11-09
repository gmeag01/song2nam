"""① 구간 점검: 무음 섞인 사인파로 길이·무음 점검, 이음새 클릭 없음. (Demucs는 쓰지 않는다)"""

import json

import numpy as np

from ampcap import separate as S

SR = S.SR


def _sine(sec, f=220.0, amp=0.5):
    t = np.arange(int(sec * SR)) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def _sig_with_silence():
    # 5초 소리, 3초 무음, 5초 소리, 2초 무음, 5초 소리 → 연주 15초
    return np.concatenate([_sine(5), np.zeros(3 * SR, np.float32), _sine(5, 330),
                           np.zeros(2 * SR, np.float32), _sine(5, 440)])


def test_active_length_excludes_silence():
    x = _sig_with_silence()
    out, rep = S.check_and_build(x, [[0.0, len(x) / SR]], min_active_sec=10)
    assert abs(rep.active_sec - 15.0) < 0.3
    assert rep.ok


def test_too_short_blocks_next_step():
    x = _sig_with_silence()
    _, rep = S.check_and_build(x, [[0.0, 6.0]], min_active_sec=15)
    assert not rep.ok and "짧" in rep.message


def test_no_clicks_at_seams():
    """이어 붙인 곳의 샘플 간 차이가 원 신호의 최대 기울기를 크게 넘지 않는다."""
    x = _sig_with_silence()
    out, _ = S.check_and_build(x, [[0.0, 7.0], [8.0, 20.0]], min_active_sec=1)
    max_slope = 2 * np.pi * 440 / SR * 0.5  # 가장 높은 사인파의 최대 기울기
    assert np.max(np.abs(np.diff(out))) < 2.5 * max_slope


def test_crossfade_concat_is_smooth():
    a, b = _sine(1, 200), -_sine(1, 300)
    out = S.crossfade_concat([a, b])
    assert len(out) == len(a) + len(b) - int(S.CROSSFADE_SEC * SR)
    assert np.max(np.abs(np.diff(out))) < 0.1


def test_empty_segments_use_full_stem_with_warning():
    x = _sine(16)
    _, rep = S.check_and_build(x, [])
    assert any(w.startswith("full_stem_used") for w in rep.warnings)


def test_clipping_warning():
    x = np.clip(_sine(16, amp=2.0), -1, 1)
    _, rep = S.check_and_build(x, [[0, 16]])
    assert "clipping" in rep.warnings


def test_multi_guitar_warning_and_same_rig():
    rng = np.random.default_rng(0)
    # 좌우가 서로 다른 독립 신호(더블 트래킹 흉내)
    st = np.stack([_sine(16, 220) + 0.1 * rng.standard_normal(16 * SR).astype(np.float32),
                   _sine(16, 233) + 0.1 * rng.standard_normal(16 * SR).astype(np.float32)])
    _, rep = S.check_and_build(st, [[0, 16]], same_rig=False)
    assert any(w == "multi_guitar_suspected" for w in rep.warnings)
    _, rep2 = S.check_and_build(st, [[0, 16]], same_rig=True)
    assert any(w == "multi_guitar_suspected:same_rig" for w in rep2.warnings)
    assert "참고" in S.multi_guitar_message(rep2.warnings)


def test_recommend_segments_finds_mono_part():
    rng = np.random.default_rng(1)
    mono = _sine(8)
    a = np.stack([mono, mono])  # 가운데 한 대
    b = np.stack([_sine(8, 220), _sine(8, 233) + 0.2 * rng.standard_normal(8 * SR).astype(np.float32)])
    st = np.concatenate([b, a, b], axis=1)
    segs = S.recommend_segments(st)
    assert segs, "추천 구간이 없음"
    a0, a1 = segs[0]
    assert 7.0 <= a0 <= 9.0 and 15.0 <= a1 <= 17.5


def test_save_outputs(tmp_path):
    x = _sine(16)
    out, rep = S.check_and_build(x, [[0, 16]], song="t")
    S.save_outputs(tmp_path, out, rep)
    d = json.loads((tmp_path / "segments.json").read_text(encoding="utf-8"))
    assert d["song"] == "t" and d["segments"] == [[0.0, 16.0]] and "active_sec" in d
    assert (tmp_path / "stem.wav").exists()


def test_rerun_reuses_demucs_but_applies_new_segments(tmp_path, monkeypatch):
    """같은 실행 폴더로 다시 돌리면 분리는 건너뛰고, 바뀐 구간은 반영한다."""
    stem = np.stack([_sine(30), _sine(30)])
    S.save_wav(tmp_path / "stem_full.wav", stem)

    def no_demucs(*a, **k):
        raise AssertionError("stem_full.wav가 있는데 다시 분리함")

    monkeypatch.setattr(S, "separate_guitar", no_demucs)
    _, r1 = S.run("unused.wav", tmp_path, [[0.0, 16.0]], song="t")
    _, r2 = S.run("unused.wav", tmp_path, [[10.0, 30.0]], song="t")
    d = json.loads((tmp_path / "segments.json").read_text(encoding="utf-8"))
    assert d["segments"] == [[10.0, 30.0]]
    assert r2.active_sec > r1.active_sec
