"""⑦ 증류·⑨ 평가·④ 합성 데이터.

⑦ ESR 테스트(미사용 DI로 G와 .nam 비교, 목표 0.01 미만)는 NAM 학습이 필요해서
AMPCAP_NAM_INPUT(NAM 학습 입력 신호 wav 경로)이 있을 때만 돈다(Colab 노트북 00).
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from ampcap.evaluate import esr, fad_mel, mel_l1, mushra_summary, theta_error_units
from ampcap.graybox.amp import N_PARAMS, GrayboxAmp, theta_from_units
from ampcap.separate import save_wav

SR = 48000


def test_gear_type_follows_boost():
    pytest.importorskip("nam.models.metadata")
    from nam.models.metadata import GearType

    from ampcap.distill import gear_type_for

    assert gear_type_for({"boost": "tube_screamer"}) == GearType.AMP_PEDAL_CAB
    assert gear_type_for({"boost": "none"}) == GearType.AMP_CAB
    assert gear_type_for(None) == GearType.AMP_CAB


def test_metrics_basic():
    from scipy.signal import butter, sosfilt

    x = np.random.default_rng(0).standard_normal(SR * 4).astype(np.float32) * 0.1
    dark = sosfilt(butter(4, 2000, fs=SR, output="sos"), x).astype(np.float32)
    assert esr(x, x) == 0
    assert abs(esr(np.zeros_like(x), x) - 1) < 1e-9
    assert mel_l1(x, x) < 1e-6
    assert fad_mel(x, x) < 1e-2 * fad_mel(x, dark)
    e = theta_error_units(theta_from_units({"pre1_gain_db": 10.0}), theta_from_units({"pre1_gain_db": 20.0}))
    assert abs(e["pre1_gain_db"] - 10.0) < 1e-6 and e["in_hp_hz(oct)"] < 1e-6
    m = mushra_summary({"nam": [70, 80, 90]})
    assert m["nam"]["mean"] == 80


def test_render_nam_output_levels(tmp_path):
    from ampcap.distill import render_nam_output

    rng = np.random.default_rng(0)
    x = (rng.standard_normal(SR * 2) * 0.2).astype(np.float32)
    save_wav(tmp_path / "in.wav", x, subtype="PCM_24")
    th = theta_from_units({"pre1_gain_db": 30.0})
    lvl = render_nam_output(th, tmp_path / "in.wav", tmp_path / "out", di_ref_rms=0.05, device="cpu")
    import soundfile as sf

    y, sr = sf.read(str(tmp_path / "out" / "output.wav"))
    info = sf.info(str(tmp_path / "out" / "output.wav"))
    assert sr == SR and len(y) == len(x) and info.subtype == "PCM_24"
    assert np.abs(y).max() <= 0.9
    assert lvl["input_gain_db"] == pytest.approx(20 * np.log10(0.05 / 0.2), abs=0.5)


def test_synth_generator_batch(tmp_path):
    from ampcap.synth_data import DIReader, SynthConfig, SynthGenerator, make_split

    rng = np.random.default_rng(0)
    d = tmp_path / "di"
    for i in range(10):
        save_wav(d / f"{i + 1}.wav", (rng.standard_normal(SR * 4) * 0.1).astype(np.float32))
    split = make_split(d, save_to=tmp_path / "split.json")
    allf = sum(split.values(), [])
    assert len(allf) == 10 and len(set(allf)) == 10  # 겹치지 않음
    reader = DIReader([d / f for f in split["train"]])
    gen = SynthGenerator(reader, cfg=SynthConfig(batch_size=3, augment=False))
    b = gen.next_batch()
    assert b["x"].shape == (3, 3 * SR) and b["theta"].shape == (3, N_PARAMS)
    assert b["session_vec"].shape == (3, 21)
    assert torch.isfinite(b["y"]).all()


def test_load_nam_unwraps_slimmable_container(tmp_path):
    pytest.importorskip("nam")
    from ampcap.distill import load_nam

    def linear(rf):
        return {"architecture": "Linear", "config": {"receptive_field": rf, "bias": False},
                "weights": [0.0] * rf, "sample_rate": 48000}

    cfg = {"architecture": "SlimmableContainer",
           "config": {"submodels": [{"max_value": 1.0, "model": linear(8)}, {"max_value": 0.5, "model": linear(4)}]}}
    p = tmp_path / "model.nam"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    assert load_nam(p).receptive_field == 8


EGDB_DI = os.environ.get("AMPCAP_EGDB", "data/EGDB/audio_DI")


@pytest.mark.skipif(not os.environ.get("AMPCAP_NAM_INPUT"), reason="NAM 학습 입력 신호 파일 필요(Colab)")
def test_distill_esr_on_unused_di(tmp_path):
    """설계 4.5: ⑦ 확인용 DI는 EGDB test 분할. 백색 잡음은 기타 DI와 분포가 달라 ESR이 3배쯤 나쁘게 나온다."""
    from ampcap.distill import distill
    from ampcap.synth_data import di_reader_for_split, normalize_di

    if not list(Path(EGDB_DI).glob("*.wav")):
        pytest.skip("EGDB DI 필요 (python -m ampcap.get_data)")
    th = theta_from_units({"pre1_gain_db": 24.0, "ts_mid_db": 4.0})
    reader = di_reader_for_split(EGDB_DI, "test", tmp_path / "split.json")
    di = list(normalize_di(reader.fixed_clips(4, seed=1)))
    out = distill(th, tmp_path, Path(os.environ["AMPCAP_NAM_INPUT"]), 0.05, name="test",
                  epochs=int(os.environ.get("AMPCAP_NAM_EPOCHS", "30")), architecture="lite", verify_di=di)
    print(out["verify"])
    assert Path(out["nam_path"]).exists()
    assert out["verify"]["esr_mean"] < 0.01


def test_full_amp_render_deterministic():
    G = GrayboxAmp()
    x = torch.randn(1, 4800) * 0.05
    th = torch.rand(N_PARAMS)
    assert torch.allclose(G.render(x, th), G.render(x, th))


def test_exp1_resume_after_truncated_line(tmp_path, monkeypatch):
    """Colab 런타임이 쓰는 도중 끊겨 마지막 줄이 반쯤 남아도 이어서 돌릴 수 있어야 한다."""
    from ampcap import evaluate as E

    res = tmp_path / "exp1_results.jsonl"
    ok = {"song": 0, "start": "center", "stats": "all4", "session": "none", "esr": 0.1,
          "mapping_version": 2}
    res.write_text(json.dumps(ok) + "\n" + '{"song": 1, "start": "mul', encoding="utf-8")
    assert E.read_results(res) == [ok]

    class Pool:
        def fixed_clips(self, n, seed):
            return np.zeros((n, 4800), np.float32)

    monkeypatch.setattr(E, "make_fake_songs", lambda *a, **k: [])
    E.run_experiment1(None, Pool(), tmp_path, device="cpu")
    with res.open("a", encoding="utf-8") as f:  # 재개 뒤 새 결과
        f.write(json.dumps({**ok, "song": 2}) + "\n")
    assert [r["song"] for r in E.read_results(res)] == [0, 2]


def test_exp1_refuses_resume_across_mapping_versions(tmp_path, monkeypatch):
    """대응표 버전이 다른 행(필드 없음 = v1)이 있으면 같은 파일에 이어 쓰지 않는다."""
    import pytest

    from ampcap import evaluate as E

    res = tmp_path / "exp1_results.jsonl"
    res.write_text(json.dumps({"song": 0, "start": "prior_center", "stats": "all4", "session": "none"}) + "\n",
                   encoding="utf-8")
    monkeypatch.setattr(E, "make_fake_songs", lambda *a, **k: [])
    with pytest.raises(ValueError, match="대응표 버전"):
        E.run_experiment1(None, None, tmp_path, device="cpu")


def test_bias_canonicalized_in_theta_error():
    from ampcap.evaluate import canonical_bias
    from ampcap.graybox.amp import INDEX

    th = np.full(N_PARAMS, 0.5)
    th[INDEX["pre1_bias"]], th[INDEX["pre2_bias"]] = 0.3, 0.8   # b1 = −0.2, b2 = +0.3
    flipped = th.copy()
    flipped[INDEX["pre1_bias"]], flipped[INDEX["pre2_bias"]] = 0.7, 0.2   # 두 바이어스를 함께 뒤집은 θ
    np.testing.assert_allclose(canonical_bias(th), canonical_bias(flipped))
    np.testing.assert_allclose(canonical_bias(flipped), flipped)   # pre1_bias ≥ 0이면 그대로


def test_separation_aug_keeps_shape_and_level():
    from ampcap.synth_data import augment_separation

    y = (np.random.default_rng(0).standard_normal(48000) * 0.1).astype(np.float32)
    out = augment_separation(y, np.random.default_rng(1))
    assert out.shape == y.shape and out.dtype == np.float32 and np.all(np.isfinite(out))
    assert 0.5 < np.sqrt(np.mean(out ** 2)) / 0.1 < 1.5


def test_exp1_prior_multi_starts_reproducible_on_resume(tmp_path, monkeypatch):
    """(곡, 세션)마다 시드를 두므로 앞 곡을 건너뛰고 이어 실행해도 prior_multi 시작점이 같다."""
    import ampcap.optimize as O
    from ampcap import evaluate as E
    from ampcap.session import MAPPING_VERSION, random_session

    rng = np.random.default_rng(0)
    songs = [{"id": f"s{i}", "session": random_session(rng), "theta": np.full(N_PARAMS, 0.5),
              "stem": np.zeros(4800, np.float32), "eval_di": None} for i in range(2)]

    class Pool:
        def fixed_clips(self, n, seed):
            return np.zeros((n, 4800), np.float32)

    seen = {}

    class Res:
        theta_star = torch.full((N_PARAMS,), 0.5)
        losses = torch.zeros(1)
        diagnostics = {"steps": 0, "terms": {}, "move": 0.0}

    def fake_fit(target, pool, prior, starts=None, **kw):
        seen.setdefault("calls", []).append(np.array(starts))
        return Res()

    monkeypatch.setattr(E, "make_fake_songs", lambda *a, **k: songs)
    monkeypatch.setattr(E, "_evaluate_theta", lambda *a, **k: {"esr": 0.0})
    monkeypatch.setattr(O, "fit", fake_fit)
    cfg = E.Exp1Config(starts=["prior_multi"], stat_sets={"all4": E.STAT_KEYS}, session_only_baseline=False)

    E.run_experiment1(None, Pool(), tmp_path / "a", cfg, device="cpu")
    full = seen.pop("calls")
    (tmp_path / "b").mkdir()
    done = [{"song": "s0", "start": "prior_multi", "stats": "all4", "session": s,
             "mapping_version": MAPPING_VERSION} for s in cfg.sessions]
    (tmp_path / "b" / "exp1_results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in done),
                                                       encoding="utf-8")
    E.run_experiment1(None, Pool(), tmp_path / "b", cfg, device="cpu")
    resumed = seen.pop("calls")
    assert len(resumed) == len(cfg.sessions)
    for a, b in zip(full[len(cfg.sessions):], resumed):
        np.testing.assert_array_equal(a, b)
    row = E.read_results(tmp_path / "a" / "exp1_results.jsonl")[0]
    assert {"theta_star", "steps", "terms"} <= set(row) and len(row["theta_star"]) == N_PARAMS


def test_gain_sum_error_ignores_exchange():
    from ampcap.evaluate import gain_sum_error

    a = theta_from_units({"in_gain_db": 3.0, "boost_drive_db": 10.0, "boost_level_db": 0.0, "pre1_gain_db": 20.0})
    b = theta_from_units({"in_gain_db": 0.0, "boost_drive_db": 13.0, "boost_level_db": 4.0, "pre1_gain_db": 16.0})
    e = gain_sum_error(a, b)
    assert e["in+boost_drive"] < 1e-6 and e["boost_level+pre1"] < 1e-6 and e["total"] < 1e-6


def test_stat_sensitivity_sees_gain_exchange_as_flat():
    """in_gain +1 dB / boost_drive −1 dB 방향은 사이에 선형 블록만 있어 통계가 변하지 않아야 한다."""
    from ampcap.evaluate import stat_sensitivity
    from ampcap.graybox.amp import INDEX, PARAMS
    from ampcap.stats import StatConfig, Stats

    rng = np.random.default_rng(0)
    di = torch.tensor(rng.standard_normal((1, 4800)) * 0.05, dtype=torch.float32)
    th = theta_from_units({"boost_drive_db": 15.0, "pre1_gain_db": 20.0})
    out = stat_sensitivity(th, di, stats=Stats(StatConfig(n_mels=16)))
    J = torch.as_tensor(out["jacobian"], dtype=torch.float64)
    span = {n: hi - lo for n, _, lo, hi in PARAMS}
    v = torch.zeros(N_PARAMS, dtype=torch.float64)
    v[INDEX["in_gain_db"]], v[INDEX["boost_drive_db"]] = 1 / span["in_gain_db"], -1 / span["boost_drive_db"]
    e = torch.zeros(N_PARAMS, dtype=torch.float64)
    e[INDEX["in_gain_db"]] = 1 / span["in_gain_db"]
    assert (J @ v).norm() < 0.05 * (J @ e).norm()
    assert out["eigvals"].min() > -1e-6 and len(out["weak"]) == 3
