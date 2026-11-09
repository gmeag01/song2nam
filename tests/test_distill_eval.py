"""⑦ 증류·⑨ 평가·④ 합성 데이터.

⑦ ESR 테스트(미사용 DI로 G와 .nam 비교, 목표 0.01 미만)는 NAM 학습이 필요해서
AMPCAP_NAM_INPUT(표준 입력 wav 경로)이 있을 때만 돈다(Colab 노트북 00).
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from ampcap.evaluate import esr, fad_mel, mel_l1, mushra_summary, theta_error_units
from ampcap.graybox.amp import N_PARAMS, GrayboxAmp, theta_from_units
from ampcap.separate import save_wav

SR = 48000


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
    assert b["session"].shape == (3, 21)
    assert torch.isfinite(b["y"]).all()


@pytest.mark.skipif(not os.environ.get("AMPCAP_NAM_INPUT"), reason="NAM 표준 입력 파일 필요(Colab)")
def test_distill_esr_on_unused_di(tmp_path):
    from ampcap.distill import distill

    rng = np.random.default_rng(1)
    th = theta_from_units({"pre1_gain_db": 24.0, "ts_mid_db": 4.0})
    di = [(rng.standard_normal(SR * 3) * 0.05).astype(np.float32) for _ in range(2)]
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
