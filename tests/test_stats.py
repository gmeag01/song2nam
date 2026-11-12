"""⑥ 통계: 게인/트레블 구분(가장 먼저 실행), 전체 음량 불변.

게인/트레블 테스트가 실패하면 통계 설계를 재검토한다(설계 문서 7장).
"""

import numpy as np
import pytest
import torch

from ampcap.graybox.amp import GrayboxAmp, theta_from_units
from ampcap.stats import STAT_KEYS, Stats, stat_losses

SR = 48000


def _plucks(levels_db=(-40, -28, -16, -4), note_sec=0.5, repeats=3, seed=0):
    """DI 기타 비슷한 음(배음 1/k^1.5, 수 kHz까지)을 4단계 세기로 반복. 완만한 감쇠 포함.

    고역에도 실제 성분이 있어야 트레블 변화가 모든 레벨 구간에서 보인다.
    """
    rng = np.random.default_rng(seed)
    n = int(note_sec * SR)
    t = np.arange(n) / SR
    env = np.exp(-t * 2.0)
    out = []
    for _ in range(repeats):
        for db in levels_db:
            f = rng.uniform(110, 220)
            ks = np.arange(1, 41)
            ks = ks[ks * f < 10000]
            tone = sum(k ** -1.5 * np.sin(2 * np.pi * k * f * t + rng.uniform(0, 2 * np.pi)) for k in ks)
            tone = tone / np.abs(tone).max()
            out.append(10 ** (db / 20) * env * tone)
    return torch.tensor(np.concatenate(out), dtype=torch.float32)


# 저게인·저레벨 기준 설정: 파워앰프·부스트도 거의 선형으로 동작하게
BASE = dict(in_gain_db=-12.0, pre1_gain_db=0.0, pre2_gain_db=0.0, pa_drive_db=0.0,
            boost_drive_db=0.0, ts_treble_db=0.0, cab_lp_hz=7000.0, pre1_lp_hz=12000.0, pre2_lp_hz=12000.0)


def _mel_hz(cfg):
    m = np.linspace(2595 * np.log10(1 + cfg.f_min / 700), 2595 * np.log10(1 + cfg.f_max / 700), cfg.n_mels + 2)[1:-1]
    return 700 * (10 ** (m / 2595) - 1)


def _level_brightness(theta, x):
    """레벨 구간별 밝기 = 고역(2.5~7kHz) 평균 − 중저역(200~1000Hz) 평균 (dB).

    기타 성분이 실제로 있는 대역만 본다. 7kHz 위는 캐비닛이 거의 다 깎아서 바닥값에 가깝다.
    """
    G = GrayboxAmp()
    st = Stats()
    with torch.no_grad():
        y = G(x[None], theta[None])[0]
        s = st(y)
    hz = _mel_hz(st.cfg)
    lev = s["level"][0]
    hi = torch.from_numpy((hz >= 2500) & (hz <= 7000))
    lo = torch.from_numpy((hz >= 200) & (hz <= 1000))
    return lev[:, hi].mean(-1) - lev[:, lo].mean(-1)  # (4,)


def test_gain_vs_treble_distinguished():
    """게인↑: 세게 친 구간만 더 밝아짐. 트레블↑: 모든 구간이 비슷하게 밝아짐."""
    x = _plucks()
    base = theta_from_units(BASE)
    b0 = _level_brightness(base, x)

    gain = theta_from_units({**BASE, "pre1_gain_db": 34.0})
    dg = _level_brightness(gain, x) - b0

    treble = theta_from_units({**BASE, "ts_treble_db": 10.0})
    dt = _level_brightness(treble, x) - b0

    print("게인 Δ밝기(조용→셈):", dg.numpy().round(2), " 트레블 Δ밝기:", dt.numpy().round(2))
    # 게인: 가장 센 구간의 증가가 가장 조용한 구간보다 확실히 크다
    assert dg[3] - dg[0] > 3.0
    assert dg[3] > 0
    # 트레블: 모든 구간이 밝아지고, 구간 간 차이는 평균 증가량에 비해 작다
    assert (dt > 0).all()
    assert (dt.max() - dt.min()) < 0.35 * dt.mean()
    # 두 변화의 "구간 의존성" 비율이 확실히 다르다
    assert (dg[3] - dg[0]) / dg.abs().mean().clamp_min(1e-6) > 2 * (dt.max() - dt.min()) / dt.mean()


@pytest.mark.parametrize("scale", [0.1, 3.0])
def test_invariant_to_overall_volume(scale):
    torch.manual_seed(0)
    G = GrayboxAmp()
    x = _plucks()
    th = theta_from_units({"pre1_gain_db": 20.0})
    y = G(x[None], th[None])[0]
    st = Stats()
    a = st(y)
    b = st(y * scale)
    for k in STAT_KEYS:
        assert torch.allclose(a[k], b[k], atol=1e-3), k


def test_shapes_and_grouping():
    st = Stats()
    x = torch.randn(3, 4, 24000) * 0.1  # 그룹 3 × 클립 4
    s = st(x)
    assert s["ltas"].shape == (3, 128)
    assert s["level"].shape == (3, 4, 128)
    assert s["flat"].shape == (3, 4)
    assert s["dyn"].shape == (3, 3)
    t = st(torch.randn(48000) * 0.1)
    ls = stat_losses(s, t)
    assert all(v.shape == (3,) for v in ls.values())


def test_stats_are_differentiable():
    st = Stats()
    G = GrayboxAmp()
    x = _plucks(repeats=1)[None]
    th = torch.full((1, 24), 0.5, requires_grad=True)
    s = st(G(x, th))
    t = st(x)
    loss = sum(v.sum() for v in stat_losses(s, t).values())
    loss.backward()
    assert torch.isfinite(th.grad).all() and th.grad.abs().sum() > 0


def test_silence_is_ignored():
    st = Stats()
    x = _plucks(repeats=1)
    padded = torch.cat([torch.zeros(SR), x, torch.zeros(SR)])
    a, b = st(x), st(padded)
    # 무음과 맞닿은 경계 프레임 몇 개가 일부만 소리를 담아 1dB 미만 차이가 난다.
    # 무음 프레임이 섞였다면 차이는 수십 dB(바닥값 −100dB)가 된다.
    assert torch.allclose(a["ltas"], b["ltas"], atol=1.0)
