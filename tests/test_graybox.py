"""③ G: 블록 중립 θ 통과, gradcheck, 주파수 샘플링 방식 vs lfilter, 처리 속도."""

import time

import numpy as np
import torch

from ampcap.graybox import blocks
from ampcap.graybox.amp import (BLOCK_ORDER, INDEX, N_PARAMS, GrayboxAmp, denormalize, neutral_theta,
                                normalize_value, theta_from_units)

SR = 48000


def _tone(T=24000, amp=1e-3, f=440.0, dtype=torch.float64):
    t = torch.arange(T, dtype=dtype) / SR
    x = amp * (torch.sin(2 * np.pi * f * t) + 0.5 * torch.sin(2 * np.pi * 2.3 * f * t))
    return x[None]


def _esr(a, b):
    return float((a - b).square().sum() / b.square().sum())


def _mag_err(y, x, skip=4000):
    """진폭 스펙트럼 상대 오차. 필터의 작은 위상 변화는 무시하고 "같은 소리인지"만 본다."""
    w = torch.hann_window(x.shape[-1] - skip, dtype=x.dtype)
    X = torch.fft.rfft(x[..., skip:] * w).abs()
    Y = torch.fft.rfft(y[..., skip:] * w).abs()
    return float((Y - X).square().sum() / X.square().sum())


def test_param_table_roundtrip():
    th = torch.rand(N_PARAMS, dtype=torch.float64)
    u = denormalize(th)
    for name, v in u.items():
        assert abs(normalize_value(name, float(v)) - float(th[INDEX[name]])) < 1e-9


def test_linear_blocks_pass_at_neutral():
    """EQ 블록(입력단, 톤스택, 프레즌스 제외 파워앰프의 선형부)은 중립 θ에서 평탄."""
    p = denormalize(neutral_theta().double()[None])
    x = _tone(amp=0.3)
    y = blocks.tonestack(x, p, SR, "fft")
    assert _esr(y[:, 2000:], x[:, 2000:]) < 1e-6
    y = blocks.input_stage(x, p, SR, "fft")  # 40Hz 하이패스만 → 440Hz 대역은 거의 그대로
    assert _mag_err(y, x) < 1e-3


def test_nonlinear_blocks_pass_small_signal_at_neutral():
    """드라이브 0dB·비대칭 0에서 작은 신호는 부스트·프리앰프·파워앰프를 거의 그대로 통과."""
    p = denormalize(neutral_theta().double()[None])
    x = _tone(amp=1e-3)
    for name in ["boost", "preamp", "power"]:
        y = blocks.BLOCK_FUNCS[name](x, p, SR, "fft")
        assert _mag_err(y, x) < 1e-3, name


def test_full_amp_passes_with_cab_bypassed():
    G = GrayboxAmp()
    x = _tone(amp=1e-3)
    y = G(x, neutral_theta().double(), bypass={"cab"})
    assert _mag_err(y, x) < 1e-3


def test_batch_and_shapes():
    torch.manual_seed(0)
    G = GrayboxAmp()
    x = torch.randn(3, 9600) * 0.05
    th = torch.rand(3, N_PARAMS)
    y = G(x, th)
    assert y.shape == x.shape and torch.isfinite(y).all()
    # 배치 안의 각 예시는 독립. float32 FFT는 배치 크기에 따라 계산 순서가 달라 1e-5 수준 차이가 난다
    # (배치끼리 섞이면 차이는 신호 크기만큼 난다).
    y1 = G(x[1:2], th[1:2])
    assert torch.allclose(y[1:2], y1, atol=1e-4)


def test_target_rms_normalization():
    G = GrayboxAmp()
    y = G(torch.randn(2, 4800) * 0.05, torch.rand(2, N_PARAMS), target_rms=0.2)
    assert torch.allclose(y.square().mean(-1).sqrt(), torch.full((2,), 0.2), atol=1e-4)


def test_gradcheck_full_amp():
    G = GrayboxAmp()
    x = (torch.randn(1, 96, dtype=torch.float64) * 0.05)
    th = (torch.rand(1, N_PARAMS, dtype=torch.float64) * 0.8 + 0.1).requires_grad_(True)

    def fn(t):
        return G(x, t, mode="fft")

    # 블록이 많아 수치 미분 오차가 커질 수 있어 허용 오차를 조금 넓힌다.
    assert torch.autograd.gradcheck(fn, (th,), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_fft_mode_matches_iir_mode():
    G = GrayboxAmp()
    x = torch.randn(2, 24000, dtype=torch.float64, generator=torch.Generator().manual_seed(0)) * 0.05
    th = torch.rand(2, N_PARAMS, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    y_fft = G(x, th, mode="fft")
    y_iir = G(x, th, mode="iir")
    assert _esr(y_fft, y_iir) < 1e-4


def test_oversampled_render_close_to_base_rate():
    """오버샘플링은 앨리어싱만 줄이므로 기본 렌더와 크게 다르지 않다(저게인)."""
    G = GrayboxAmp()
    x = _tone(T=24000, amp=0.02, dtype=torch.float32)
    th = theta_from_units({"pre1_gain_db": 6.0, "pre2_gain_db": 0.0})
    y0 = G(x, th, mode="iir")
    y1 = G.render(x, th)
    assert y1.shape == y0.shape
    assert _esr(y1[:, 4000:], y0[:, 4000:]) < 0.05


def test_speed_allows_realtime_synthesis():
    """④ 실시간 생성: 3초 클립 8개 렌더가 오디오 길이(24초)보다 충분히 빨라야 한다."""
    G = GrayboxAmp()
    x = torch.randn(8, 3 * SR) * 0.05
    th = torch.rand(8, N_PARAMS)
    G(x[:1, :4800], th[:1])  # 워밍업
    t0 = time.perf_counter()
    with torch.no_grad():
        G(x, th)
    dt = time.perf_counter() - t0
    print(f"G 렌더 8×3초: {dt:.3f}s (실시간 대비 {24 / dt:.0f}배)")
    assert dt < 24.0


def test_block_responses_for_visualization():
    G = GrayboxAmp()
    r = G.block_responses(torch.rand(N_PARAMS))
    assert set(BLOCK_ORDER) <= set(r)
    assert all(torch.isfinite(r[b]).all() for b in BLOCK_ORDER)


def test_fit_render_matches_final_render_high_gain():
    """⑥(FFT 필터 + 2배 오버샘플링)과 ⑦ 렌더링(lfilter + 2배 오버샘플링)이 같은 소리를 낸다."""
    G = GrayboxAmp()
    x = _tone(T=3 * SR, amp=0.05, f=110.0, dtype=torch.float32)
    th = theta_from_units({"pre1_gain_db": 36.0, "pre2_gain_db": 30.0, "boost_drive_db": 20.0,
                           "pa_drive_db": 12.0})
    with torch.no_grad():
        y6 = G(x, th, mode="fft", oversample=True)
        y7 = G.render(x, th)
    assert _esr(y6, y7) < 1e-4
