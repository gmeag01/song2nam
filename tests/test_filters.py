"""③ 필터: scipy freqz·sosfilt 기준 비교, gradcheck(double), 주파수 영역 vs lfilter."""

import numpy as np
import pytest
import torch
from scipy import signal

from ampcap.graybox import filters as F

SR = 48000

CASES = [
    ("lowpass", lambda d: F.lowpass(torch.tensor([1000.0, 6000.0], dtype=d), 0.7071, SR)),
    ("highpass", lambda d: F.highpass(torch.tensor([60.0, 150.0], dtype=d), 0.9, SR)),
    ("peaking", lambda d: F.peaking(torch.tensor([800.0, 3000.0], dtype=d),
                                    torch.tensor([6.0, -9.0], dtype=d), 2.0, SR)),
    ("lowshelf", lambda d: F.lowshelf(torch.tensor([120.0, 300.0], dtype=d),
                                      torch.tensor([8.0, -6.0], dtype=d), 0.7071, SR)),
    ("highshelf", lambda d: F.highshelf(torch.tensor([3200.0, 5000.0], dtype=d),
                                        torch.tensor([-10.0, 12.0], dtype=d), 0.7071, SR)),
]


def _rbj_reference(kind, f0, gain_db, q):
    """RBJ 쿡북 공식을 numpy로 독립 구현한 기준값."""
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / SR
    c, s = np.cos(w0), np.sin(w0)
    al = s / (2 * q)
    if kind == "lowpass":
        b = [(1 - c) / 2, 1 - c, (1 - c) / 2]; a = [1 + al, -2 * c, 1 - al]
    elif kind == "highpass":
        b = [(1 + c) / 2, -(1 + c), (1 + c) / 2]; a = [1 + al, -2 * c, 1 - al]
    elif kind == "peaking":
        b = [1 + al * A, -2 * c, 1 - al * A]; a = [1 + al / A, -2 * c, 1 - al / A]
    elif kind == "lowshelf":
        sa = 2 * np.sqrt(A) * al
        b = [A * ((A + 1) - (A - 1) * c + sa), 2 * A * ((A - 1) - (A + 1) * c), A * ((A + 1) - (A - 1) * c - sa)]
        a = [(A + 1) + (A - 1) * c + sa, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - sa]
    else:
        sa = 2 * np.sqrt(A) * al
        b = [A * ((A + 1) + (A - 1) * c + sa), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - sa)]
        a = [(A + 1) - (A - 1) * c + sa, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - sa]
    b, a = np.array(b), np.array(a)
    return b / a[0], a / a[0]


REF_PARAMS = {
    "lowpass": [(1000.0, 0, 0.7071), (6000.0, 0, 0.7071)],
    "highpass": [(60.0, 0, 0.9), (150.0, 0, 0.9)],
    "peaking": [(800.0, 6.0, 2.0), (3000.0, -9.0, 2.0)],
    "lowshelf": [(120.0, 8.0, 0.7071), (300.0, -6.0, 0.7071)],
    "highshelf": [(3200.0, -10.0, 0.7071), (5000.0, 12.0, 0.7071)],
}


@pytest.mark.parametrize("kind,make", CASES, ids=[c[0] for c in CASES])
def test_coeffs_match_rbj_reference(kind, make):
    b, a = make(torch.float64)
    for i, (f0, g, q) in enumerate(REF_PARAMS[kind]):
        rb, ra = _rbj_reference(kind, f0, g, q)
        np.testing.assert_allclose(b[i].numpy(), rb, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(a[i].numpy(), ra, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("kind,make", CASES, ids=[c[0] for c in CASES])
def test_freq_response_matches_scipy_freqz(kind, make):
    n_fft = 4096
    b, a = make(torch.float64)
    H = F.freq_response(b, a, n_fft).numpy()
    w = 2 * np.pi * np.arange(n_fft // 2 + 1) / n_fft
    for i in range(b.shape[0]):
        _, h = signal.freqz(b[i].numpy(), a[i].numpy(), worN=w)
        np.testing.assert_allclose(H[i], h, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("kind,make", CASES, ids=[c[0] for c in CASES])
def test_lfilter_matches_scipy_sosfilt(kind, make):
    b, a = make(torch.float64)
    x = torch.randn(b.shape[0], 4000, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    y = F.lfilter(x, b, a).numpy()
    for i in range(b.shape[0]):
        sos = np.concatenate([b[i].numpy(), a[i].numpy()])[None]
        ref = signal.sosfilt(sos, x[i].numpy())
        np.testing.assert_allclose(y[i], ref, rtol=1e-7, atol=1e-9)


def test_fft_filter_matches_lfilter():
    """주파수 영역 FIR 근사 ≈ 시간 영역 IIR (직렬 5개)."""
    d = torch.float64
    co = [m(d) for _, m in CASES]
    x = torch.randn(2, 20000, dtype=d, generator=torch.Generator().manual_seed(1))
    y_fft = F.apply_biquads_fft(x, co)
    y_iir = F.apply_biquads_lfilter(x, co)
    err = (y_fft - y_iir).square().sum() / y_iir.square().sum()
    assert err < 1e-8


def test_gradcheck_freq_response():
    def fn(f0, g, q):
        b, a = F.peaking(f0, g, q, SR)
        H = F.freq_response(b, a, 256)
        return torch.view_as_real(H)

    f0 = torch.tensor([700.0, 2500.0], dtype=torch.float64, requires_grad=True)
    g = torch.tensor([4.0, -6.0], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(fn, (f0, g, q), eps=1e-6, atol=1e-5)


def test_gradcheck_fft_filter_wrt_signal_and_params():
    def fn(x, f0, g):
        co = [F.highshelf(f0, g, 0.7071, SR), F.lowpass(f0 * 2, 0.7071, SR)]
        return F.apply_biquads_fft(x, co, tail=256)

    x = torch.randn(2, 128, dtype=torch.float64, requires_grad=True)
    f0 = torch.tensor([2000.0, 4000.0], dtype=torch.float64, requires_grad=True)
    g = torch.tensor([3.0, -5.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(fn, (x, f0, g), eps=1e-6, atol=1e-5)


def test_frequency_and_q_are_clamped():
    b, a = F.lowpass(torch.tensor([30000.0, 1.0]), 100.0, SR)
    assert torch.isfinite(b).all() and torch.isfinite(a).all()
    # 클램프 후 극점은 단위원 안쪽(안정)
    for i in range(2):
        assert np.all(np.abs(np.roots(a[i].numpy())) < 1)


def test_fixed_frequency_with_batched_gain():
    """주파수가 스칼라, 게인만 배치인 경우도 모양이 맞아야 한다."""
    b, a = F.peaking(torch.tensor(1000.0), torch.tensor([0.0, 6.0, -6.0]), 1.0, SR)
    assert b.shape == (3, 3) and a.shape == (3, 3)
