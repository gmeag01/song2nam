"""biquad 계수(RBJ), 주파수 영역 필터, lfilter 래퍼.

모든 함수는 배치를 지원한다. 계수 함수는 (B,) 모양의 텐서를 받아
b, a를 각각 (B, 3)으로 돌려준다(a0 = 1로 정규화).

- 학습 중: `freq_response`로 응답을 구해 곱한 뒤 `fft_filter`로 적용(FIR 근사).
- ⑦ 최종 렌더링·검증: `lfilter`(torchaudio, 시간 영역).

계수 공식은 R. Bristow-Johnson, "Cookbook formulae for audio EQ biquad filter
coefficients"를 따른다.
"""

from __future__ import annotations

import math

import torch

SR = 48000

# 설계 문서 4.4 구현 결정 6: 나이퀴스트 근처와 극단적 Q를 피한다.
F_MIN = 10.0
F_MAX_RATIO = 0.45  # × sr
Q_MIN = 0.1
Q_MAX = 10.0

# FIR 근사에서 IIR 꼬리가 순환 컨볼루션으로 감기지 않도록 붙이는 여유 길이.
DEFAULT_TAIL = 8192


def _prep(f0, q, sr):
    f0 = torch.as_tensor(f0)
    q = torch.as_tensor(q, dtype=f0.dtype, device=f0.device)
    f0 = f0.clamp(F_MIN, F_MAX_RATIO * sr)
    q = q.clamp(Q_MIN, Q_MAX)
    w0 = 2 * math.pi * f0 / sr
    cos_w0 = torch.cos(w0)
    alpha = torch.sin(w0) / (2 * q)
    return cos_w0, alpha


def _normalize(b0, b1, b2, a0, a1, a2):
    # 주파수가 고정(스칼라)이고 게인만 배치인 경우처럼 모양이 다를 수 있어 맞춰 준다.
    b0, b1, b2, a0, a1, a2 = torch.broadcast_tensors(
        *[torch.as_tensor(v) for v in (b0, b1, b2, a0, a1, a2)])
    b = torch.stack([b0, b1, b2], dim=-1) / a0.unsqueeze(-1)
    a = torch.stack([torch.ones_like(a0), a1 / a0, a2 / a0], dim=-1)
    return b, a


def lowpass(f0, q=1 / math.sqrt(2), sr=SR):
    c, al = _prep(f0, q, sr)
    b1 = 1 - c
    return _normalize(b1 / 2, b1, b1 / 2, 1 + al, -2 * c, 1 - al)


def highpass(f0, q=1 / math.sqrt(2), sr=SR):
    c, al = _prep(f0, q, sr)
    b1 = -(1 + c)
    return _normalize(-b1 / 2, b1, -b1 / 2, 1 + al, -2 * c, 1 - al)


def peaking(f0, gain_db, q=1.0, sr=SR):
    c, al = _prep(f0, q, sr)
    A = 10 ** (torch.as_tensor(gain_db, dtype=c.dtype, device=c.device) / 40)
    return _normalize(1 + al * A, -2 * c, 1 - al * A, 1 + al / A, -2 * c, 1 - al / A)


def _shelf(f0, gain_db, q, sr, high: bool):
    c, al = _prep(f0, q, sr)
    A = 10 ** (torch.as_tensor(gain_db, dtype=c.dtype, device=c.device) / 40)
    sA = 2 * torch.sqrt(A) * al
    s = 1 if high else -1
    # RBJ: 로우셸프(s=-1)와 하이셸프(s=+1)는 cos 항의 부호만 다르다.
    b0 = A * ((A + 1) + s * (A - 1) * c + sA)
    b1 = -2 * s * A * ((A - 1) + s * (A + 1) * c)
    b2 = A * ((A + 1) + s * (A - 1) * c - sA)
    a0 = (A + 1) - s * (A - 1) * c + sA
    a1 = 2 * s * ((A - 1) - s * (A + 1) * c)
    a2 = (A + 1) - s * (A - 1) * c - sA
    return _normalize(b0, b1, b2, a0, a1, a2)


def lowshelf(f0, gain_db, q=1 / math.sqrt(2), sr=SR):
    return _shelf(f0, gain_db, q, sr, high=False)


def highshelf(f0, gain_db, q=1 / math.sqrt(2), sr=SR):
    return _shelf(f0, gain_db, q, sr, high=True)


def onepole_lowpass(tau_sec, like: torch.Tensor, sr=SR):
    """1차 로우패스 y[n] = (1-p)x[n] + p·y[n-1] (엔벨로프 추적용, 지수 커널).

    biquad 모양(b, a 각 3개)으로 돌려줘서 같은 적용 함수를 쓴다.
    `like` (B,)와 같은 배치·dtype·장치로 만든다.
    """
    p = torch.full_like(like, math.exp(-1.0 / (tau_sec * sr)))
    z = torch.zeros_like(p)
    b = torch.stack([1 - p, z, z], dim=-1)
    a = torch.stack([torch.ones_like(p), -p, z], dim=-1)
    return b, a


# ---------------------------------------------------------------- 주파수 영역


def freq_response(b: torch.Tensor, a: torch.Tensor, n_fft: int) -> torch.Tensor:
    """rfft 빈(n_fft // 2 + 1개)에서 H(e^jw)를 구한다. 결과는 (..., F) 복소수."""
    k = torch.arange(n_fft // 2 + 1, device=b.device, dtype=b.dtype)
    w = 2 * math.pi * k / n_fft
    # z^-n = e^{-jwn}, n = 0, 1, 2
    n = torch.arange(3, device=b.device, dtype=b.dtype)
    zn = torch.polar(torch.ones_like(w)[:, None].expand(-1, 3), -w[:, None] * n)  # (F, 3)
    num = (b.unsqueeze(-2).to(zn.dtype) * zn).sum(-1)
    den = (a.unsqueeze(-2).to(zn.dtype) * zn).sum(-1)
    return num / den


def fft_size(n_samples: int, tail: int = DEFAULT_TAIL) -> int:
    return 1 << math.ceil(math.log2(n_samples + tail))


def fft_filter(x: torch.Tensor, H: torch.Tensor, n_fft: int) -> torch.Tensor:
    """x (..., T)에 주파수 응답 H (..., n_fft//2+1)를 곱한다(FIR 근사).

    n_fft는 T보다 충분히 커야 한다(`fft_size` 사용). 결과 길이는 T.
    """
    T = x.shape[-1]
    if n_fft < T:
        raise ValueError(f"n_fft({n_fft})가 신호 길이({T})보다 짧다")
    X = torch.fft.rfft(x, n=n_fft)
    return torch.fft.irfft(X * H, n=n_fft)[..., :T]


def apply_biquads_fft(x: torch.Tensor, coeffs: list[tuple[torch.Tensor, torch.Tensor]],
                      tail: int = DEFAULT_TAIL) -> torch.Tensor:
    """직렬 biquad 여러 개를 응답의 곱으로 한 번에 적용한다."""
    n_fft = fft_size(x.shape[-1], tail)
    H = None
    for b, a in coeffs:
        h = freq_response(b, a, n_fft)
        H = h if H is None else H * h
    return fft_filter(x, H, n_fft)


# ---------------------------------------------------------------- 시간 영역


def lfilter(x: torch.Tensor, b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """x (B, T), b·a (B, 3). 배치마다 다른 필터를 시간 영역에서 적용한다."""
    import torchaudio.functional as AF

    return AF.lfilter(x, a, b, clamp=False, batching=True)


def apply_biquads_lfilter(x: torch.Tensor,
                          coeffs: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    for b, a in coeffs:
        x = lfilter(x, b, a)
    return x


def apply_biquads(x: torch.Tensor, coeffs, mode: str = "fft") -> torch.Tensor:
    """mode: "fft"(학습용 FIR 근사) 또는 "iir"(torchaudio lfilter)."""
    if mode == "fft":
        return apply_biquads_fft(x, coeffs)
    if mode == "iir":
        return apply_biquads_lfilter(x, coeffs)
    raise ValueError(f"알 수 없는 필터 모드: {mode}")
