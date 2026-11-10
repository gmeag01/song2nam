"""G(x; θ), θ 분할과 단위 변환.

θ는 24차원, 0~1 정규화 벡터다(설계 문서 4.4). 블록 안에서 실제 단위로 바꾼다.
주파수("log")는 로그 스케일, 게인·기타("lin")는 선형 스케일로 변환한다.

설계 문서에 범위가 적혀 있지 않은 파라미터(부스터 레벨, 비대칭, 톤스택 미드 주파수,
프레즌스, 캐비닛 하이패스·공진·노치)는 아래 표의 값을 초기 가정으로 쓴다.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from . import blocks
from .filters import SR

# (이름, 스케일, 하한, 상한). 순서가 곧 θ의 차원 순서다.
PARAMS: list[tuple[str, str, float, float]] = [
    # 입력단 (2)
    ("in_hp_hz", "log", 40.0, 200.0),
    ("in_gain_db", "lin", -12.0, 12.0),
    # 부스터/OD (4)
    ("boost_mid_hz", "log", 500.0, 1500.0),
    ("boost_mid_db", "lin", 0.0, 12.0),
    ("boost_drive_db", "lin", 0.0, 30.0),
    ("boost_level_db", "lin", -12.0, 12.0),
    # 프리앰프 2단 (6)
    ("pre1_gain_db", "lin", 0.0, 40.0),
    ("pre1_bias", "lin", -0.5, 0.5),
    ("pre1_lp_hz", "log", 2000.0, 12000.0),
    ("pre2_gain_db", "lin", 0.0, 40.0),
    ("pre2_bias", "lin", -0.5, 0.5),
    ("pre2_lp_hz", "log", 2000.0, 12000.0),
    # 톤스택 (4)
    ("ts_bass_db", "lin", -12.0, 12.0),
    ("ts_mid_hz", "log", 250.0, 2500.0),
    ("ts_mid_db", "lin", -12.0, 12.0),
    ("ts_treble_db", "lin", -12.0, 12.0),
    # 파워앰프 (3)
    ("pa_drive_db", "lin", 0.0, 20.0),
    ("pa_sag", "lin", 0.0, 1.0),
    ("pa_presence_db", "lin", -12.0, 12.0),
    # 캐비닛 (5)
    ("cab_hp_hz", "log", 40.0, 200.0),
    ("cab_res_hz", "log", 60.0, 200.0),
    ("cab_lp_hz", "log", 3000.0, 7000.0),
    ("cab_notch_hz", "log", 1000.0, 5000.0),
    ("cab_notch_db", "lin", 0.0, 18.0),
]

N_PARAMS = len(PARAMS)
NAMES = [p[0] for p in PARAMS]
INDEX = {n: i for i, n in enumerate(NAMES)}

BLOCKS: dict[str, slice] = {
    "input": slice(0, 2),
    "boost": slice(2, 6),
    "preamp": slice(6, 12),
    "tonestack": slice(12, 16),
    "power": slice(16, 19),
    "cab": slice(19, 24),
}
BLOCK_ORDER = ["input", "boost", "preamp", "tonestack", "power", "cab"]

assert N_PARAMS == 24
assert sum(s.stop - s.start for s in BLOCKS.values()) == 24


# ---------------------------------------------------------------- 단위 변환


def denormalize(theta: torch.Tensor) -> dict[str, torch.Tensor]:
    """θ (..., 24) ∈ [0,1] → {이름: 실제 단위 (...)}."""
    out = {}
    for i, (name, scale, lo, hi) in enumerate(PARAMS):
        t = theta[..., i]
        if scale == "log":
            out[name] = lo * (hi / lo) ** t
        else:
            out[name] = lo + (hi - lo) * t
    return out


def normalize_value(name: str, value: float) -> float:
    """실제 단위 값 하나를 0~1로 바꾼다."""
    _, scale, lo, hi = PARAMS[INDEX[name]]
    if scale == "log":
        t = math.log(value / lo) / math.log(hi / lo)
    else:
        t = (value - lo) / (hi - lo)
    return min(max(t, 0.0), 1.0)


def theta_from_units(values: dict[str, float], base: torch.Tensor | None = None) -> torch.Tensor:
    """일부 파라미터를 실제 단위로 지정해 θ (24,)를 만든다. 나머지는 base(기본 중립값)."""
    theta = neutral_theta() if base is None else base.clone()
    for name, v in values.items():
        theta[INDEX[name]] = normalize_value(name, v)
    return theta


def theta_to_units_dict(theta: torch.Tensor) -> dict[str, float]:
    u = denormalize(theta.detach().double().cpu())
    return {k: float(v) for k, v in u.items()}


def neutral_theta() -> torch.Tensor:
    """선형 블록이 평탄해지는 값: EQ 게인 0dB, 비대칭 0, 드라이브 최소, 필터는 가장 넓게.

    전체 G가 항등이 되지는 않는다(tanh, 캐비닛 공진). 블록별 통과 테스트에 쓴다.
    """
    v = {
        "in_hp_hz": 40.0, "in_gain_db": 0.0,
        "boost_mid_hz": 800.0, "boost_mid_db": 0.0, "boost_drive_db": 0.0, "boost_level_db": 0.0,
        "pre1_gain_db": 0.0, "pre1_bias": 0.0, "pre1_lp_hz": 12000.0,
        "pre2_gain_db": 0.0, "pre2_bias": 0.0, "pre2_lp_hz": 12000.0,
        "ts_bass_db": 0.0, "ts_mid_hz": 800.0, "ts_mid_db": 0.0, "ts_treble_db": 0.0,
        "pa_drive_db": 0.0, "pa_sag": 0.0, "pa_presence_db": 0.0,
        "cab_hp_hz": 40.0, "cab_res_hz": 100.0, "cab_lp_hz": 7000.0,
        "cab_notch_hz": 2500.0, "cab_notch_db": 0.0,
    }
    t = torch.zeros(N_PARAMS, dtype=torch.float64)
    for n, x in v.items():
        t[INDEX[n]] = normalize_value(n, x)
    return t.float()


def rms(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x.pow(2).mean(-1).add(eps).sqrt()


# ---------------------------------------------------------------- G


class GrayboxAmp(nn.Module):
    """미분 가능한 그레이박스 앰프 G(x; θ).

    forward(x, theta)
      x:     (B, T) 48kHz 모노
      theta: (B, 24) 또는 (24,) ∈ [0,1]
      mode:  "fft"(학습, 주파수 영역 FIR 근사) / "iir"(lfilter, ⑦ 렌더링·검증)
      oversample: True면 2배 샘플레이트에서 전체 체인을 돌린다(⑥ 미세조정, ⑦ 최종 렌더링).
      target_rms: 지정하면 출력 RMS를 이 값으로 맞춘다(스템 RMS 정규화). 스칼라 또는 (B,).
      bypass: 건너뛸 블록 이름 집합(시각화·테스트용).
    """

    def __init__(self, sr: int = SR):
        super().__init__()
        self.sr = sr

    def forward(self, x: torch.Tensor, theta: torch.Tensor, mode: str = "fft",
                oversample: bool = False, target_rms=None, bypass=()) -> torch.Tensor:
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)
        B, T = x.shape
        if theta.dim() == 1:
            theta = theta.unsqueeze(0).expand(B, -1)
        if theta.shape != (B, N_PARAMS):
            raise ValueError(f"theta 모양 {tuple(theta.shape)}, 기대 ({B}, {N_PARAMS})")
        theta = theta.to(x.dtype)

        sr = self.sr
        if oversample:
            import torchaudio.functional as AF

            x = AF.resample(x, sr, 2 * sr)
            sr = 2 * sr

        p = denormalize(theta)
        y = x
        for name in BLOCK_ORDER:
            if name in bypass:
                continue
            y = blocks.BLOCK_FUNCS[name](y, p, sr, mode)

        if oversample:
            import torchaudio.functional as AF

            y = AF.resample(y, sr, self.sr)[..., :T]
            if y.shape[-1] < T:
                y = torch.nn.functional.pad(y, (0, T - y.shape[-1]))

        if target_rms is not None:
            tr = torch.as_tensor(target_rms, dtype=y.dtype, device=y.device)
            if tr.dim() == 1:
                tr = tr[:, None]
            y = y / rms(y)[:, None] * tr

        return y.squeeze(0) if squeeze else y

    @torch.no_grad()
    def render(self, x: torch.Tensor, theta: torch.Tensor, target_rms=None) -> torch.Tensor:
        """⑦ 최종 렌더링: lfilter + 2배 오버샘플링."""
        return self.forward(x, theta, mode="iir", oversample=True, target_rms=target_rms)

    @torch.no_grad()
    def block_responses(self, theta: torch.Tensor, n_fft: int = 8192) -> dict[str, torch.Tensor]:
        """블록별 선형 필터의 진폭 응답(dB). 시각화용. 반환: {블록: (F,)}, 'freqs': (F,)."""
        theta = theta.reshape(1, -1).double()
        p = denormalize(theta)
        out = {"freqs": torch.arange(n_fft // 2 + 1, dtype=torch.float64) * self.sr / n_fft}
        for name in BLOCK_ORDER:
            H = blocks.block_linear_response(name, p, self.sr, n_fft)
            out[name] = 20 * torch.log10(H.abs().clamp_min(1e-9))[0]
        return out
