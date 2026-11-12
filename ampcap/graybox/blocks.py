"""입력단, 부스트(오버드라이브), 프리앰프, 톤스택, 파워앰프, 캐비닛 블록.

각 블록은 (x (B, T), p {이름: (B,) 실제 단위}, sr, mode) → (B, T).
mode는 filters.apply_biquads의 "fft" / "iir".
고정 주파수·Q는 설계 문서에 적힌 블록 구성을 따른 초기 가정이다.
"""

from __future__ import annotations

import torch

from . import filters as F

BUTTER_Q = 0.7071067811865476
DC_BLOCK_HZ = 10.0
SAG_TAU_SEC = 0.05
SAG_DEPTH = 2.0

TS_BASS_HZ = 120.0
TS_MID_Q = 0.8
TS_TREBLE_HZ = 3200.0
PRESENCE_HZ = 4000.0
CAB_RES_DB = 6.0
CAB_RES_Q = 2.0
CAB_NOTCH_Q = 4.0
BOOST_MID_Q = 0.7


def db2lin(db: torch.Tensor) -> torch.Tensor:
    return 10 ** (db / 20)


def _full(like: torch.Tensor, v: float) -> torch.Tensor:
    return torch.full_like(like, v)


# ---------------------------------------------------------------- 계수 목록
# 선형 부분의 계수를 따로 만들어 두면 시각화(block_linear_response)에서 재사용할 수 있다.


def _input_coeffs(p, sr):
    return [F.highpass(p["in_hp_hz"], BUTTER_Q, sr)]


def _boost_coeffs(p, sr):
    return [F.peaking(p["boost_mid_hz"], p["boost_mid_db"], BOOST_MID_Q, sr)]


def _pre_coeffs(p, sr, i):
    lp = p[f"pre{i}_lp_hz"]
    return [F.highpass(_full(lp, DC_BLOCK_HZ), BUTTER_Q, sr), F.lowpass(lp, BUTTER_Q, sr)]


def _ts_coeffs(p, sr):
    bass = p["ts_bass_db"]
    return [
        F.lowshelf(_full(bass, TS_BASS_HZ), bass, BUTTER_Q, sr),
        F.peaking(p["ts_mid_hz"], p["ts_mid_db"], TS_MID_Q, sr),
        F.highshelf(_full(bass, TS_TREBLE_HZ), p["ts_treble_db"], BUTTER_Q, sr),
    ]


def _presence_coeffs(p, sr):
    pr = p["pa_presence_db"]
    return [F.highshelf(_full(pr, PRESENCE_HZ), pr, BUTTER_Q, sr)]


def _cab_coeffs(p, sr):
    res = p["cab_res_hz"]
    lp = p["cab_lp_hz"]
    return [
        F.highpass(p["cab_hp_hz"], 0.8, sr),
        F.peaking(res, _full(res, CAB_RES_DB), CAB_RES_Q, sr),
        # 스피커의 가파른 고역 감쇠를 흉내 내려고 같은 로우패스를 두 번(4차) 건다.
        F.lowpass(lp, BUTTER_Q, sr),
        F.lowpass(lp, BUTTER_Q, sr),
        F.peaking(p["cab_notch_hz"], -p["cab_notch_db"], CAB_NOTCH_Q, sr),
    ]


# ---------------------------------------------------------------- 블록


def input_stage(x, p, sr, mode):
    """하이패스 → 게인."""
    x = F.apply_biquads(x, _input_coeffs(p, sr), mode)
    return x * db2lin(p["in_gain_db"])[:, None]


def booster(x, p, sr, mode):
    """피킹 EQ → 드라이브 게인 → tanh → 레벨."""
    x = F.apply_biquads(x, _boost_coeffs(p, sr), mode)
    x = torch.tanh(x * db2lin(p["boost_drive_db"])[:, None])
    return x * db2lin(p["boost_level_db"])[:, None]


def _preamp_stage(x, p, sr, mode, i):
    """게인 → DC 바이어스 → tanh → DC 제거 → 로우패스."""
    g = db2lin(p[f"pre{i}_gain_db"])[:, None]
    bias = p[f"pre{i}_bias"][:, None]
    # tanh(bias)를 빼서 무음일 때 출력이 0이 되게 한다(정적 DC 제거). 남는 DC는 하이패스가 지운다.
    x = torch.tanh(x * g + bias) - torch.tanh(bias)
    return F.apply_biquads(x, _pre_coeffs(p, sr, i), mode)


def preamp(x, p, sr, mode):
    x = _preamp_stage(x, p, sr, mode, 1)
    return _preamp_stage(x, p, sr, mode, 2)


def tonestack(x, p, sr, mode):
    """로우셸프 → 피킹 → 하이셸프."""
    return F.apply_biquads(x, _ts_coeffs(p, sr), mode)


def power_amp(x, p, sr, mode):
    """드라이브 게인 → tanh → 새그 → 하이셸프(프레즌스).

    새그: |y|를 지수 커널(1차 로우패스)로 추적한 엔벨로프만큼 게인을 줄인다.
    """
    y = torch.tanh(x * db2lin(p["pa_drive_db"])[:, None])
    sag = p["pa_sag"]
    env = F.apply_biquads(y.abs(), [F.onepole_lowpass(SAG_TAU_SEC, sag, sr)], mode)
    y = y / (1 + SAG_DEPTH * sag[:, None] * env.clamp_min(0))
    return F.apply_biquads(y, _presence_coeffs(p, sr), mode)


def cabinet(x, p, sr, mode):
    """하이패스 → 피킹(공진) → 로우패스 → 피킹(음수 게인, 딥)."""
    return F.apply_biquads(x, _cab_coeffs(p, sr), mode)


BLOCK_FUNCS = {
    "input": input_stage,
    "boost": booster,
    "preamp": preamp,
    "tonestack": tonestack,
    "power": power_amp,
    "cab": cabinet,
}


def block_linear_response(name, p, sr, n_fft):
    """블록 안 선형 필터들의 합성 주파수 응답 (B, F). 비선형 부분은 제외한다."""
    if name == "input":
        co = _input_coeffs(p, sr)
    elif name == "boost":
        co = _boost_coeffs(p, sr)
    elif name == "preamp":
        co = _pre_coeffs(p, sr, 1) + _pre_coeffs(p, sr, 2)
    elif name == "tonestack":
        co = _ts_coeffs(p, sr)
    elif name == "power":
        co = _presence_coeffs(p, sr)
    elif name == "cab":
        co = _cab_coeffs(p, sr)
    else:
        raise KeyError(name)
    H = None
    for b, a in co:
        h = F.freq_response(b, a, n_fft)
        H = h if H is None else H * h
    if name == "input":
        H = H * db2lin(p["in_gain_db"])[:, None]
    return H
