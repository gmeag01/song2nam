"""② 세션 스키마 검증, 대응표, 사전 범위, 세션 벡터.

대응표 v2 값은 회로·측정 자료로 다시 검증한 범위다(설계 문서 4.3). 근거가 약한 칸이 남아 있다.
값이나 사전 범위 규칙을 바꾸면 `MAPPING_VERSION`을 올린다.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .graybox.amp import INDEX, N_PARAMS, NAMES, normalize_value

# ---------------------------------------------------------------- 스키마

PICKUPS = ["single", "p90", "humbucker"]
FAMILIES = ["fender", "vox", "british", "american_hg", "modern_hg"]
GAINS = ["low", "mid", "high"]
CABS = ["1x12", "2x12", "4x12"]
BOOSTS = ["none", "tube_screamer"]  # tube_screamer: Tube Screamer 계열 오버드라이브
TARGETS = ["rhythm", "lead"]

# 세션 벡터 순서: 픽업 4, 앰프 계열 6, 게인 4, 캐비닛 4, 부스트 3 = 21
FIELDS: list[tuple[str, list[str], str]] = [
    # (필드 경로, 선택지(unknown 제외), confidence 키)
    ("guitar.pickup", PICKUPS, "guitar"),
    ("amp.family", FAMILIES, "amp"),
    ("amp.gain", GAINS, "amp"),
    ("cab.size", CABS, "cab"),
    ("boost", BOOSTS, "boost"),
]
SESSION_DIM = sum(len(opts) + 1 for _, opts, _ in FIELDS)
assert SESSION_DIM == 21

# 알려진 모델명 → 계열 (소문자, 공백·하이픈 제거 후 부분 일치)
KNOWN_MODELS: dict[str, str] = {
    "twin": "fender", "deluxe": "fender", "bassman": "fender", "princeton": "fender",
    "fender": "fender", "hotrod": "fender", "bluesjr": "fender",
    "ac30": "vox", "ac15": "vox", "vox": "vox",
    "jcm800": "british", "jcm900": "british", "plexi": "british", "jtm45": "british",
    "superlead": "british", "1959": "british", "1987": "british", "orange": "british",
    "rockerverb": "british", "jvm": "british", "silverjubilee": "british", "marshall": "british",
    "rectifier": "american_hg", "dualrec": "american_hg", "5150": "american_hg",
    "6505": "american_hg", "markiv": "american_hg", "markiic": "american_hg", "mesa": "american_hg",
    "evh": "american_hg", "soldano": "american_hg", "slo100": "american_hg",
    "diezel": "modern_hg", "vh4": "modern_hg", "herbert": "modern_hg", "engl": "modern_hg",
    "powerball": "modern_hg", "uberschall": "modern_hg", "fortin": "modern_hg",
    "revv": "modern_hg", "friedman": "modern_hg", "axefx": "modern_hg", "kemper": "modern_hg",
}


class SessionError(ValueError):
    pass


def _get(d: dict, path: str):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _set(d: dict, path: str, v):
    keys = path.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = v


def model_to_family(model: str | None) -> str | None:
    if not model:
        return None
    key = re.sub(r"[\s\-_.]", "", model.lower())
    for name, fam in KNOWN_MODELS.items():
        if name in key:
            return fam
    return None


def validate(session: dict) -> dict:
    """스키마를 검증하고 빈 필드를 unknown으로 채운 사본을 돌려준다. 잘못된 값은 거부."""
    if not isinstance(session, dict):
        raise SessionError("세션은 JSON 객체여야 한다")
    s = json.loads(json.dumps(session))  # 깊은 복사

    tgt = s.get("capture_target", "rhythm")
    if tgt not in TARGETS:
        raise SessionError(f"capture_target 값이 잘못됨: {tgt!r} (가능: {TARGETS})")
    s["capture_target"] = tgt
    same_rig = s.get("same_rig", True)
    if not isinstance(same_rig, bool):
        raise SessionError("same_rig는 true/false")
    s["same_rig"] = same_rig

    # 모델명으로 계열 채우기
    fam = _get(s, "amp.family")
    if fam in (None, "unknown"):
        guess = model_to_family(_get(s, "amp.model"))
        if guess:
            _set(s, "amp.family", guess)

    if s.get("boost") == "ts":  # 예전 session.json 호환 (톤스택 θ 접두사 ts_와 겹쳐 이름을 바꿈)
        s["boost"] = "tube_screamer"

    for path, opts, _ in FIELDS:
        v = _get(s, path)
        if v is None:
            v = "unknown"
        if not isinstance(v, str) or (v not in opts and v != "unknown"):
            raise SessionError(f"{path} 값이 잘못됨: {v!r} (가능: {opts + ['unknown']})")
        _set(s, path, v)

    conf = s.get("confidence") or {}
    if not isinstance(conf, dict):
        raise SessionError("confidence는 객체여야 한다")
    clean = {}
    for key in {c for _, _, c in FIELDS}:
        c = conf.get(key, 1.0)
        if not isinstance(c, (int, float)) or not 0.0 <= float(c) <= 1.0:
            raise SessionError(f"confidence.{key}는 0~1 숫자여야 한다: {c!r}")
        clean[key] = float(c)
    extra = set(conf) - set(clean)
    if extra:
        raise SessionError(f"알 수 없는 confidence 키: {sorted(extra)}")
    s["confidence"] = clean
    return s


def load_session(path: str | Path) -> dict:
    return validate(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------- 대응표
# 요소 → {θ 이름: (하한, 상한)} (0~1 정규화 값). 실제 단위로 적고 싶으면 _u()를 쓴다.


def _u(name: str, lo: float, hi: float) -> tuple[float, float]:
    """실제 단위 범위를 0~1 범위로."""
    a, b = normalize_value(name, lo), normalize_value(name, hi)
    return (min(a, b), max(a, b))


# 대응표 버전: MAPPING 값이나 사전 범위 규칙(게인 우선 규칙 등)을 바꾸면 올린다. θ 버전과 별개.
MAPPING_VERSION = 2

MAPPING: dict[str, dict[str, dict[str, tuple[float, float]]]] = {
    "guitar.pickup": {
        # in_hp_hz는 v2에서 계열로 이동 (픽업은 LC 저역통과, 종류별 하이패스 근거 없음)
        "single": {"in_gain_db": _u("in_gain_db", -10, 0)},
        "p90": {"in_gain_db": _u("in_gain_db", -4, 4)},
        "humbucker": {"in_gain_db": _u("in_gain_db", -2, 11)},
    },
    "amp.family": {
        "fender": {
            "pre1_gain_db": _u("pre1_gain_db", 0, 18), "pre2_gain_db": _u("pre2_gain_db", 0, 16),
            "pre1_bias": _u("pre1_bias", -0.1, 0.1),
            "ts_bass_db": _u("ts_bass_db", -6, 12), "ts_mid_hz": _u("ts_mid_hz", 300, 1250),
            "ts_mid_db": _u("ts_mid_db", -12, -3), "ts_treble_db": _u("ts_treble_db", -5, 12),
            "pa_drive_db": _u("pa_drive_db", 2, 18), "pa_sag": (0.1, 0.8),
            "pa_presence_db": _u("pa_presence_db", -6, 6), "in_hp_hz": _u("in_hp_hz", 40, 80),
        },
        "vox": {
            "pre1_gain_db": _u("pre1_gain_db", 0, 26), "pre2_gain_db": _u("pre2_gain_db", 0, 20),
            "ts_mid_hz": _u("ts_mid_hz", 450, 1350), "ts_mid_db": _u("ts_mid_db", -12, -4),
            "ts_treble_db": _u("ts_treble_db", -5, 12),
            "pa_drive_db": _u("pa_drive_db", 3, 20), "pa_sag": (0.2, 0.9),
            "pa_presence_db": _u("pa_presence_db", -9, 6), "in_hp_hz": _u("in_hp_hz", 40, 150),
        },
        "british": {
            "pre1_gain_db": _u("pre1_gain_db", 12, 36), "pre2_gain_db": _u("pre2_gain_db", 10, 34),
            "pre1_bias": _u("pre1_bias", 0.0, 0.35),
            "ts_bass_db": _u("ts_bass_db", -3, 8), "ts_mid_hz": _u("ts_mid_hz", 400, 1350),
            "ts_mid_db": _u("ts_mid_db", -11, -2), "ts_treble_db": _u("ts_treble_db", -6, 3),
            "pa_drive_db": _u("pa_drive_db", 4, 18), "pa_sag": (0.1, 0.6),
            "in_hp_hz": _u("in_hp_hz", 60, 160),
        },
        "american_hg": {
            "pre1_gain_db": _u("pre1_gain_db", 22, 40), "pre2_gain_db": _u("pre2_gain_db", 18, 40),
            "pre1_bias": _u("pre1_bias", 0.0, 0.35),
            "ts_bass_db": _u("ts_bass_db", -6, 9), "ts_mid_hz": _u("ts_mid_hz", 300, 1150),
            "ts_mid_db": _u("ts_mid_db", -12, -2), "ts_treble_db": _u("ts_treble_db", -5, 9),
            "pa_drive_db": _u("pa_drive_db", 2, 10), "pa_sag": (0.0, 0.5),
            "in_hp_hz": _u("in_hp_hz", 70, 180),
        },
        "modern_hg": {
            "pre1_gain_db": _u("pre1_gain_db", 24, 40), "pre2_gain_db": _u("pre2_gain_db", 20, 40),
            "ts_mid_db": _u("ts_mid_db", -12, -1), "in_hp_hz": _u("in_hp_hz", 80, 200),
            "pa_drive_db": _u("pa_drive_db", 0, 8), "pa_sag": (0.0, 0.3),
        },
    },
    "amp.gain": {
        "low": {"pre1_gain_db": _u("pre1_gain_db", 0, 14), "pre2_gain_db": _u("pre2_gain_db", 0, 12),
                "boost_drive_db": _u("boost_drive_db", 0, 10)},
        "mid": {"pre1_gain_db": _u("pre1_gain_db", 10, 28), "pre2_gain_db": _u("pre2_gain_db", 6, 24)},
        "high": {"pre1_gain_db": _u("pre1_gain_db", 22, 40), "pre2_gain_db": _u("pre2_gain_db", 18, 40)},
    },
    "cab.size": {
        "1x12": {"cab_hp_hz": _u("cab_hp_hz", 60, 160), "cab_res_hz": _u("cab_res_hz", 70, 130),
                 "cab_lp_hz": _u("cab_lp_hz", 3500, 6500)},
        "2x12": {"cab_hp_hz": _u("cab_hp_hz", 50, 130), "cab_res_hz": _u("cab_res_hz", 75, 140)},
        "4x12": {"cab_hp_hz": _u("cab_hp_hz", 40, 100), "cab_res_hz": _u("cab_res_hz", 65, 130),
                 "cab_lp_hz": _u("cab_lp_hz", 3000, 6000)},
    },
    "boost": {
        "none": {"boost_drive_db": (0.0, 0.0), "boost_mid_db": (0.0, 0.0),
                 "boost_level_db": (0.5, 0.5)},
        "tube_screamer": {"boost_mid_hz": _u("boost_mid_hz", 600, 1000), "boost_mid_db": _u("boost_mid_db", 8, 12),
                          "boost_drive_db": _u("boost_drive_db", 6, 30), "boost_level_db": _u("boost_level_db", 0, 10)},
    },
}


# ---------------------------------------------------------------- 사전 범위
# Prior는 확률분포가 아니라 차원별 [lo, hi]와 중심값이다. ④에서만 이 범위에서 균등하게 뽑는다.


@dataclass
class Prior:
    lo: np.ndarray            # (24,)
    hi: np.ndarray            # (24,)
    center: np.ndarray        # (24,)
    warnings: list[str] = field(default_factory=list)

    @property
    def fixed(self) -> np.ndarray:
        """하한=상한인 차원(예: boost=none의 드라이브)."""
        return np.isclose(self.lo, self.hi)

    def sample(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.uniform(self.lo, self.hi, size=(n, N_PARAMS))

    def to_dict(self) -> dict:
        return {
            "names": NAMES,
            "lo": self.lo.round(4).tolist(), "hi": self.hi.round(4).tolist(),
            "center": self.center.round(4).tolist(), "warnings": self.warnings,
            "mapping_version": MAPPING_VERSION,
        }

    @staticmethod
    def full() -> "Prior":
        lo, hi = np.zeros(N_PARAMS), np.ones(N_PARAMS)
        return Prior(lo, hi, (lo + hi) / 2, [])


def _widen(lo: float, hi: float, c: float) -> tuple[float, float]:
    """confidence c<1이면 그만큼 전체 범위 [0,1] 쪽으로 넓힌다."""
    return lo * c, hi * c + (1 - c)


def compute_prior(session: dict) -> Prior:
    s = validate(session)
    ranges: dict[int, list[tuple[float, float, str]]] = {}
    for path, _, ckey in FIELDS:
        v = _get(s, path)
        if v == "unknown":
            continue
        c = s["confidence"][ckey]
        for name, (lo, hi) in MAPPING[path][v].items():
            lo, hi = _widen(lo, hi, c)
            ranges.setdefault(INDEX[name], []).append((lo, hi, f"{path}={v}"))

    lo = np.zeros(N_PARAMS)
    hi = np.ones(N_PARAMS)
    warnings = []
    for i, rs in ranges.items():
        a = max(r[0] for r in rs)
        b = min(r[1] for r in rs)
        if a > b + 1e-9:  # 빈 교집합: 게인 우선 규칙(amp.gain 범위), amp.gain이 없으면 합집합
            gain = [r for r in rs if r[2].startswith("amp.gain=")]
            a = gain[0][0] if gain else min(r[0] for r in rs)
            b = gain[0][1] if gain else max(r[1] for r in rs)
            warnings.append(f"range_conflict:{NAMES[i]}:" + ",".join(r[2] for r in rs))
        lo[i], hi[i] = a, max(a, b)
    return Prior(lo, hi, (lo + hi) / 2, warnings)


# ---------------------------------------------------------------- 세션 벡터


def session_vector(session: dict) -> np.ndarray:
    """21차원: 필드별 unknown 포함 원-핫 × confidence (unknown 칸은 1)."""
    s = validate(session)
    out = []
    for path, opts, ckey in FIELDS:
        v = _get(s, path)
        oh = np.zeros(len(opts) + 1, dtype=np.float32)
        if v == "unknown":
            oh[-1] = 1.0
        else:
            oh[opts.index(v)] = s["confidence"][ckey]
        out.append(oh)
    return np.concatenate(out)


def random_session(rng: np.random.Generator, p_unknown: float = 0.0) -> dict:
    """④ 합성 데이터용 무작위 세션."""
    s: dict = {"capture_target": "rhythm", "same_rig": True}
    for path, opts, _ in FIELDS:
        v = "unknown" if rng.random() < p_unknown else str(rng.choice(opts))
        _set(s, path, v)
    s["confidence"] = {c: 1.0 for _, _, c in FIELDS}
    return s


def dropout_session(session: dict, rng: np.random.Generator, p_unknown: float = 0.3,
                    p_lower_conf: float = 0.3) -> dict:
    """조건 드롭아웃(필드를 일정 확률로 unknown으로 가림)과 confidence 증강(무작위로 낮춤)."""
    s = validate(session)
    for path, _, ckey in FIELDS:
        if rng.random() < p_unknown:
            _set(s, path, "unknown")
    for ckey in list(s["confidence"]):
        if rng.random() < p_lower_conf:
            s["confidence"][ckey] = float(rng.uniform(0.3, 1.0))
    return s


def unknown_session() -> dict:
    return validate({})
