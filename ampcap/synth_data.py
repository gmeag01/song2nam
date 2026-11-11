"""④ 합성 데이터 생성기와 EGDB DI 분할.

샘플 하나의 생성 과정(설계 문서 4.5):
1. 세션 필드 값을 무작위로 정하고 ②로 사전 범위를 만든다. 약 20%는 θ 전체 범위에서 뽑는다.
2. 사전 범위 안에서 θ를 균등하게 뽑는다.
3. 세션에 조건 드롭아웃(unknown 마스킹)과 confidence 증강을 적용한다.
4. EGDB DI에서 3초를 잘라 DI 레벨을 무작위로 바꾼다(DI 레벨 증강).
5. y = G(x; θ)를 렌더링한다(GPU).
6. pedalboard로 무작위 EQ, 컴프, 짧은 리버브 증강을 건다.

데이터는 학습 중 바로 만들고 저장하지 않는다.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .graybox.amp import GrayboxAmp, rms
from .session import Prior, compute_prior, dropout_session, random_session, session_vector

SR = 48000
CLIP_SEC = 3.0
SPLITS = ("train", "val", "pool", "test")
SPLIT_RATIOS = (0.7, 0.1, 0.1, 0.1)
# DI 기준 음량. ④·⑥·⑦ 모두 DI 클립을 이 RMS로 맞춰 G에 넣는다(입력 게인 θ의 기준점).
DI_REF_RMS = 0.05


def normalize_di(x: np.ndarray, target_rms: float = DI_REF_RMS) -> np.ndarray:
    """(…, T) DI를 클립별 RMS target_rms로 맞춘다."""
    r = np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True)) + 1e-8
    return (x / r * target_rms).astype(np.float32)


# ---------------------------------------------------------------- EGDB 분할


def _natural_key(p: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)]


def list_di_files(di_dir: str | Path) -> list[Path]:
    files = sorted(Path(di_dir).rglob("*.wav"), key=_natural_key)
    if not files:
        raise FileNotFoundError(f"DI wav가 없다: {di_dir}")
    return files


def make_split(di_dir: str | Path, seed: int = 0, ratios=SPLIT_RATIOS,
               save_to: str | Path | None = None) -> dict[str, list[str]]:
    """곡(파일) 단위로 train/val/pool/test를 겹치지 않게 나눈다. 같은 seed면 항상 같다."""
    files = list_di_files(di_dir)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(files))
    n = len(files)
    cuts = np.cumsum([int(round(r * n)) for r in ratios[:-1]])
    parts = np.split(idx, cuts)
    root = Path(di_dir)
    split = {name: sorted([str(files[i].relative_to(root)) for i in part], key=lambda s: _natural_key(Path(s)))
             for name, part in zip(SPLITS, parts)}
    if save_to:
        Path(save_to).write_text(json.dumps(split, indent=1), encoding="utf-8")
    return split


def load_split(di_dir: str | Path, split_file: str | Path | None = None, seed: int = 0):
    if split_file and Path(split_file).exists():
        return json.loads(Path(split_file).read_text(encoding="utf-8"))
    return make_split(di_dir, seed, save_to=split_file)


# ---------------------------------------------------------------- DI 클립


class DIReader:
    """DI 파일에서 무작위 3초 클립을 48kHz로 잘라 온다(무음 클립은 다시 뽑는다)."""

    def __init__(self, files: list[str | Path], clip_sec: float = CLIP_SEC, sr: int = SR,
                 min_rms_db: float = -45.0, preload: bool = True):
        import soundfile as sf

        self.files = [Path(f) for f in files]
        self.clip = int(clip_sec * sr)
        self.sr = sr
        self.min_rms = 10 ** (min_rms_db / 20)
        self.infos = [sf.info(str(f)) for f in self.files]
        self.cache: dict[int, np.ndarray] = {}
        if preload:
            for i in range(len(self.files)):
                self._load(i)

    def _load(self, i: int) -> np.ndarray:
        if i not in self.cache:
            from .separate import load_audio

            self.cache[i] = load_audio(self.files[i], self.sr, mono=True)
        return self.cache[i]

    def full(self, i: int) -> np.ndarray:
        return self._load(i)

    def random_clip(self, rng: np.random.Generator, tries: int = 10) -> np.ndarray:
        for _ in range(tries):
            x = self._load(int(rng.integers(len(self.files))))
            if len(x) <= self.clip:
                c = np.pad(x, (0, self.clip - len(x)))
            else:
                s = int(rng.integers(0, len(x) - self.clip))
                c = x[s:s + self.clip]
            if np.sqrt(np.mean(c ** 2)) > self.min_rms:
                return c.astype(np.float32)
        return c.astype(np.float32)

    def batch(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.stack([self.random_clip(rng) for _ in range(n)])

    def fixed_clips(self, n: int, seed: int = 0) -> np.ndarray:
        """⑥ DI 풀처럼 재현 가능한 클립 n개."""
        return self.batch(n, np.random.default_rng(seed))


def di_reader_for_split(di_dir, split_name: str, split_file=None, **kw) -> DIReader:
    split = load_split(di_dir, split_file)
    return DIReader([Path(di_dir) / f for f in split[split_name]], **kw)


# ---------------------------------------------------------------- 증강


def augment_mix(y: np.ndarray, rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    """믹스 처리 흉내: 무작위 EQ, 컴프, 짧은 리버브 (pedalboard). 없으면 원본 그대로."""
    try:
        import pedalboard as pb
    except ImportError:
        return y
    fx = [
        pb.LowShelfFilter(cutoff_frequency_hz=float(rng.uniform(80, 250)), gain_db=float(rng.uniform(-4, 4))),
        pb.PeakFilter(cutoff_frequency_hz=float(np.exp(rng.uniform(np.log(300), np.log(5000)))),
                      gain_db=float(rng.uniform(-4, 4)), q=float(rng.uniform(0.5, 2.0))),
        pb.HighShelfFilter(cutoff_frequency_hz=float(rng.uniform(3000, 9000)), gain_db=float(rng.uniform(-4, 4))),
    ]
    if rng.random() < 0.7:
        fx.append(pb.Compressor(threshold_db=float(rng.uniform(-24, -6)), ratio=float(rng.uniform(1.5, 4)),
                                attack_ms=float(rng.uniform(2, 20)), release_ms=float(rng.uniform(50, 200))))
    if rng.random() < 0.5:
        fx.append(pb.Reverb(room_size=float(rng.uniform(0.05, 0.25)), wet_level=float(rng.uniform(0.03, 0.15)),
                            dry_level=1.0, width=0.0))
    board = pb.Pedalboard(fx)
    peak = np.abs(y).max() + 1e-9
    out = board(y.astype(np.float32) / peak, sr) * peak
    return out.astype(np.float32)


# ---------------------------------------------------------------- 생성기


@dataclass
class SynthConfig:
    batch_size: int = 16
    p_full_range: float = 0.2
    p_unknown: float = 0.3
    p_lower_conf: float = 0.3
    di_gain_range_db: tuple[float, float] = (-12.0, 6.0)  # DI 레벨 증강 범위 (θ의 in_gain_db와 별개)
    augment: bool = True
    target_rms: float = 0.1


class SynthGenerator:
    """배치 단위로 (DI x, 깨끗한 렌더 y, 증강된 y_aug, θ, 세션 벡터)를 만든다."""

    def __init__(self, reader: DIReader, device: str | torch.device = "cpu",
                 cfg: SynthConfig | None = None, seed: int = 0, G: GrayboxAmp | None = None):
        self.reader = reader
        self.device = torch.device(device)
        self.cfg = cfg or SynthConfig()
        self.rng = np.random.default_rng(seed)
        self.G = (G or GrayboxAmp()).to(self.device)

    def sample_theta_session(self) -> tuple[np.ndarray, np.ndarray, dict]:
        c = self.cfg
        sess = random_session(self.rng)
        if self.rng.random() < c.p_full_range:
            prior = Prior.full()
            sess_obs = {"confidence": {}}  # 전부 unknown
        else:
            prior = compute_prior(sess)
            sess_obs = dropout_session(sess, self.rng, c.p_unknown, c.p_lower_conf)
        theta = self.rng.uniform(prior.lo, prior.hi)
        return theta, session_vector(sess_obs), sess

    @torch.no_grad()
    def next_batch(self) -> dict[str, torch.Tensor]:
        c = self.cfg
        n = c.batch_size
        thetas, svecs = [], []
        for _ in range(n):
            th, sv, _ = self.sample_theta_session()
            thetas.append(th)
            svecs.append(sv)
        x = self.reader.batch(n, self.rng)
        g = 10 ** (self.rng.uniform(*c.di_gain_range_db, size=(n, 1)) / 20)
        x = (normalize_di(x) * g).astype(np.float32)

        xt = torch.from_numpy(x).to(self.device)
        th = torch.as_tensor(np.stack(thetas), dtype=torch.float32, device=self.device)
        y = self.G(xt, th, mode="fft")
        y = y / rms(y)[:, None] * c.target_rms
        y = torch.nan_to_num(y)

        if c.augment:
            y_np = y.cpu().numpy()
            y_aug = np.stack([augment_mix(v, self.rng) for v in y_np])
            y_aug = torch.from_numpy(y_aug).to(self.device)
            y_aug = y_aug / rms(y_aug)[:, None] * c.target_rms
        else:
            y_aug = y
        return {"x": xt, "y": y, "y_aug": y_aug, "theta": th,
                "session_vec": torch.as_tensor(np.stack(svecs), device=self.device)}
