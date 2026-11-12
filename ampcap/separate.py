"""① 분리, 구간 점검, 추천 구간.

1. Demucs `htdemucs_6s`로 스테레오 기타 스템을 얻는다.
2. 짧은 창마다 좌우 상관도와 Side/Mid 에너지 비로 단일 기타 가능성을 매겨 추천 구간을 낸다(참고용).
3. 사용자가 고른 구간을 점검(유효 연주 길이, 무음 제거, 클리핑, 여러 기타 혼입)하고 크로스페이드로 이어 붙여
   48kHz 모노 `stem.wav`와 `segments.json`으로 저장한다.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

SR = 48000
MIN_ACTIVE_SEC = 15.0
SILENCE_DB = -50.0          # 스템 최대 프레임 대비
FRAME = 1024
CROSSFADE_SEC = 0.01
CLIP_LEVEL = 0.999

GUIDANCE = {
    "rhythm": "리듬 기타 캡처: 코드 백킹이 길게 나오고 리드·솔로가 겹치지 않는 구간을 고르세요. "
              "더블 트래킹이면 한쪽만 들리는 인트로·브레이크 구간이 좋습니다.",
    "lead": "리드 기타 캡처: 솔로·멜로디가 나오고 리듬 기타가 작거나 없는 구간을 고르세요. "
            "딜레이·리버브가 적은 부분이 좋습니다.",
}


# ---------------------------------------------------------------- 입출력


def load_audio(path: str | Path, sr: int = SR, mono: bool = False) -> np.ndarray:
    """파일을 읽어 sr로 변환한다. 반환 (C, T) float32 (mono면 (T,))."""
    import soundfile as sf

    try:
        x, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
        x = x.T
    except Exception:
        # mp3 등 soundfile이 못 읽는 형식은 librosa(audioread/ffmpeg)로
        import librosa

        x, file_sr = librosa.load(str(path), sr=None, mono=False)
        x = np.atleast_2d(x).astype(np.float32)
    if file_sr != sr:
        x = resample(x, file_sr, sr)
    if mono:
        return x.mean(0)
    return x


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(int(sr_in), int(sr_out))
    return resample_poly(x, sr_out // g, sr_in // g, axis=-1).astype(np.float32)


def save_wav(path: str | Path, x: np.ndarray, sr: int = SR, subtype: str = "FLOAT"):
    import soundfile as sf

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x, dtype=np.float32)
    sf.write(str(path), x.T if x.ndim == 2 else x, sr, subtype=subtype)


# ---------------------------------------------------------------- Demucs


def separate_guitar(audio: np.ndarray, sr: int = SR, device: str | None = None,
                    model_name: str = "htdemucs_6s") -> np.ndarray:
    """스테레오 음원 (2, T) → 스테레오 기타 스템 (2, T), 같은 sr.

    GPU 메모리가 부족하면 세그먼트를 줄여 다시 시도하고, 그래도 안 되면 CPU로 돌린다.
    """
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    if audio.ndim == 1:
        audio = np.stack([audio, audio])
    if audio.shape[0] == 1:
        audio = np.concatenate([audio, audio])
    model = get_model(model_name)
    model.eval()
    msr = model.samplerate
    wav = torch.from_numpy(resample(audio, sr, msr) if sr != msr else audio)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    gi = model.sources.index("guitar")
    attempts = [(device, None), (device, 4.0), ("cpu", None)] if device != "cpu" else [("cpu", None)]
    last_err = None
    for dev, seg in attempts:
        try:
            kw = {"segment": seg} if seg else {}
            with torch.no_grad():
                out = apply_model(model, wav[None], device=dev, split=True, overlap=0.25,
                                  progress=False, **kw)[0]
            break
        except RuntimeError as e:  # CUDA OOM 등
            last_err = e
            if "out of memory" not in str(e).lower() and dev == "cpu":
                raise
            torch.cuda.empty_cache()
    else:
        raise RuntimeError(f"Demucs 분리 실패: {last_err}")

    stem = (out[gi] * (ref.std() + 1e-8) + ref.mean()).cpu().numpy()
    if msr != sr:
        stem = resample(stem, msr, sr)
    n = audio.shape[-1]
    if stem.shape[-1] < n:
        stem = np.pad(stem, ((0, 0), (0, n - stem.shape[-1])))
    return stem[:, :n].astype(np.float32)


# ---------------------------------------------------------------- 추천 구간


def stereo_analysis(stem: np.ndarray, sr: int = SR, win_sec: float = 1.0, hop_sec: float = 0.5):
    """창마다 (시작 초, 좌우 상관도, Side/Mid 에너지 비, 에너지 dB, 단일 기타 점수)."""
    if stem.ndim == 1 or stem.shape[0] == 1:
        stem = np.stack([np.ravel(stem)] * 2)
    L, R = stem[0], stem[1]
    win, hop = int(win_sec * sr), int(hop_sec * sr)
    rows = []
    for s in range(0, max(1, len(L) - win + 1), hop):
        l, r = L[s:s + win], R[s:s + win]
        el, er = float(np.sum(l * l)), float(np.sum(r * r))
        corr = float(np.sum(l * r) / (np.sqrt(el * er) + 1e-12))
        mid, side = (l + r) / 2, (l - r) / 2
        sm = float(np.sum(side ** 2) / (np.sum(mid ** 2) + 1e-12))
        energy_db = 10 * np.log10((el + er) / (2 * win) + 1e-12)
        # 한쪽으로만 패닝된 기타(좌우 에너지 불균형)도 한 대로 본다.
        imbalance = abs(el - er) / (el + er + 1e-12)
        score = max(corr, imbalance) * (1 - min(sm, 1.0) * 0.5)
        rows.append((s / sr, corr, sm, energy_db, float(np.clip(score, 0, 1))))
    return rows


def recommend_segments(stem: np.ndarray, sr: int = SR, threshold: float = 0.7,
                       min_len_sec: float = 3.0) -> list[list[float]]:
    """단일 기타 가능성이 높은 구간 [[시작, 끝], ...]. 판정은 참고용이다."""
    rows = stereo_analysis(stem, sr)
    if not rows:
        return []
    max_db = max(r[3] for r in rows)
    hop = rows[1][0] - rows[0][0] if len(rows) > 1 else 0.5
    segs, cur = [], None
    for t, _, _, edb, score in rows:
        ok = score >= threshold and edb > max_db + SILENCE_DB + 10
        if ok and cur is None:
            cur = [t, t + 2 * hop]
        elif ok:
            cur[1] = t + 2 * hop
        elif cur is not None:
            segs.append(cur)
            cur = None
    if cur is not None:
        segs.append(cur)
    return [[round(a, 2), round(b, 2)] for a, b in segs if b - a >= min_len_sec]


# ---------------------------------------------------------------- 점검과 이어 붙이기


def frame_db(x: np.ndarray, frame: int = FRAME) -> np.ndarray:
    n = len(x) // frame
    if n == 0:
        return np.array([10 * np.log10(np.mean(x ** 2) + 1e-12)])
    f = x[: n * frame].reshape(n, frame)
    return 10 * np.log10(np.mean(f ** 2, axis=1) + 1e-12)


def active_mask(x: np.ndarray, ref_db: float | None = None, frame: int = FRAME,
                silence_db: float = SILENCE_DB) -> np.ndarray:
    """프레임별 연주 여부. ref_db(기본: 신호 최대 프레임) 대비 silence_db보다 크면 연주."""
    db = frame_db(x, frame)
    ref = db.max() if ref_db is None else ref_db
    return db > ref + silence_db


def crossfade_concat(chunks: list[np.ndarray], sr: int = SR,
                     fade_sec: float = CROSSFADE_SEC, edge_fade: bool = True) -> np.ndarray:
    """조각을 등전력 크로스페이드로 이어 붙인다(크로스페이드 접합부 클릭 방지)."""
    chunks = [c for c in chunks if len(c) > 0]
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    n = int(fade_sec * sr)
    out = chunks[0].astype(np.float32).copy()
    for c in chunks[1:]:
        c = c.astype(np.float32)
        k = min(n, len(out), len(c))
        if k == 0:
            out = np.concatenate([out, c])
            continue
        t = np.linspace(0, np.pi / 2, k, dtype=np.float32)
        mixed = out[-k:] * np.cos(t) + c[:k] * np.sin(t)
        out = np.concatenate([out[:-k], mixed, c[k:]])
    # 맨 앞·뒤도 짧게 페이드
    k = min(n, len(out) // 2)
    if edge_fade and k > 0:
        ramp = np.sin(np.linspace(0, np.pi / 2, k, dtype=np.float32))
        out[:k] *= ramp
        out[-k:] *= ramp[::-1]
    return out


def remove_silence(x: np.ndarray, sr: int = SR, ref_db: float | None = None,
                   frame: int = FRAME, min_gap_frames: int = 4) -> np.ndarray:
    """무음 프레임(연속 min_gap_frames 이상)을 빼고 크로스페이드로 잇는다."""
    m = active_mask(x, ref_db, frame)
    # 짧은 무음은 남긴다(음 사이 쉼).
    keep = m.copy()
    i = 0
    while i < len(m):
        if not m[i]:
            j = i
            while j < len(m) and not m[j]:
                j += 1
            if j - i < min_gap_frames:
                keep[i:j] = True
            i = j
        else:
            i += 1
    chunks, start = [], None
    for i, k in enumerate(keep):
        if k and start is None:
            start = i
        elif not k and start is not None:
            chunks.append(x[start * frame:i * frame])
            start = None
    if start is not None:
        chunks.append(x[start * frame:])
    return crossfade_concat(chunks, sr, edge_fade=False)


@dataclass
class SegmentReport:
    song: str
    segments: list[list[float]]
    active_sec: float
    warnings: list[str] = field(default_factory=list)
    ok: bool = True
    message: str = ""


def check_and_build(stem: np.ndarray, segments: list[list[float]], sr: int = SR, song: str = "",
                    same_rig: bool = True, min_active_sec: float = MIN_ACTIVE_SEC,
                    stereo: np.ndarray | None = None) -> tuple[np.ndarray, SegmentReport]:
    """구간을 잘라 점검하고 이어 붙인다.

    stem: (T,) 모노 또는 (2, T) 스테레오. stereo를 주면 여러 기타 혼입 점검에 쓴다.
    segments가 비면 전체 스템을 쓰고 "여러 기타가 섞인 평균 톤" 경고를 단다.
    """
    if stem.ndim == 2:
        stereo = stem if stereo is None else stereo
        mono = stem.mean(0)
    else:
        mono = stem
    warnings: list[str] = []
    if not segments:
        segments = [[0.0, len(mono) / sr]]
        warnings.append("full_stem_used:여러 기타가 섞인 평균 톤")

    ref_db = frame_db(mono).max()
    pieces, raw_pieces = [], []
    for a, b in segments:
        if b <= a:
            raise ValueError(f"구간 끝이 시작보다 앞섬: {[a, b]}")
        s, e = int(a * sr), min(int(b * sr), len(mono))
        pieces.append(remove_silence(mono[s:e], sr, ref_db=ref_db))
        raw_pieces.append(stereo[:, s:e] if stereo is not None else mono[s:e])
    out = crossfade_concat(pieces, sr)
    active_sec = len(out) / sr

    raw = np.concatenate(raw_pieces, axis=-1)
    if np.mean(np.abs(raw) >= CLIP_LEVEL) > 1e-4:  # 고른 구간, 채널별
        warnings.append("clipping")
    if stereo is not None:
        rows = stereo_analysis(raw, sr)
        if rows and np.mean([r[4] for r in rows]) < 0.5:
            warnings.append("multi_guitar_suspected" + (":same_rig" if same_rig else ""))

    rep = SegmentReport(song=song, segments=[[float(a), float(b)] for a, b in segments],
                        active_sec=round(active_sec, 2), warnings=warnings)
    if active_sec < min_active_sec:
        rep.ok = False
        rep.message = (f"무음을 뺀 연주 길이가 {active_sec:.1f}초로 기준 {min_active_sec:.0f}초보다 짧습니다. "
                       "구간을 더 추가하세요.")
    return out, rep


def multi_guitar_message(warnings: list[str]) -> str | None:
    for w in warnings:
        if w.startswith("multi_guitar_suspected"):
            if w.endswith(":same_rig"):
                return "참고: 기타가 여러 대일 수 있지만 같은 장비라고 하셨으니 진행해도 괜찮습니다."
            return "경고: 기타가 여러 대 섞인 것 같습니다. 한 대만 나오는 구간을 고르면 정확해집니다."
    return None


def save_outputs(run_dir: str | Path, stem_mono: np.ndarray, report: SegmentReport, sr: int = SR):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_wav(run_dir / "stem.wav", stem_mono, sr)
    (run_dir / "segments.json").write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")


def run(audio_path: str | Path, run_dir: str | Path, segments: list[list[float]] | None = None,
        song: str = "", same_rig: bool = True, device: str | None = None):
    """음원 파일 → stem_full.wav(스테레오), stem.wav, segments.json. segments가 None이면 추천 구간 사용.

    Demucs 결과(stem_full.wav)가 이미 있으면 다시 분리하지 않는다. 구간 처리는 매번 새로 한다.
    """
    run_dir = Path(run_dir)
    full = run_dir / "stem_full.wav"
    if full.exists():
        stem = load_audio(full, SR)
    else:
        stem = separate_guitar(load_audio(audio_path, SR), SR, device=device)
        save_wav(full, stem, SR)
    if segments is None:
        segments = recommend_segments(stem, SR)
    mono, rep = check_and_build(stem, segments, SR, song=song or Path(audio_path).stem,
                                same_rig=same_rig)
    save_outputs(run_dir, mono, rep, SR)
    return mono, rep
