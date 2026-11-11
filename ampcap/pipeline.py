"""음원 → .nam 전체 실행 (설계 문서 4.11 "전체 실행 스크립트").

    python -m ampcap.pipeline song.wav --session session.json --egdb data/EGDB/audio_DI \
        --nam-input data/nam/v3_0_0.wav [--segments 42:58.5,131.2:140] [--encoder checkpoints/encoder_v1_best.pt]

각 단계 결과는 runs/<날짜_곡명>/에 파일로 남기고, 다음 단계는 그 파일을 읽는다.
--session-only를 주면 (b) 세션만 경로: 사전 범위 중심값을 θ로 쓰고 ⑤⑥을 건너뛴다.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

import numpy as np
import torch

from . import separate, session as sess_mod
from .encoder import BEST_PATH
from .optimize import FitConfig, fit, load_theta, save_result
from .synth_data import DI_REF_RMS, di_reader_for_split, normalize_di

DEFAULT_RUNS = Path("runs")


def make_run_dir(song: str, root: str | Path = DEFAULT_RUNS) -> Path:
    safe = re.sub(r"[^\w\-가-힣]+", "_", song).strip("_") or "song"
    d = Path(root) / f"{dt.date.today().isoformat()}_{safe}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def parse_segments(s: str | None) -> list[list[float]] | None:
    if not s:
        return None
    out = []
    for part in s.split(","):
        a, b = part.split(":")
        out.append([float(a), float(b)])
    return out


def load_di_pool(egdb_di_dir, n_clips: int = 48, split_file=None, seed: int = 0,
                 split: str = "pool") -> np.ndarray:
    reader = di_reader_for_split(egdb_di_dir, split, split_file)
    return normalize_di(reader.fixed_clips(n_clips, seed))


def step_separate(audio_path, run_dir, segments=None, song="", same_rig=True, device=None):
    _, rep = separate.run(audio_path, run_dir, segments, song=song, same_rig=same_rig, device=device)
    print("① 구간:", rep.segments, "유효 연주 길이", rep.active_sec, "초", rep.warnings)
    if not rep.ok:
        raise SystemExit(rep.message)


def step_session(session_path, run_dir) -> tuple[dict, sess_mod.Prior]:
    s = sess_mod.load_session(session_path) if session_path else sess_mod.unknown_session()
    prior = sess_mod.compute_prior(s)
    (Path(run_dir) / "session.json").write_text(json.dumps(s, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    (Path(run_dir) / "prior.json").write_text(json.dumps(prior.to_dict(), indent=2), encoding="utf-8")
    if prior.warnings:
        print("② 경고:", prior.warnings)
    return s, prior


def step_encoder(encoder_path, stem, session, device) -> np.ndarray | None:
    from . import encoder as enc

    model = enc.try_load(encoder_path, device)
    if model is None:
        return None
    return enc.predict_song(model, stem, sess_mod.session_vector(session))


def step_fit(stem, di_pool, prior, theta_hat, run_dir, device, cfg: FitConfig | None = None):
    target = torch.from_numpy(stem).float().to(device)
    pool = torch.from_numpy(di_pool).float().to(device)
    res = fit(target, pool, prior, theta_hat=theta_hat, cfg=cfg)
    save_result(res, run_dir, di_pool=pool.cpu(), target_rms=float(np.sqrt(np.mean(stem ** 2))))
    print("⑥ θ* 손실", float(res.losses[res.candidates[0]]), res.diagnostics["messages"])
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description="음원 → .nam")
    ap.add_argument("audio")
    ap.add_argument("--session", default=None, help="세션 JSON (없으면 전부 unknown)")
    ap.add_argument("--segments", default=None, help="시작:끝,시작:끝 (초). 없으면 추천 구간")
    ap.add_argument("--egdb", required=True, help="EGDB DI 폴더")
    ap.add_argument("--split-file", default="data/egdb_split.json")
    ap.add_argument("--nam-input", default="data/nam/v3_0_0.wav")
    ap.add_argument("--encoder", default=str(BEST_PATH), help="'' 이면 인코더 없이")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--architecture", default="standard")
    ap.add_argument("--session-only", action="store_true", help="(b) 세션만 경로: 사전 범위 중심을 θ로 쓰고 ⑤⑥을 건너뜀")
    ap.add_argument("--skip-distill", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    song = Path(a.audio).stem
    run_dir = Path(a.run_dir) if a.run_dir else make_run_dir(song)
    print("실행 폴더:", run_dir)

    s0 = sess_mod.load_session(a.session) if a.session else sess_mod.unknown_session()
    step_separate(a.audio, run_dir, parse_segments(a.segments), song, s0["same_rig"], device)
    session, prior = step_session(a.session, run_dir)
    stem = separate.load_audio(run_dir / "stem.wav", mono=True)

    if a.session_only:
        theta = torch.tensor(prior.center, dtype=torch.float32)
        (run_dir / "theta_candidates.json").write_text(json.dumps({
            "theta_star": theta.tolist(), "mode": "session_only",
            "note": "음원 분석 없이 세션 정보만으로 추정"}, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        di_pool = load_di_pool(a.egdb, split_file=a.split_file)
        theta_hat = step_encoder(a.encoder, stem, session, device)
        step_fit(stem, di_pool, prior, theta_hat, run_dir, device, FitConfig(steps=a.steps))
        theta = load_theta(run_dir / "theta_candidates.json")

    if a.skip_distill:
        return
    from .distill import distill

    reader = di_reader_for_split(a.egdb, "test", a.split_file)
    verify_di = list(normalize_di(reader.fixed_clips(4, seed=123)))
    out = distill(theta, run_dir, a.nam_input, DI_REF_RMS, session=session,
                  name=f"{session.get('artist') or ''} {session.get('song') or song}".strip(),
                  epochs=a.epochs, architecture=a.architecture, verify_di=verify_di, device=device)
    print("⑦", json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
