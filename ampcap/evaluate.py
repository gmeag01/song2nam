"""⑨ 평가 지표와 실험.

정답이 있는 비교: ESR(주 지표), MR-STFT, Mel L1, θ 오차(실제 단위)
정답이 없는 비교: ⑥ 통계 거리, 장기 평균 스펙트럼 대역별 dB 차이 그래프, FAD(mel 특징, 참고용),
                 청취 평가(MUSHRA 0~100) 집계

실험 1(합성 되찾기): 앰프 계열 5 × 정답 θ 4 = 가짜 곡 20개를 EGDB test DI로 만들고 믹스 증강을 건다.
조건별로 θ*를 찾고, 가짜 곡에 쓰지 않은 test DI를 정답 G와 결과(G(θ*) 또는 .nam)에 넣어 비교한다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .graybox.amp import NAMES, PARAMS, GrayboxAmp, denormalize
from .stats import STAT_KEYS, Stats, stat_losses

# ---------------------------------------------------------------- 지표


def esr(pred, target) -> float:
    p = torch.as_tensor(pred, dtype=torch.float64).reshape(-1)
    t = torch.as_tensor(target, dtype=torch.float64).reshape(-1)
    n = min(len(p), len(t))
    return float((p[:n] - t[:n]).square().sum() / t[:n].square().sum().clamp_min(1e-12))


def mrstft_distance(pred, target) -> float:
    from .encoder import mrstft

    p = torch.as_tensor(pred, dtype=torch.float32).reshape(1, -1)
    t = torch.as_tensor(target, dtype=torch.float32).reshape(1, -1)
    n = min(p.shape[-1], t.shape[-1])
    return float(mrstft(p[:, :n], t[:, :n]))


def mel_l1(pred, target, stats: Stats | None = None) -> float:
    """프레임별 log-Mel(dB) L1 평균."""
    stats = stats or Stats()
    p = torch.as_tensor(pred, dtype=torch.float32).reshape(1, -1)
    t = torch.as_tensor(target, dtype=torch.float32).reshape(1, -1)
    n = min(p.shape[-1], t.shape[-1])
    mp = stats.frames(p[:, :n])[0]
    mt = stats.frames(t[:, :n])[0]
    return float((mp - mt).abs().mean())


def theta_error_units(theta_pred, theta_true) -> dict[str, float]:
    """θ 오차를 실제 단위로(주파수는 옥타브, 나머지는 해당 단위)."""
    up = denormalize(torch.as_tensor(theta_pred, dtype=torch.float64))
    ut = denormalize(torch.as_tensor(theta_true, dtype=torch.float64))
    out = {}
    for name, scale, _, _ in PARAMS:
        if scale == "log":
            out[name + "(oct)"] = float(torch.log2(up[name] / ut[name]).abs())
        else:
            out[name] = float((up[name] - ut[name]).abs())
    return out


def stat_distance(pred_audio, target_audio, stats: Stats | None = None) -> dict[str, float]:
    stats = stats or Stats()
    with torch.no_grad():
        a = stats(torch.as_tensor(pred_audio, dtype=torch.float32))
        b = stats(torch.as_tensor(target_audio, dtype=torch.float32))
        return {k: float(v.mean()) for k, v in stat_losses(a, b).items()}


def fad_mel(pred_audio, target_audio, stats: Stats | None = None) -> float:
    """log-Mel 프레임 특징의 프레셰 거리(참고용 FAD 대체; VGGish 등 임베딩 아님)."""
    from scipy import linalg

    stats = stats or Stats()
    with torch.no_grad():
        fp = stats.frames(torch.as_tensor(pred_audio, dtype=torch.float32).reshape(1, -1))[0][0].T.numpy()
        ft = stats.frames(torch.as_tensor(target_audio, dtype=torch.float32).reshape(1, -1))[0][0].T.numpy()
    fp = fp - fp.mean(1, keepdims=True)  # 프레임별 음량 제거
    ft = ft - ft.mean(1, keepdims=True)
    mu1, mu2 = fp.mean(0), ft.mean(0)
    s1, s2 = np.cov(fp, rowvar=False), np.cov(ft, rowvar=False)
    covmean = linalg.sqrtm(s1 @ s2)  # scipy 1.18에서 disp 인자 삭제됨
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def mushra_summary(scores: dict[str, list[float]]) -> dict[str, dict]:
    """{조건: [청취자 점수 0~100]} → 평균, 95% 신뢰구간. 인원이 적으면 참고 수치."""
    out = {}
    for k, v in scores.items():
        a = np.asarray(v, dtype=float)
        ci = 1.96 * a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else float("nan")
        out[k] = {"mean": float(a.mean()), "ci95": float(ci), "n": int(len(a))}
    return out


def plot_band_diff(pred_audio, target_audio, path, title: str = "", stats: Stats | None = None):
    """장기 평균 스펙트럼 차이(예측 − 타깃, dB)를 mel 주파수 축으로 그린다."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stats = stats or Stats()
    with torch.no_grad():
        a = stats(torch.as_tensor(pred_audio, dtype=torch.float32))["ltas"].reshape(-1).numpy()
        b = stats(torch.as_tensor(target_audio, dtype=torch.float32))["ltas"].reshape(-1).numpy()
    c = stats.cfg
    m = np.linspace(2595 * np.log10(1 + c.f_min / 700), 2595 * np.log10(1 + c.f_max / 700), c.n_mels + 2)[1:-1]
    hz = 700 * (10 ** (m / 2595) - 1)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.semilogx(hz, a - b)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Hz")
    ax.set_ylabel("dB (pred − target)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def compare_to_truth(theta_true, y_pred_fn, di_clips: np.ndarray, G: GrayboxAmp | None = None,
                     device: str = "cpu") -> dict:
    """정답 G(θ_true)와 결과(y_pred_fn(di) → 오디오)를 미사용 DI로 비교. 음량은 RMS로 맞춰 비교."""
    G = (G or GrayboxAmp()).to(device)
    th = torch.as_tensor(theta_true, dtype=torch.float32, device=device).reshape(1, -1)
    rows = []
    for di in di_clips:
        x = torch.as_tensor(di, dtype=torch.float32, device=device)[None]
        yt = G.render(x, th)[0].cpu().numpy()
        yp = np.asarray(y_pred_fn(di), dtype=np.float32)
        n = min(len(yt), len(yp))
        yt, yp = yt[:n], yp[:n]
        yp = yp / (np.sqrt(np.mean(yp ** 2)) + 1e-9) * np.sqrt(np.mean(yt ** 2))
        rows.append({"esr": esr(yp, yt), "mrstft": mrstft_distance(yp, yt), "mel_l1": mel_l1(yp, yt)})
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


# ---------------------------------------------------------------- 실험 1


FAMILIES5 = ["fender", "vox", "british", "american_hg", "modern_hg"]


@dataclass
class Exp1Config:
    n_theta_per_family: int = 4
    song_clips: int = 8          # 가짜 곡 하나에 쓰는 test DI 3초 클립 수(≈ 24초)
    eval_clips: int = 4          # 비교용 미사용 test DI 클립 수
    pool_clips: int = 48
    steps: int = 600
    starts: list = field(default_factory=lambda: ["center", "multi", "encoder"])
    stat_sets: dict = field(default_factory=lambda: {
        "ltas": ("ltas",), "ltas+level": ("ltas", "level"), "all4": STAT_KEYS})
    sessions: list = field(default_factory=lambda: ["none", "correct", "wrong"])
    session_only_baseline: bool = True
    seed: int = 0


def make_fake_songs(test_reader, cfg: Exp1Config, G: GrayboxAmp, device: str = "cpu"):
    """가짜 곡 20개: (세션, θ_true, 스템 오디오, 비교용 DI)."""
    from .session import _set, compute_prior, random_session
    from .synth_data import augment_mix, normalize_di

    rng = np.random.default_rng(cfg.seed)
    songs = []
    for fam in FAMILIES5:
        for k in range(cfg.n_theta_per_family):
            s = random_session(rng)
            _set(s, "amp.family", fam)
            prior = compute_prior(s)
            theta = rng.uniform(prior.lo, prior.hi)
            clips = normalize_di(test_reader.batch(cfg.song_clips + cfg.eval_clips, rng))
            song_di, eval_di = clips[: cfg.song_clips], clips[cfg.song_clips:]
            x = torch.from_numpy(song_di.reshape(1, -1)).to(device)
            y = G.render(x, torch.as_tensor(theta, dtype=torch.float32, device=device)[None])[0].cpu().numpy()
            y = augment_mix(y / (np.abs(y).max() + 1e-9) * 0.5, rng)
            songs.append({"id": f"{fam}_{k}", "session": s, "theta": theta, "stem": y.astype(np.float32),
                          "eval_di": eval_di})
    return songs


def wrong_session(s: dict) -> dict:
    from .session import _set, validate

    s = validate(s)
    fam = s["amp"]["family"]
    _set(s, "amp.family", FAMILIES5[(FAMILIES5.index(fam) + 2) % len(FAMILIES5)])
    gain = s["amp"]["gain"]
    if gain != "unknown":
        _set(s, "amp.gain", {"low": "high", "mid": "low", "high": "low"}[gain])
    return s


def run_experiment1(test_reader, pool_reader, out_dir: str | Path, cfg: Exp1Config | None = None,
                    encoder_model=None, device: str = "cuda", only_songs: int | None = None,
                    log=print) -> list[dict]:
    """실험 1. 결과는 out_dir/exp1_results.jsonl에 한 줄씩 쌓는다(중단 후 이어서 실행 가능)."""
    from .optimize import FitConfig, fit, make_starts
    from .session import compute_prior, session_vector, unknown_session
    from .synth_data import normalize_di

    cfg = cfg or Exp1Config()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "exp1_results.jsonl"
    done = set()
    if res_path.exists():
        for line in res_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["song"], r["start"], r["stats"], r["session"]))

    G = GrayboxAmp().to(device)
    songs = make_fake_songs(test_reader, cfg, G, device)
    if only_songs:
        songs = songs[:only_songs]
    pool = torch.from_numpy(normalize_di(pool_reader.fixed_clips(cfg.pool_clips, cfg.seed))).to(device)
    rng = np.random.default_rng(cfg.seed)
    results = []

    def record(r):
        results.append(r)
        with res_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    for song in songs:
        target = torch.from_numpy(song["stem"]).to(device)
        for sess_name in cfg.sessions:
            sess = {"none": unknown_session(), "correct": song["session"],
                    "wrong": wrong_session(song["session"])}[sess_name]
            prior = compute_prior(sess)

            if cfg.session_only_baseline and (song["id"], "session_only", "-", sess_name) not in done:
                m = _evaluate_theta(prior.center, song, G, device)
                record({"song": song["id"], "start": "session_only", "stats": "-", "session": sess_name, **m})

            theta_hat = None
            if encoder_model is not None:
                from .encoder import predict_song

                theta_hat = predict_song(encoder_model, song["stem"], session_vector(sess))

            for stat_name, keys in cfg.stat_sets.items():
                for start in cfg.starts:
                    key = (song["id"], start, stat_name, sess_name)
                    if key in done:
                        continue
                    if start == "encoder" and theta_hat is None:
                        continue
                    if start == "center":
                        starts, names = prior.center[None], ["prior_center"]
                    elif start == "multi":
                        starts, names = make_starts(prior, None, 4, rng)
                    else:
                        starts, names = np.asarray(theta_hat)[None], ["encoder"]
                    fc = FitConfig(steps=cfg.steps, seed=cfg.seed)
                    res = fit(target, pool, prior, theta_hat=theta_hat if start == "encoder" else None,
                              cfg=fc, starts=starts, start_names=names, stat_keys=keys)
                    m = _evaluate_theta(res.theta_best.numpy(), song, G, device)
                    r = {"song": song["id"], "start": start, "stats": stat_name, "session": sess_name,
                         "fit_loss": float(res.losses.min()), **m}
                    record(r)
                    log(json.dumps({k: r[k] for k in ("song", "start", "stats", "session", "esr")}))
    return results


def _evaluate_theta(theta, song, G, device) -> dict:
    th = torch.as_tensor(theta, dtype=torch.float32, device=device).reshape(1, -1)

    def pred(di):
        x = torch.as_tensor(di, dtype=torch.float32, device=device)[None]
        return G.render(x, th)[0].cpu().numpy()

    m = compare_to_truth(song["theta"], pred, song["eval_di"], G, device)
    err = theta_error_units(np.asarray(theta), song["theta"])
    m["theta_rmse_norm"] = float(np.sqrt(np.mean((np.asarray(theta) - song["theta"]) ** 2)))
    m["theta_err"] = err
    return m


def summarize_exp1(results_path: str | Path) -> "object":
    """조건별 평균 표(pandas DataFrame)."""
    import pandas as pd

    rows = [json.loads(l) for l in Path(results_path).read_text(encoding="utf-8").splitlines()]
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "theta_err"} for r in rows])
    return df.groupby(["session", "stats", "start"])[["esr", "mrstft", "mel_l1", "theta_rmse_norm"]] \
        .agg(["mean", "median"]).round(4)


__all__ = ["esr", "mrstft_distance", "mel_l1", "theta_error_units", "stat_distance", "fad_mel",
           "mushra_summary", "plot_band_diff", "compare_to_truth", "Exp1Config", "run_experiment1",
           "summarize_exp1", "NAMES"]
