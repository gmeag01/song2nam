"""⑥ 곡별 파라미터 최적화(inference-time optimization), 후보 선택.

네트워크 가중치는 바꾸지 않고 θ만 곡마다 최적화한다.

L(θ) = Σ wₖ · L1(통계ₖ(G(DI; θ)), 통계ₖ(스템)) / (시작점 손실)   ← 최적화 중
       최종 θ* 선택은 모든 시작점이 같은 기준(고정 평가 클립에서 시작점 평균 손실)으로 나눈 값으로 한다
     + λ_prior · (사전 범위를 벗어난 거리)²
     + λ_enc · ‖θ − θ̂‖²
     + λ_anchor · Σ_드라이브 ((θ − θ₀) / σ)²        ← 드라이브 앵커(θ₀ = 그 시작점)

θ = sigmoid(u), u를 Adam으로 최적화한다. 시작점 여러 개를 한 배치로 병렬 최적화한다.
교환 게인(in_gain, boost_level)은 시작점 값에 고정한다(설계 4.7 게인 식별). θ*는 손실이 최소의
(1 + select_eps) 안에 드는 시작점 중 시작점에서 가장 덜 움직인 것이다(최소 이동 선택).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .graybox.amp import INDEX, NAMES, GrayboxAmp, theta_to_units_dict
from .session import Prior
from .stats import STAT_KEYS, Stats, band_diff_db, describe_residual, stat_losses


@dataclass
class FitConfig:
    steps: int = 800
    lr: float = 0.05
    clips_per_step: int = 8
    n_prior_samples: int = 4
    weights: dict = field(default_factory=lambda: {k: 1.0 for k in STAT_KEYS})
    lambda_prior: float = 0.1
    lambda_enc: float = 0.05
    patience: int = 150          # EMA 손실이 이만큼 좋아지지 않으면 종료
    ema: float = 0.9
    min_delta: float = 1e-3
    eval_clips: int = 16         # 마지막 평가에 쓰는 고정 DI 클립 수
    max_restarts: int = 3
    dedupe_dist: float = 0.08    # 후보 중복 제거 θ 거리 (RMS, 0~1 단위)
    n_candidates: int = 4
    seed: int = 0
    freeze_exchangeable: bool = True   # 교환 게인 고정
    lambda_anchor: float = 0.1         # 드라이브 앵커 세기
    # shortcut: 드라이브 4개 공통 σ(정규화 단위, 0.1 ≈ 3~4 dB). 인코더 파라미터별 검증 RMSE를 저장하게 되면 그 값으로 바꾼다
    anchor_sigma: float = 0.1
    select_eps: float = 0.02           # 최소 이동 선택에서 '손실이 같다'로 보는 비율


@dataclass
class FitResult:
    theta_star: torch.Tensor           # (24,) θ* = 후보 0 (최소 이동 선택)
    thetas: torch.Tensor               # (S, 24) 시작점별 최종 θ
    losses: torch.Tensor               # (S,) 고정 클립 평가 손실(시작점 공통 기준으로 정규화)
    start_names: list[str]
    curve: list[list[float]]           # 스텝별 시작점 손실
    candidates: list[int]              # 서로 다른 상위 결과의 인덱스(손실 순)
    diagnostics: dict
    restarts: int = 0


def _logit(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-4, 1 - 1e-4)
    return torch.log(p) - torch.log1p(-p)


# 사이에 선형 블록만 있어 dB 합만 출력에 남는 게인 쌍(앞쪽, 뒤쪽). 뒤쪽이 자유로울 때만 앞쪽을 고정한다.
# boost=none이면 boost_drive가 고정이라 in_gain이 첫 tanh의 유일한 드라이브가 되므로 풀어 둔다.
EXCHANGEABLE = [("in_gain_db", "boost_drive_db"), ("boost_level_db", "pre1_gain_db")]
DRIVES = ["boost_drive_db", "pre1_gain_db", "pre2_gain_db", "pa_drive_db"]   # 비선형 4곳의 입력 게인


def exchangeable_mask(fixed: torch.Tensor) -> torch.Tensor:
    m = torch.zeros_like(fixed)
    for a, b in EXCHANGEABLE:
        if not bool(fixed[INDEX[b]]):
            m[INDEX[a]] = True
    return m & ~fixed


def prior_penalty(theta: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    d = torch.relu(lo - theta) + torch.relu(theta - hi)
    return d.square().sum(-1)


def make_starts(prior: Prior, theta_hat: np.ndarray | None, n_samples: int,
                rng: np.random.Generator) -> tuple[np.ndarray, list[str]]:
    starts = [prior.center]
    names = ["prior_center"]
    if n_samples > 0:
        for i, s in enumerate(prior.sample(n_samples, rng)):
            starts.append(s)
            names.append(f"prior_sample_{i}")
    if theta_hat is not None:
        starts.append(np.asarray(theta_hat, dtype=np.float64))
        names.append("encoder")
    return np.stack(starts), names


def _pick(pool: torch.Tensor, n: int, gen: torch.Generator) -> torch.Tensor:
    """DI 풀에서 n개를 중복 없이 고른다. 풀이 n개 이하면 풀 전체(통계는 순서와 무관)."""
    if n >= pool.shape[0]:
        return pool
    idx = torch.randperm(pool.shape[0], generator=gen)[:n]
    return pool[idx.to(pool.device)]


def fit(target_audio: torch.Tensor, di_pool: torch.Tensor, prior: Prior,
        theta_hat: np.ndarray | None = None, cfg: FitConfig | None = None,
        G: GrayboxAmp | None = None, stats: Stats | None = None,
        starts: np.ndarray | None = None, start_names: list[str] | None = None,
        stat_keys=STAT_KEYS, progress=None) -> FitResult:
    """target_audio: 스템 (T,) (무음 제외). di_pool: (P, T_clip) DI 클립.

    starts를 직접 주면 그 시작점들만 쓴다(실험 1의 시작점 비교).
    progress(step, losses)가 주어지면 매 스텝 호출한다(Gradio 실시간 표시).
    """
    cfg = cfg or FitConfig()
    dev = di_pool.device
    G = G or GrayboxAmp()
    stats = (stats or Stats()).to(dev)
    rng = np.random.default_rng(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)

    with torch.no_grad():
        tgt = stats(target_audio.to(dev).float())

    if starts is None:
        starts, start_names = make_starts(prior, theta_hat, cfg.n_prior_samples, rng)
    start_names = start_names or [f"start_{i}" for i in range(len(starts))]
    S = len(starts)

    lo = torch.as_tensor(prior.lo, dtype=torch.float32, device=dev)
    hi = torch.as_tensor(prior.hi, dtype=torch.float32, device=dev)
    fixed = torch.as_tensor(prior.fixed, device=dev)
    fixed_val = lo.clone()
    th_hat = None if theta_hat is None else torch.as_tensor(theta_hat, dtype=torch.float32, device=dev)
    u0 = _logit(torch.as_tensor(starts, dtype=torch.float32, device=dev))
    th0 = torch.where(fixed, fixed_val.expand_as(u0), torch.sigmoid(u0))       # (S, 24) 시작점
    freeze = exchangeable_mask(fixed) if cfg.freeze_exchangeable else torch.zeros_like(fixed)
    drive = torch.tensor([INDEX[n] for n in DRIVES], device=dev)

    def theta_of(u):
        th = torch.where(freeze, th0, torch.sigmoid(u))
        return torch.where(fixed, fixed_val.expand_as(th), th)

    def losses_for(th, clips, norm=None):
        n = clips.shape[0]
        x = clips[None].expand(S, -1, -1).reshape(S * n, -1)
        # ⑦ 렌더링(96kHz)과 같은 앨리어싱 조건에서 맞춘다(설계 4.4 구현 결정 5)
        y = G(x, th.repeat_interleave(n, 0), mode="fft", oversample=True).reshape(S, n, -1)
        st = stats(y)
        ls = stat_losses(st, tgt, stat_keys)
        terms = torch.stack([ls[k] for k in stat_keys], -1)       # (S, K)
        if norm is None:
            norm = terms.detach().clamp_min(1e-6)
        return combine(th, terms, norm), terms, norm, st

    def combine(th, terms, norm):
        w = torch.tensor([cfg.weights.get(k, 1.0) for k in stat_keys], device=dev)
        total = (terms / norm * w).sum(-1)
        total = total + cfg.lambda_prior * prior_penalty(th, lo, hi)
        if th_hat is not None and cfg.lambda_enc > 0:
            total = total + cfg.lambda_enc * (th - th_hat).square().sum(-1)
        if cfg.lambda_anchor > 0:
            total = total + cfg.lambda_anchor * ((th - th0)[:, drive] / cfg.anchor_sigma).square().sum(-1)
        return total

    u = u0.clone().requires_grad_(True)
    lr = cfg.lr
    opt = torch.optim.Adam([u], lr=lr)

    with torch.no_grad():
        _, _, norm, _ = losses_for(theta_of(u), _pick(di_pool, cfg.clips_per_step, gen))

    curve: list[list[float]] = []
    ema = None
    best_ema = torch.full((S,), float("inf"), device=dev)
    best_u = u.detach().clone()
    since_best = 0
    restarts = 0
    step = 0
    while step < cfg.steps:
        clips = _pick(di_pool, cfg.clips_per_step, gen)
        th = theta_of(u)
        total, _, _, _ = losses_for(th, clips, norm)
        if not torch.isfinite(total).all():
            # NaN·발산: 최선 θ로 되돌리고 학습률을 낮춰 재시작
            restarts += 1
            if restarts > cfg.max_restarts:
                break
            with torch.no_grad():
                u.copy_(best_u)
            lr *= 0.5
            opt = torch.optim.Adam([u], lr=lr)
            continue
        opt.zero_grad()
        total.sum().backward()
        torch.nn.utils.clip_grad_norm_([u], 10.0)
        opt.step()

        tl = total.detach()
        ema = tl if ema is None else cfg.ema * ema + (1 - cfg.ema) * tl
        curve.append(tl.cpu().tolist())
        improved = ema < best_ema - cfg.min_delta
        with torch.no_grad():
            best_u[improved] = u.detach()[improved]
            best_ema = torch.where(improved, ema, best_ema)
        since_best = 0 if improved.any() else since_best + 1
        if progress is not None:
            progress(step, tl.cpu().tolist())
        step += 1
        if since_best >= cfg.patience:
            break

    # 고정 클립으로 시작점별 최선 θ를 평가. 최적화 중의 시작점별 정규화를 그대로 쓰면 손실이 많이 준
    # 시작점이 θ*가 되므로(처음부터 좋던 θ̂가 불리), 모든 시작점에 같은 기준(1, K)을 쓴다.
    with torch.no_grad():
        n_eval = min(cfg.eval_clips, di_pool.shape[0])
        eval_clips = di_pool[:n_eval]
        th_final = theta_of(best_u)
        _, terms, _, st = losses_for(th_final, eval_clips, norm)
        finite = torch.where(torch.isfinite(terms), terms, torch.full_like(terms, float("nan")))
        ref = finite.nanmean(0, keepdim=True).nan_to_num(1.0).clamp_min(1e-6)
        total = torch.nan_to_num(combine(th_final, terms, ref), nan=float("inf"))

    # 최소 이동 선택: 손실이 거의 같으면 시작점에서 덜 움직인 쪽(평평한 방향으로 흘러간 θ를 피한다), 같으면 인코더
    move = (th_final - th0).square().mean(-1).sqrt()
    lim = total.min() * (1 + cfg.select_eps)
    near = [i for i in range(S) if total[i] <= lim]
    best = min(near, key=lambda i: (round(float(move[i]), 3), start_names[i] != "encoder"))
    order = [best] + [i for i in torch.argsort(total).tolist() if i != best]
    cands = []
    for i in order:
        if all(((th_final[i] - th_final[j]).square().mean().sqrt() > cfg.dedupe_dist) for j in cands):
            cands.append(i)
        if len(cands) >= cfg.n_candidates:
            break

    bands = band_diff_db(st["ltas"][best], tgt["ltas"][0], stats.cfg)
    diagnostics = {
        "terms": {k: float(terms[best, i]) for i, k in enumerate(stat_keys)},
        "band_diff_db": bands,
        "messages": describe_residual(bands),
        "prior_violation": float(prior_penalty(th_final[best], lo, hi)),
        "steps": step, "lr_final": lr,
        "move": float(move[best]), "frozen": [NAMES[i] for i in torch.nonzero(freeze).flatten().tolist()],
    }
    return FitResult(theta_star=th_final[best].cpu(), thetas=th_final.cpu(), losses=total.cpu(),
                     start_names=start_names, curve=curve, candidates=cands,
                     diagnostics=diagnostics, restarts=restarts)


# ---------------------------------------------------------------- 저장


def save_result(res: FitResult, run_dir: str | Path, di_pool: torch.Tensor | None = None,
                G: GrayboxAmp | None = None, target_rms: float | None = None, n_audio_clips: int = 4):
    """theta_candidates.json, loss_curve.png, candidate_k.wav 저장."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cands = []
    for rank, i in enumerate(res.candidates):
        th = res.thetas[i]
        cands.append({
            "rank": rank, "start": res.start_names[i], "loss": float(res.losses[i]),
            "theta": [round(float(v), 5) for v in th], "units": theta_to_units_dict(th),
        })
    data = {
        "names": NAMES,
        "theta_star": [round(float(v), 5) for v in res.theta_star],
        "theta_star_units": theta_to_units_dict(res.theta_star),
        "candidates": cands,
        "all_starts": [{"start": n, "loss": float(l)} for n, l in zip(res.start_names, res.losses)],
        "diagnostics": res.diagnostics, "restarts": res.restarts,
    }
    (run_dir / "theta_candidates.json").write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
    plot_curve(res, run_dir / "loss_curve.png")

    if di_pool is not None:
        from .separate import save_wav

        G = G or GrayboxAmp()
        x = di_pool[:n_audio_clips].reshape(1, -1).cpu()
        for rank, i in enumerate(res.candidates):
            y = G.render(x, res.thetas[i].reshape(1, -1).float(), target_rms=target_rms)[0]
            y = y / max(1.0, float(y.abs().max()) / 0.99)
            save_wav(run_dir / f"candidate_{rank}.wav", y.numpy())
    return data


def plot_curve(res: FitResult, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = np.array(res.curve) if res.curve else np.zeros((0, len(res.start_names)))
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for j, n in enumerate(res.start_names):
        if c.size:
            ax.plot(c[:, j], label=n, lw=1)
    ax.set_xlabel("step")
    ax.set_ylabel("loss (normalized)")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def load_theta(path: str | Path, rank: int | None = None) -> torch.Tensor:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if rank is None:
        return torch.tensor(d["theta_star"], dtype=torch.float32)
    return torch.tensor(d["candidates"][rank]["theta"], dtype=torch.float32)


def theta_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).square().mean().sqrt())

