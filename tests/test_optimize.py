"""⑥ 최적화: 같은 DI와 정답 θ로 렌더링한 타깃 스템에서 θ 복원 검사.

CPU에서도 돌도록 3개 차원만 풀고(나머지는 정답으로 고정) 짧은 클립을 쓴다.
24차원 전체 복원은 실험 1(Colab GPU)에서 확인한다.
"""

import json

import numpy as np
import pytest
import torch

from ampcap.graybox.amp import INDEX, N_PARAMS, GrayboxAmp, theta_from_units
from ampcap.optimize import FitConfig, fit, load_theta, save_result
from ampcap.session import Prior

SR = 48000


def _di_clips(n=4, sec=1.0, seed=0):
    rng = np.random.default_rng(seed)
    T = int(sec * SR)
    t = np.arange(T) / SR
    clips = []
    for _ in range(n):
        x = np.zeros(T)
        for start in rng.uniform(0, sec - 0.3, size=3):
            f = rng.uniform(90, 250)
            s = int(start * SR)
            tt = t[: T - s]
            note = sum(k ** -1.5 * np.sin(2 * np.pi * k * f * tt) for k in range(1, 30) if k * f < 10000)
            x[s:] += rng.uniform(0.2, 1.0) * np.exp(-3 * tt) * note
        clips.append(x / np.sqrt(np.mean(x ** 2)) * 0.05)
    return torch.tensor(np.stack(clips), dtype=torch.float32)


FREE = ["pre1_gain_db", "ts_treble_db", "ts_bass_db"]


def _prior_fixing_all_but(theta_true, free):
    lo = theta_true.double().numpy().copy()
    hi = lo.copy()
    for n in free:
        lo[INDEX[n]], hi[INDEX[n]] = 0.0, 1.0
    return Prior(lo, hi, (lo + hi) / 2)


@pytest.mark.slow
def test_recovers_theta_from_same_di():
    torch.manual_seed(0)
    G = GrayboxAmp()
    di = _di_clips()
    theta_true = theta_from_units({"pre1_gain_db": 26.0, "ts_treble_db": 6.0, "ts_bass_db": -5.0,
                                   "pre2_gain_db": 6.0, "in_gain_db": 0.0})
    with torch.no_grad():   # 타깃은 ⑦과 같은 렌더링(lfilter + 2배 오버샘플링)
        stem = G.render(di, theta_true.expand(len(di), -1)).reshape(-1)
    prior = _prior_fixing_all_but(theta_true, FREE)

    cfg = FitConfig(steps=400, lr=0.05, clips_per_step=len(di), n_prior_samples=2,
                    lambda_prior=0.0, patience=400, eval_clips=len(di), lambda_anchor=0.0)
    res = fit(stem, di, prior, cfg=cfg, G=G)
    err = {n: abs(float(res.theta_star[INDEX[n]] - theta_true[INDEX[n]])) for n in FREE}
    print("θ 복원 오차(0~1):", err, "손실", float(res.losses.min()))
    assert all(e < 0.08 for e in err.values()), err


def test_fit_outputs_and_fixed_dims(tmp_path):
    di = _di_clips(n=2, sec=0.5)
    theta_true = theta_from_units({"pre1_gain_db": 20.0})
    G = GrayboxAmp()
    with torch.no_grad():
        stem = G(di, theta_true.expand(2, -1)).reshape(-1)
    prior = _prior_fixing_all_but(theta_true, ["pre1_gain_db"])
    res = fit(stem, di, prior, cfg=FitConfig(steps=5, clips_per_step=2, n_prior_samples=2), G=G,
              theta_hat=np.full(N_PARAMS, 0.5))
    assert res.thetas.shape == (4, N_PARAMS)        # 중심 + 샘플 2 + 인코더
    assert res.start_names[-1] == "encoder"
    fixed = [i for i in range(N_PARAMS) if i != INDEX["pre1_gain_db"]]
    assert torch.allclose(res.thetas[:, fixed], theta_true[fixed].expand(4, -1), atol=1e-6)

    data = save_result(res, tmp_path, di_pool=di, G=G, n_audio_clips=2)
    assert (tmp_path / "theta_candidates.json").exists()
    assert (tmp_path / "loss_curve.png").exists()
    assert (tmp_path / "candidate_0.wav").exists()
    d = json.loads((tmp_path / "theta_candidates.json").read_text(encoding="utf-8"))
    assert len(d["theta_star"]) == N_PARAMS and d["candidates"]
    assert torch.allclose(load_theta(tmp_path / "theta_candidates.json"), res.theta_star, atol=1e-4)
    assert "band_diff_db" in data["diagnostics"]


def test_nan_triggers_restart(monkeypatch):
    di = _di_clips(n=2, sec=0.5)
    G = GrayboxAmp()
    calls = {"n": 0}
    orig = G.forward

    def flaky(*a, **k):
        calls["n"] += 1
        y = orig(*a, **k)
        return y * float("nan") if calls["n"] == 3 else y

    monkeypatch.setattr(G, "forward", flaky)
    res = fit(di.reshape(-1), di, Prior.full(), cfg=FitConfig(steps=6, clips_per_step=2, n_prior_samples=0), G=G)
    assert res.restarts >= 1
    assert torch.isfinite(res.theta_star).all()


def test_theta_star_uses_shared_reference_across_starts():
    """θ*는 손실이 가장 많이 준 시작점이 아니라 실제로 가장 잘 맞는 시작점이어야 한다."""
    torch.manual_seed(0)
    G = GrayboxAmp()
    di = _di_clips(n=3, sec=0.5)
    theta_true = theta_from_units({"pre1_gain_db": 26.0, "ts_treble_db": 6.0, "ts_bass_db": -5.0})
    with torch.no_grad():
        stem = G.render(di, theta_true.expand(len(di), -1)).reshape(-1)
    far = theta_true.clone()
    for n in FREE:
        far[INDEX[n]] = 1.0 - far[INDEX[n]]
    prior = _prior_fixing_all_but(theta_true, FREE)
    starts = torch.stack([far, theta_true]).double().numpy()
    cfg = FitConfig(steps=15, lr=0.05, clips_per_step=len(di), lambda_prior=0.0, patience=100,
                    eval_clips=len(di), lambda_anchor=0.0)
    res = fit(stem, di, prior, cfg=cfg, G=G, starts=starts, start_names=["far", "true"])
    assert res.start_names[int(torch.argmin(res.losses))] == "true", res.losses
    assert torch.allclose(res.theta_star, res.thetas[1])


def test_exchangeable_gains_stay_at_start():
    """in_gain·boost_level은 시작점 값에 고정, boost=none(boost_drive 고정)이면 in_gain은 푼다."""
    from ampcap.optimize import exchangeable_mask

    di = _di_clips(n=2, sec=0.5)
    G = GrayboxAmp()
    stem = (G(di, theta_from_units({"pre1_gain_db": 30.0}).expand(2, -1)).reshape(-1)).detach()
    start = np.full((1, N_PARAMS), 0.3)
    res = fit(stem, di, Prior.full(), cfg=FitConfig(steps=5, clips_per_step=2, lr=0.2), G=G,
              starts=start, start_names=["s"])
    for n in ("in_gain_db", "boost_level_db"):
        assert abs(float(res.thetas[0, INDEX[n]]) - 0.3) < 1e-5
    assert abs(float(res.thetas[0, INDEX["boost_drive_db"]]) - 0.3) > 1e-4
    assert set(res.diagnostics["frozen"]) == {"in_gain_db", "boost_level_db"}

    fixed = torch.zeros(N_PARAMS, dtype=torch.bool)
    fixed[INDEX["boost_drive_db"]] = fixed[INDEX["boost_level_db"]] = True
    m = exchangeable_mask(fixed)
    assert not m[INDEX["in_gain_db"]] and not m[INDEX["boost_level_db"]]


def test_min_move_selection_prefers_encoder_on_tie():
    di = _di_clips(n=2, sec=0.5)
    G = GrayboxAmp()
    th = theta_from_units({"pre1_gain_db": 20.0})
    stem = G(di, th.expand(2, -1)).reshape(-1).detach()
    starts = th.expand(2, -1).double().numpy()
    res = fit(stem, di, Prior.full(), cfg=FitConfig(steps=0, eval_clips=2), G=G,
              starts=starts, start_names=["prior_center", "encoder"])
    assert res.candidates[0] == 1 and torch.allclose(res.theta_star, res.thetas[1])
