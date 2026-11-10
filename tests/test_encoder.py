"""⑤ 인코더: 작은 배치 과적합, 실제 곡 중앙값 예측 모양, 저장·불러오기."""

import numpy as np
import torch

from ampcap.encoder import Encoder, EncoderConfig, encoder_loss, load, mrstft, predict_song, save
from ampcap.graybox.amp import N_PARAMS, GrayboxAmp
from ampcap.session import SESSION_DIM

SR = 48000


def _small():
    return Encoder(EncoderConfig(n_mels=64, channels=[8, 16, 32], hidden=64, dropout=0.0))


def test_overfits_small_batch():
    torch.manual_seed(0)
    model = _small()
    B, T = 4, SR // 2
    batch = {
        "y_aug": torch.randn(B, T) * 0.1,
        "session": torch.rand(B, SESSION_DIM),
        "theta": torch.rand(B, N_PARAMS),
    }
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = None
    for step in range(400):
        model.train()
        loss, _, _ = encoder_loss(model, batch, G=None, w_mrstft=0.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else float(loss)
    model.eval()
    with torch.no_grad():
        final = float((model(batch["y_aug"], batch["session"]) - batch["theta"]).square().mean())
    print("θ MSE", first, "→", final)
    assert final < 1e-3


def test_loss_with_mrstft_is_differentiable():
    model = _small()
    G = GrayboxAmp()
    x = torch.randn(2, 9600) * 0.05
    th = torch.rand(2, N_PARAMS)
    with torch.no_grad():
        y = G(x, th)
        y = y / y.square().mean(-1, keepdim=True).sqrt() * 0.1
    batch = {"x": x, "y": y, "y_aug": y, "session": torch.rand(2, SESSION_DIM), "theta": th}
    loss, parts, _ = encoder_loss(model, batch, G=G, w_mrstft=0.1)
    loss.backward()
    assert "mrstft" in parts and torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_mrstft_zero_for_identical():
    x = torch.randn(2, 8000)
    assert float(mrstft(x, x)) < 1e-5
    assert float(mrstft(x, x * 0.5)) > 0.1


def test_predict_song_and_save_load(tmp_path):
    model = _small().eval()
    stem = np.random.default_rng(0).standard_normal(SR * 7).astype(np.float32) * 0.1
    th = predict_song(model, stem, np.zeros(SESSION_DIM, np.float32))
    assert th.shape == (N_PARAMS,) and np.all((th >= 0) & (th <= 1))
    save(model, tmp_path / "enc.pt")
    m2 = load(tmp_path / "enc.pt")
    th2 = predict_song(m2, stem, np.zeros(SESSION_DIM, np.float32))
    np.testing.assert_allclose(th, th2, atol=1e-5)


def test_theta_version_mismatch_falls_back_to_no_encoder(tmp_path, monkeypatch):
    """설계 4.6: θ 버전이 다른 체크포인트는 오류, 앱·파이프라인은 인코더 없이 진행."""
    import pytest

    from ampcap import encoder as E

    p = tmp_path / "enc.pt"
    save(_small(), p, val=0.1)
    assert torch.load(p)["theta_version"] == E.THETA_VERSION
    monkeypatch.setattr(E, "THETA_VERSION", E.THETA_VERSION + 1)
    with pytest.raises(E.ThetaVersionError):
        load(p)
    msgs = []
    assert E.try_load(p, log=msgs.append) is None and "θ 버전" in msgs[0]
    assert E.try_load(tmp_path / "none.pt", log=msgs.append) is None and "없습니다" in msgs[1]
    assert E.try_load(None) is None
