"""⑤ 파라미터 인코더.

스템 3초 → log-Mel(128) → CNN → Global Pooling → 오디오 벡터
[오디오 벡터 ; 세션 벡터(21)] → MLP → sigmoid → θ̂ (24)

손실: θ MSE + MR-STFT(G(x; θ̂) vs y). 실제 곡은 3초 창 여러 개의 θ̂ 중앙값.
인코더는 ⑥의 시작점과 정규화 기준이며, 없어도 파이프라인은 동작한다.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .graybox.amp import N_PARAMS, GrayboxAmp, rms
from .session import SESSION_DIM
from .stats import mel_filterbank


@dataclass
class EncoderConfig:
    sr: int = 48000
    n_fft: int = 2048
    hop: int = 512
    n_mels: int = 128
    channels: list = field(default_factory=lambda: [32, 64, 128, 128, 256])
    hidden: int = 256
    dropout: float = 0.1


class LogMel(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("window", torch.hann_window(cfg.n_fft), persistent=False)
        self.register_buffer("fb", mel_filterbank(cfg.n_fft // 2 + 1, 40.0, cfg.sr / 2 * 0.95,
                                                  cfg.n_mels, cfg.sr), persistent=False)

    def forward(self, x):  # (B, T) → (B, 1, M, L), 음량 불변(예시별 평균 dB 제거)
        X = torch.stft(x, self.cfg.n_fft, self.cfg.hop, window=self.window, center=True,
                       return_complex=True)
        mel = torch.einsum("bfl,fm->bml", X.abs().square(), self.fb)
        db = 10 * torch.log10(mel + 1e-10)
        db = db - db.mean(dim=(1, 2), keepdim=True)
        return (db / 20.0).unsqueeze(1)


class Encoder(nn.Module):
    def __init__(self, cfg: EncoderConfig | None = None):
        super().__init__()
        self.cfg = cfg or EncoderConfig()
        c = self.cfg
        self.mel = LogMel(c)
        layers, ch = [], 1
        for out in c.channels:
            layers += [nn.Conv2d(ch, out, 3, padding=1), nn.BatchNorm2d(out), nn.GELU(),
                       nn.MaxPool2d(2)]
            ch = out
        self.cnn = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(2 * ch + SESSION_DIM, c.hidden), nn.GELU(), nn.Dropout(c.dropout),
            nn.Linear(c.hidden, c.hidden), nn.GELU(),
            nn.Linear(c.hidden, N_PARAMS),
        )

    def forward(self, audio: torch.Tensor, session: torch.Tensor) -> torch.Tensor:
        h = self.cnn(self.mel(audio))                     # (B, C, M', L')
        h = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], -1)  # Global avg + max pooling
        return torch.sigmoid(self.head(torch.cat([h, session], -1)))


# ---------------------------------------------------------------- 손실


def mrstft(pred: torch.Tensor, target: torch.Tensor,
           ffts=(512, 1024, 2048), eps: float = 1e-7) -> torch.Tensor:
    """다중 해상도 STFT 거리(스펙트럼 수렴 + 로그 크기 L1). 배치 평균."""
    loss = 0.0
    for n in ffts:
        w = torch.hann_window(n, device=pred.device, dtype=pred.dtype)
        P = torch.stft(pred, n, n // 4, window=w, return_complex=True).abs()
        T = torch.stft(target, n, n // 4, window=w, return_complex=True).abs()
        sc = torch.linalg.norm(T - P, dim=(-2, -1)) / torch.linalg.norm(T, dim=(-2, -1)).clamp_min(eps)
        lm = (torch.log(P + eps) - torch.log(T + eps)).abs().mean(dim=(-2, -1))
        loss = loss + (sc + lm).mean()
    return loss / len(ffts)


def encoder_loss(model: Encoder, batch: dict, G: GrayboxAmp | None = None,
                 w_mrstft: float = 0.1, target_rms: float = 0.1):
    th_hat = model(batch["y_aug"], batch["session"])
    l_theta = (th_hat - batch["theta"]).square().mean()
    out = {"theta_mse": l_theta}
    total = l_theta
    if G is not None and w_mrstft > 0:
        y_hat = G(batch["x"], th_hat, mode="fft")
        y_hat = y_hat / rms(y_hat)[:, None] * target_rms
        l_spec = mrstft(y_hat, batch["y"])
        out["mrstft"] = l_spec
        total = total + w_mrstft * l_spec
    out["total"] = total
    return total, out, th_hat


# ---------------------------------------------------------------- 학습


@dataclass
class TrainConfig:
    steps: int = 20000
    lr: float = 3e-4
    weight_decay: float = 1e-4
    w_mrstft: float = 0.1
    mrstft_after: int = 2000      # 처음에는 θ MSE만으로 안정적으로 시작
    val_every: int = 500
    val_batches: int = 8
    ckpt_every: int = 1000
    max_hours: float = 4.0        # Colab 세션 한도 대비


def train(model: Encoder, train_gen, val_gen=None, cfg: TrainConfig | None = None,
          ckpt_path: str | Path = "runs/encoder/encoder.pt", device: str = "cuda",
          resume: bool = True, log=print) -> dict:
    """합성 데이터 생성기로 인코더를 학습한다. Colab 세션이 끊겨도 체크포인트에서 이어 간다."""
    cfg = cfg or TrainConfig()
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    G = GrayboxAmp().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps)
    start, history = 0, []
    if resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, history = ck["step"], ck.get("history", [])
        log(f"체크포인트에서 이어서 학습: step {start}")

    t0 = time.time()
    best_val = min([h["val"] for h in history if "val" in h], default=float("inf"))
    for step in range(start, cfg.steps):
        model.train()
        batch = train_gen.next_batch()
        w = cfg.w_mrstft if step >= cfg.mrstft_after else 0.0
        total, parts, _ = encoder_loss(model, batch, G, w)
        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 100 == 0:
            rec = {"step": step, **{k: float(v) for k, v in parts.items()}}
            history.append(rec)
            log(json.dumps(rec))
        if val_gen is not None and (step + 1) % cfg.val_every == 0:
            v = evaluate(model, val_gen, cfg.val_batches)
            history.append({"step": step + 1, "val": v})
            log(f"val theta_mse {v:.5f}")
            if v < best_val:
                best_val = v
                save(model, ckpt_path.with_name(ckpt_path.stem + "_best.pt"))
        if (step + 1) % cfg.ckpt_every == 0 or step + 1 == cfg.steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "step": step + 1, "history": history,
                        "config": asdict(model.cfg)}, ckpt_path)
        if (time.time() - t0) / 3600 > cfg.max_hours:
            log("시간 한도 도달, 저장 후 종료")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "step": step + 1, "history": history,
                        "config": asdict(model.cfg)}, ckpt_path)
            break
    return {"history": history, "best_val": best_val}


@torch.no_grad()
def evaluate(model: Encoder, gen, n_batches: int = 8) -> float:
    model.eval()
    vals = []
    for _ in range(n_batches):
        b = gen.next_batch()
        vals.append(float((model(b["y_aug"], b["session"]) - b["theta"]).square().mean()))
    return float(np.mean(vals))


def save(model: Encoder, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": asdict(model.cfg)}, path)


def load(path: str | Path, device: str = "cpu") -> Encoder:
    ck = torch.load(path, map_location=device)
    model = Encoder(EncoderConfig(**ck["config"]))
    model.load_state_dict(ck["model"])
    return model.to(device).eval()


# ---------------------------------------------------------------- 실제 곡


@torch.no_grad()
def predict_song(model: Encoder, stem: np.ndarray, session_vec: np.ndarray, sr: int = 48000,
                 win_sec: float = 3.0, hop_sec: float = 1.5, min_rms_db: float = -45.0,
                 batch: int = 16, target_rms: float = 0.1) -> np.ndarray:
    """3초 창 여러 개의 θ̂ 중앙값 (24,)."""
    model.eval()
    dev = next(model.parameters()).device
    win, hop = int(win_sec * sr), int(hop_sec * sr)
    if len(stem) < win:
        stem = np.pad(stem, (0, win - len(stem)))
    wins = []
    for s in range(0, len(stem) - win + 1, hop):
        w = stem[s:s + win]
        r = np.sqrt(np.mean(w ** 2))
        if 20 * np.log10(r + 1e-12) > min_rms_db:
            wins.append(w / (r + 1e-8) * target_rms)
    if not wins:
        wins = [stem[:win]]
    sv = torch.as_tensor(session_vec, dtype=torch.float32, device=dev)
    preds = []
    for i in range(0, len(wins), batch):
        a = torch.as_tensor(np.stack(wins[i:i + batch]), dtype=torch.float32, device=dev)
        preds.append(model(a, sv.expand(len(a), -1)).cpu())
    return torch.cat(preds).median(0).values.numpy()
