"""⑥ 통계 4종. 모두 크기 기반이고 전체 음량 차이를 제거한다(설계 문서 4.7).

1. ltas:  장기 평균 스펙트럼 — 연주 프레임 log-Mel 평균 (M,)
2. level: 레벨별 스펙트럼 — 신호 내 음량 분위수 4구간별 평균 스펙트럼 (4, M)
3. flat:  레벨 구간별 스펙트럼 평탄도(dB) (4,)
4. dyn:   다이내믹 — 크레스트 팩터 평균·분산, 프레임 음량 분포 폭 (3,)

입력 x는 (..., N, T): 마지막 두 축(클립 N개 × 프레임)을 한 덩어리로 모아 통계를 낸다.
앞쪽 축(...)은 독립 그룹(예: ⑥의 시작점들)이다.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

STAT_KEYS = ("ltas", "level", "flat", "dyn")
EPS = 1e-12
FLOOR_DB = -100.0  # 신호별 최대 mel 칸 대비 바닥값. 절대 바닥값을 쓰면 음량에 따라 통계가 바뀐다.


@dataclass
class StatConfig:
    sr: int = 48000
    n_fft: int = 2048
    hop: int = 512
    n_mels: int = 128
    f_min: float = 40.0
    f_max: float = 16000.0
    n_levels: int = 4
    silence_db: float = -50.0   # 신호별 최대 프레임 대비
    tau_db: float = 1.0         # 레벨 구간 소프트 경계 폭


def mel_filterbank(n_freqs: int, f_min: float, f_max: float, n_mels: int, sr: int) -> torch.Tensor:
    """(n_freqs, n_mels) HTK mel 삼각 필터."""
    import torchaudio.functional as AF

    return AF.melscale_fbanks(n_freqs, f_min, f_max, n_mels, sr, norm=None, mel_scale="htk")


class Stats(nn.Module):
    def __init__(self, cfg: StatConfig | None = None):
        super().__init__()
        self.cfg = cfg or StatConfig()
        c = self.cfg
        self.register_buffer("window", torch.hann_window(c.n_fft), persistent=False)
        self.register_buffer("fb", mel_filterbank(c.n_fft // 2 + 1, c.f_min, c.f_max, c.n_mels, c.sr),
                             persistent=False)

    # ------------------------------------------------------------ 프레임 단위 특징

    def frames(self, x: torch.Tensor):
        """x (K, T) → mel_db (K, M, L), mel_pow (K, M, L), loud_db (K, L), crest_db (K, L)."""
        c = self.cfg
        X = torch.stft(x, c.n_fft, c.hop, window=self.window.to(x.dtype), center=True,
                       pad_mode="constant", return_complex=True)
        pw = X.real.square() + X.imag.square()                       # (K, F, L)
        mel = torch.einsum("kfl,fm->kml", pw, self.fb.to(pw.dtype))  # (K, M, L)
        # 거의 빈 칸(예: mp3의 16kHz 이상)이 음량과 무관한 상수가 되지 않도록 신호 기준 상대 바닥값을 더한다.
        floor = mel.detach().amax(dim=(1, 2), keepdim=True) * 10 ** (FLOOR_DB / 10) + EPS
        mel = mel + floor
        mel_db = 10 * torch.log10(mel)

        xp = torch.nn.functional.pad(x, (c.n_fft // 2, c.n_fft // 2))
        fr = xp.unfold(-1, c.n_fft, c.hop)[:, : mel.shape[-1]]       # (K, L, n_fft)
        ms = fr.square().mean(-1)
        loud_db = 10 * torch.log10(ms + EPS)
        peak = fr.abs().amax(-1)
        crest_db = 20 * torch.log10((peak + 1e-9) / (ms.sqrt() + 1e-9))
        return mel_db, mel, loud_db, crest_db

    # ------------------------------------------------------------ 통계

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        c = self.cfg
        if x.dim() == 1:
            x = x[None, None]
        elif x.dim() == 2:
            x = x[None]
        lead = x.shape[:-2]
        N, T = x.shape[-2:]
        xf = x.reshape(-1, T)
        mel_db, mel, loud, crest = self.frames(xf)
        K, M, L = mel_db.shape
        G = K // N

        # 연주 프레임: 신호별 최대 프레임 대비 silence_db 이상 (경계는 미분하지 않는다)
        with torch.no_grad():
            active = (loud > loud.amax(-1, keepdim=True) + c.silence_db).to(x.dtype)

        # (G, N*L)로 모은다
        def pool(t):  # (K, ..., L) → (G, ..., N*L)
            if t.dim() == 2:
                return t.reshape(G, N, L).reshape(G, N * L)
            return t.reshape(G, N, M, L).permute(0, 2, 1, 3).reshape(G, M, N * L)

        db = pool(mel_db)
        pw = pool(mel)
        ld = pool(loud)
        cr = pool(crest)
        w = pool(active)

        def wmean(v, wt):  # v (..., P), wt (..., P)
            return (v * wt).sum(-1) / wt.sum(-1).clamp_min(1e-6)

        # 1. 장기 평균 스펙트럼
        ltas = wmean(db, w[:, None, :])
        ltas = ltas - ltas.mean(-1, keepdim=True)

        # 2. 레벨별 스펙트럼: 연주 프레임 음량의 분위수로 소프트하게 4구간
        with torch.no_grad():
            ld_nan = torch.where(w > 0, ld, torch.full_like(ld, float("nan")))
            qs = torch.linspace(0, 1, c.n_levels + 1, dtype=ld.dtype, device=ld.device)[1:-1]
            thr = torch.nanquantile(ld_nan, qs, dim=-1)              # (n_levels-1, G)
        s = torch.sigmoid((ld[None] - thr[..., None]) / c.tau_db)    # (n_levels-1, G, P)
        ones = torch.ones_like(s[:1])
        zeros = torch.zeros_like(s[:1])
        upper = torch.cat([ones, s], 0)
        lower = torch.cat([s, zeros], 0)
        memb = (upper - lower).permute(1, 0, 2) * w[:, None, :]      # (G, n_levels, P)

        level = torch.einsum("gmp,gkp->gkm", db, memb) / memb.sum(-1, keepdim=True).clamp_min(1e-6)
        level = level - level.mean(-1, keepdim=True)

        # 3. 레벨 구간별 스펙트럼 평탄도 (dB): 기하평균/산술평균
        flat_frame = db.mean(1) - 10 * torch.log10(pw.mean(1))  # (G, P)
        flat = wmean(flat_frame[:, None, :], memb)                    # (G, n_levels)

        # 4. 다이내믹
        cm = wmean(cr, w)
        cv = wmean((cr - cm[:, None]).square(), w)
        lm = wmean(ld, w)
        lw = wmean((ld - lm[:, None]).square(), w).clamp_min(1e-6).sqrt()
        dyn = torch.stack([cm, cv.clamp_min(1e-6).sqrt(), lw], -1)

        out = {"ltas": ltas, "level": level, "flat": flat, "dyn": dyn}
        return {k: v.reshape(*lead, *v.shape[1:]) for k, v in out.items()}


def stat_losses(pred: dict, target: dict, keys=STAT_KEYS) -> dict[str, torch.Tensor]:
    """통계별 L1 (그룹 축 유지). target은 그룹 축 1개 또는 브로드캐스트 가능."""
    out = {}
    for k in keys:
        d = (pred[k] - target[k]).abs()
        out[k] = d.reshape(*d.shape[: pred[k].dim() - _feat_dims(k)], -1).mean(-1)
    return out


def _feat_dims(k: str) -> int:
    return 2 if k == "level" else 1


def band_diff_db(pred_ltas: torch.Tensor, target_ltas: torch.Tensor, cfg: StatConfig | None = None,
                 edges=(40, 250, 2000, 6000, 16000)) -> dict[str, float]:
    """장기 평균 스펙트럼의 대역별 dB 차이(예측 − 타깃). 진단·⑨ 그래프용."""
    cfg = cfg or StatConfig()
    import numpy as np

    m = np.linspace(2595 * np.log10(1 + cfg.f_min / 700), 2595 * np.log10(1 + cfg.f_max / 700),
                    cfg.n_mels + 2)[1:-1]
    hz = 700 * (10 ** (m / 2595) - 1)
    d = (pred_ltas - target_ltas).detach().cpu().double().reshape(-1, cfg.n_mels).mean(0).numpy()
    out = {}
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (hz >= a) & (hz < b)
        if sel.any():
            out[f"{a}-{b}Hz"] = float(d[sel].mean())
    return out


def describe_residual(bands: dict[str, float], tol_db: float = 2.0) -> list[str]:
    """대역 차이를 사람이 읽을 문장으로 (예: '고역 부족 3.1 dB')."""
    names = {0: "저역", 1: "중저역", 2: "중고역", 3: "고역"}
    msgs = []
    for i, (k, v) in enumerate(bands.items()):
        if abs(v) >= tol_db:
            msgs.append(f"{names.get(i, k)}({k}) {'과다' if v > 0 else '부족'} {abs(v):.1f} dB")
    return msgs
