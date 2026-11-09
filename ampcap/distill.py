"""⑦ .nam 증류.

1. θ*를 G에 고정하고 lfilter + 2배 오버샘플링으로 렌더링한다.
2. NAM 표준 학습 입력(v3_0_0.wav 등)을 G에 통과시켜 output.wav를 만든다.
   입력은 ⑥ DI 풀과 같은 음량(RMS)으로 맞춰 G에 넣고, 그 배율을 "입력 레벨"로 기록한다.
   출력은 클리핑되지 않게 피크를 맞춘다.
3. neural-amp-modeler로 학습한다(지연 보정 0).
4. .nam 메타데이터에 이름, 장비 정보, "음원에서 생성됨"을 기록한다.
5. 학습에 쓰지 않은 DI로 G와 .nam을 비교한다(ESR).

NAM 표준 입력 파일은 저장소에 넣지 않는다. NAM 문서의 링크에서 받아 data/nam/에 둔다.
"""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import numpy as np
import torch

from .graybox.amp import GrayboxAmp, theta_to_units_dict
from .evaluate import esr
from .separate import SR, save_wav

OUTPUT_PEAK = 0.89  # ≈ −1 dBFS
ESR_TARGET = 0.01


def active_rms(x: np.ndarray, silence_db: float = -50.0) -> float:
    from .separate import active_mask

    m = active_mask(x, silence_db=silence_db)
    frame = 1024
    sel = np.repeat(m, frame)[: len(x)]
    if sel.sum() == 0:
        return float(np.sqrt(np.mean(x ** 2) + 1e-12))
    return float(np.sqrt(np.mean(x[: len(sel)][sel] ** 2) + 1e-12))


# ---------------------------------------------------------------- 1~2. 렌더링


def render_nam_output(theta: torch.Tensor, nam_input_path: str | Path, out_dir: str | Path,
                      di_ref_rms: float, device: str | None = None,
                      G: GrayboxAmp | None = None) -> dict:
    """NAM 표준 입력 → G(θ*) → output.wav. 반환: 입력 레벨 정보.

    di_ref_rms: ⑥에서 쓴 DI 풀의 연주 RMS. NAM 입력을 이 RMS로 맞춰 G에 넣는다.
    사용자는 DAW에서 DI 트랙을 같은 기준(input_gain_db)으로 맞춘다.
    """
    import soundfile as sf

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = sf.info(str(nam_input_path))
    if info.samplerate != SR:
        raise ValueError(f"NAM 입력은 {SR}Hz여야 한다: {info.samplerate}")
    x, _ = sf.read(str(nam_input_path), dtype="float32", always_2d=True)
    x = x[:, 0]

    in_rms = active_rms(x)
    gain = di_ref_rms / in_rms
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    G = (G or GrayboxAmp()).to(device)
    xt = torch.from_numpy(x * gain).to(device)[None]
    y = G.render(xt, theta.reshape(1, -1).float().to(device))[0].cpu().numpy()
    peak = float(np.abs(y).max()) + 1e-12
    out_gain = OUTPUT_PEAK / peak
    y = y * out_gain
    # NAM이 입력과 같은 형식을 기대하므로 같은 subtype으로 쓴다.
    save_wav(out_dir / "output.wav", y, SR, subtype=info.subtype)
    level = {
        "input_gain_db": 20 * math.log10(gain),
        "input_active_rms_dbfs": 20 * math.log10(in_rms),
        "di_ref_rms_dbfs": 20 * math.log10(di_ref_rms),
        "output_gain_db": 20 * math.log10(out_gain),
        "note": "DI 트랙 연주 RMS를 di_ref_rms_dbfs에 맞추면 ⑥과 같은 드라이브가 된다.",
    }
    (out_dir / "input_level.json").write_text(json.dumps(level, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    return level


# ---------------------------------------------------------------- 3. NAM 학습


def _call_with_supported(fn, **kwargs):
    """설치된 NAM 버전이 받는 인자만 넘긴다(버전마다 delay/latency 등 이름이 다르다)."""
    sig = inspect.signature(fn)
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(**kwargs)
    return fn(**{k: v for k, v in kwargs.items() if k in sig.parameters})


def train_nam(input_path: str | Path, output_path: str | Path, train_dir: str | Path,
              epochs: int = 100, architecture: str = "standard", seed: int = 0):
    """neural-amp-modeler 학습. 반환: 학습된 모델 객체(lightning 모듈)."""
    from nam.train import core

    arch = architecture
    try:
        from nam.train.core import Architecture

        arch = Architecture(architecture)
    except Exception:
        pass

    train_dir = Path(train_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    out = _call_with_supported(
        core.train,
        input_path=str(input_path), output_path=str(output_path), train_path=str(train_dir),
        epochs=epochs, delay=0, latency=0, architecture=arch, model_type="WaveNet",
        seed=seed, save_plot=True, silent=True, modelname="model", ignore_checks=False,
        local=True, fit_mrstft=True,
    )
    model = getattr(out, "model", out)
    if model is None:
        raise RuntimeError("NAM 학습 실패(체크 통과 못함). train_dir 로그를 확인하세요.")
    return model


def export_nam(model, out_dir: str | Path, name: str, session: dict | None, theta: torch.Tensor,
               level: dict | None = None) -> Path:
    """model.nam 저장 후 메타데이터를 덧붙인다."""
    out_dir = Path(out_dir)
    net = getattr(model, "net", model)
    kw = {"basename": "model"}
    try:
        from nam.models.metadata import GearType, ToneType, UserMetadata

        gain = (session or {}).get("amp", {}).get("gain", "unknown")
        tone = {"low": ToneType.CLEAN, "mid": ToneType.CRUNCH, "high": ToneType.HI_GAIN}.get(gain)
        kw["user_metadata"] = UserMetadata(
            name=name, modeled_by="ampcap (음원에서 생성됨)", gear_type=GearType.AMP_CAB,
            gear_make=None, gear_model=(session or {}).get("amp", {}).get("model"), tone_type=tone)
    except Exception:
        pass
    _call_with_supported(net.export, outdir=out_dir, **kw)
    path = out_dir / "model.nam"

    d = json.loads(path.read_text(encoding="utf-8"))
    meta = d.setdefault("metadata", {}) or {}
    meta.setdefault("name", name)
    meta["ampcap"] = {
        "generated_from": "음원에서 생성됨 (song2nam/ampcap)",
        "session": session, "theta": [float(v) for v in theta],
        "theta_units": theta_to_units_dict(theta), "input_level": level,
        "note": "캐비닛 포함. NAM 플러그인의 IR 로더를 끄고 사용.",
    }
    d["metadata"] = meta
    path.write_text(json.dumps(d), encoding="utf-8")
    return path


# ---------------------------------------------------------------- 5. 검증


def load_nam(path: str | Path):
    """.nam → PyTorch 모듈. NAM 버전마다 경로가 달라 몇 군데를 시도한다."""
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if cfg.get("architecture") == "SlimmableContainer":
        # NAM 0.13의 lite 내보내기는 크기별 하위 모델 묶음인데 init_from_nam이 못 읽는다. 가장 큰 모델로 확인.
        cfg = max(cfg["config"]["submodels"], key=lambda s: s["max_value"])["model"]
    for mod, fn in [("nam.models", "init_from_nam"), ("nam.models._from_nam", "init_from_nam")]:
        try:
            m = __import__(mod, fromlist=[fn])
            return getattr(m, fn)(cfg)
        except (ImportError, AttributeError):
            continue
    raise ImportError("설치된 neural-amp-modeler에서 .nam 로더를 찾지 못했다")


@torch.no_grad()
def run_nam(net, x: np.ndarray, device: str = "cpu") -> np.ndarray:
    net = net.to(device).eval()
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
    try:
        y = net(xt, pad_start=True)
    except TypeError:
        y = net(xt)
    return y.reshape(-1).cpu().numpy()


def verify(theta: torch.Tensor, net, di_clips: list[np.ndarray], input_gain_db: float,
           output_gain_db: float, device: str = "cpu", G: GrayboxAmp | None = None) -> dict:
    """학습에 쓰지 않은 DI로 G(θ*)와 .nam 출력 비교.

    DI는 NAM 입력 기준 레벨(= di_ref / 10^(input_gain_db/20))로 넣는다.
    """
    G = (G or GrayboxAmp()).to(device)
    gi, go = 10 ** (input_gain_db / 20), 10 ** (output_gain_db / 20)
    vals = []
    for di in di_clips:
        x_nam = di / gi                 # NAM 입력 기준 레벨의 DI
        y_g = G.render(torch.from_numpy(di.astype(np.float32)).to(device)[None],
                       theta.reshape(1, -1).float().to(device))[0].cpu().numpy() * go
        y_n = run_nam(net, x_nam, device)
        vals.append(esr(y_n, y_g))
    return {"esr_mean": float(np.mean(vals)), "esr_each": [float(v) for v in vals],
            "ok": bool(np.mean(vals) < ESR_TARGET)}


# ---------------------------------------------------------------- 전체


def distill(theta: torch.Tensor, run_dir: str | Path, nam_input_path: str | Path, di_ref_rms: float,
            session: dict | None = None, name: str = "ampcap", epochs: int = 100,
            architecture: str = "standard", verify_di: list[np.ndarray] | None = None,
            device: str | None = None) -> dict:
    run_dir = Path(run_dir)
    work = run_dir / "distill"
    level = render_nam_output(theta, nam_input_path, work, di_ref_rms, device=device)
    model = train_nam(nam_input_path, work / "output.wav", work / "train",
                      epochs=epochs, architecture=architecture)
    path = export_nam(model, run_dir, name, session, theta, level)
    result = {"nam_path": str(path), "input_level": level}
    if verify_di:
        try:
            net = load_nam(path)
        except ImportError:
            net = getattr(model, "net", model)
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        result["verify"] = verify(theta, net, verify_di, level["input_gain_db"],
                                  level["output_gain_db"], device=dev)
        if not result["verify"]["ok"]:
            result["warning"] = (f"G와 .nam의 ESR {result['verify']['esr_mean']:.4f}이 "
                                 f"목표 {ESR_TARGET}보다 큽니다. 더 큰 모델(architecture='standard')이나 "
                                 "더 긴 학습(epochs)으로 다시 학습할 수 있습니다.")
    (run_dir / "distill.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    return result


__all__ = ["render_nam_output", "train_nam", "export_nam", "load_nam", "verify", "distill"]
