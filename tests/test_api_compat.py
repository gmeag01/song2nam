"""설치된 외부 패키지가 우리가 부르는 API를 실제로 갖고 있는지 확인한다(다운로드·학습 없음).

NAM·Demucs·anthropic은 버전마다 이름이 달라서, 맞지 않으면 Colab에서 오래 돌린 뒤에야 터진다.
"""

import inspect

import pytest


def _params(fn):
    return set(inspect.signature(fn).parameters)


def test_nam_train_api():
    core = pytest.importorskip("nam.train.core")
    p = _params(core.train)
    assert {"input_path", "output_path", "train_path", "epochs"} <= p
    assert {"delay", "latency"} & p, "지연 보정 인자 이름을 찾지 못함"
    if hasattr(core, "Architecture"):
        for a in ("standard", "lite"):
            core.Architecture(a)


def test_nam_loader_and_metadata():
    pytest.importorskip("nam")
    from ampcap.distill import load_nam  # noqa: F401  (경로 탐색 코드)

    found = False
    for mod in ("nam.models", "nam.models._from_nam"):
        try:
            m = __import__(mod, fromlist=["init_from_nam"])
            found = found or hasattr(m, "init_from_nam")
        except ImportError:
            pass
    assert found, "init_from_nam을 찾지 못함 → load_nam 경로 수정 필요"

    from nam.models.metadata import GearType, ToneType, UserMetadata

    assert {"name", "modeled_by", "gear_type", "gear_model", "tone_type"} <= _params(UserMetadata)
    GearType.AMP_CAB, GearType.AMP_PEDAL_CAB, ToneType.CLEAN, ToneType.CRUNCH, ToneType.HI_GAIN  # noqa: B018


def test_demucs_api():
    apply = pytest.importorskip("demucs.apply")
    from demucs.pretrained import get_model  # noqa: F401

    assert {"model", "mix", "device", "split", "overlap", "progress", "segment"} <= _params(apply.apply_model)


def test_anthropic_accepts_fallbacks_and_server_tools():
    anthropic = pytest.importorskip("anthropic")
    client = anthropic.Anthropic(api_key="test")
    p = _params(client.beta.messages.create)
    for name in ("model", "max_tokens", "system", "messages", "tools", "output_config", "betas", "fallbacks"):
        assert name in p, f"beta.messages.create에 {name} 인자가 없음"


def test_torchaudio_api():
    AF = pytest.importorskip("torchaudio.functional")
    assert "batching" in _params(AF.lfilter)
    assert {"norm", "mel_scale"} <= _params(AF.melscale_fbanks)


def test_pedalboard_effects():
    pb = pytest.importorskip("pedalboard")
    for n in ("Pedalboard", "LowShelfFilter", "PeakFilter", "HighShelfFilter", "Compressor", "Reverb"):
        assert hasattr(pb, n), n
