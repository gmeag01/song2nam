"""②-0 LLM 장비 조사: 모의 응답으로 스키마 검증 (실제 API를 부르지 않는다)."""

import json

from ampcap import research_llm as R
from ampcap import session as S

GOOD = {
    "artist": "A", "song": "B",
    "fields": {
        "guitar.pickup": {"value": "humbucker", "source": "https://ex.com/1", "scope": "song", "confidence": 0.9},
        "amp.family": {"value": "british", "source": "https://ex.com/2", "scope": "song", "confidence": 0.8},
        "amp.model": {"value": "JCM800", "source": "https://ex.com/2", "scope": "song", "confidence": 0.8},
        "amp.gain": {"value": "high", "source": "https://ex.com/3", "scope": "usual", "confidence": 0.9},
        "cab.size": {"value": "4x12", "source": None, "scope": "song", "confidence": 0.9},
        "boost": {"value": "tube_screamer", "source": "https://ex.com/4", "scope": "song", "confidence": 0.7},
    },
}


def _mock(*responses):
    it = iter(responses)
    calls = []

    def call(prompt):
        calls.append(prompt)
        r = next(it)
        if isinstance(r, Exception):
            raise r
        return r if isinstance(r, str) else "검색 결과 요약...\n" + json.dumps(r)

    call.calls = calls
    return call


def test_valid_response_is_cleaned():
    d = R.research("A", "B", call=_mock(GOOD))
    f = d["fields"]
    assert f["amp.family"]["value"] == "british"
    assert f["cab.size"]["value"] == "unknown"          # 출처 없음 → unknown
    assert f["amp.gain"]["confidence"] <= R.USUAL_CONFIDENCE  # 평소 장비 → 확신도 낮춤
    assert d["problems"] == []


def test_out_of_schema_value_rejected_then_retry():
    bad = json.loads(json.dumps(GOOD))
    bad["fields"]["amp.family"]["value"] = "marshall"
    call = _mock(bad, GOOD)
    d = R.research("A", "B", call=call)
    assert len(call.calls) == 2 and "did not match" in call.calls[1]
    assert d["fields"]["amp.family"]["value"] == "british"


def test_second_failure_keeps_valid_fields_only():
    bad = json.loads(json.dumps(GOOD))
    bad["fields"]["amp.family"]["value"] = "marshall"
    d = R.research("A", "B", call=_mock(bad, "죄송합니다, JSON이 아닙니다"))
    assert d["fields"]["amp.family"]["value"] == "unknown"
    assert d["fields"]["guitar.pickup"]["value"] == "humbucker"
    assert d["problems"]


def test_api_failure_returns_none():
    assert R.research("A", "B", call=_mock(TimeoutError("timeout"))) is None


def test_draft_to_session_confirmed_is_full_confidence():
    d = R.research("A", "B", call=_mock(GOOD))
    s = R.draft_to_session(d, confirmed={"amp.family", "amp.gain", "amp.model"})
    S.validate(s)
    assert s["amp"]["family"] == "british" and s["confidence"]["amp"] == 1.0
    assert s["cab"]["size"] == "unknown"
    s2 = R.draft_to_session(d, overrides={"cab.size": "2x12"})
    assert s2["cab"]["size"] == "2x12" and s2["confidence"]["cab"] == 1.0
    S.compute_prior(s2)


def test_parse_title():
    assert R.parse_title("Metallica - Master of Puppets (Official Audio)") == ("Metallica", "Master of Puppets")
    assert R.parse_title("Song Only")[1] == "Song Only"


def test_sourceless_model_does_not_become_full_confidence_family():
    raw = json.loads(json.dumps(GOOD))
    raw["fields"]["amp.family"] = {"value": "unknown", "source": None, "scope": "song", "confidence": 0}
    raw["fields"]["amp.gain"] = {"value": "unknown", "source": None, "scope": "song", "confidence": 0}
    raw["fields"]["amp.model"] = {"value": "JCM800", "source": None, "scope": "song", "confidence": 0.9}
    d = R.research("A", "B", call=_mock(raw))
    assert d["fields"]["amp.model"]["value"] == "unknown"
    s = R.draft_to_session(d)
    assert s["amp"]["family"] == "unknown"
