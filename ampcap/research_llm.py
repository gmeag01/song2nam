"""②-0 LLM 장비 조사 (선택).

1. 곡 링크의 제목 정보에서 아티스트·곡명을 뽑고 사용자가 확인한다(오디오는 내려받지 않는다).
2. Claude + 웹 검색으로 두 단계 조사: 곡에서 쓴 장비 → 아티스트가 평소 쓰는 장비.
   평소 장비로 채운 필드는 확신도를 낮게(0.5) 매긴다.
3. 출력은 ②의 스키마 선택지 값만 허용하고 필드마다 출처 URL과 확신도를 받는다. 출처 없으면 unknown.
4. 사용자가 출처와 함께 초안을 보고 수정·승인한다. 확정 필드의 확신도는 1.0.

오류 처리: API 실패·시간 초과 → None(직접 입력 폼으로). 스키마 불일치 → 한 번 재요청,
또 실패하면 맞는 필드만 채운다.

API: Anthropic Claude (`anthropic` 패키지, 환경 변수 ANTHROPIC_API_KEY).
"""

from __future__ import annotations

import json
import re
from typing import Callable

from .session import BOOSTS, CABS, FAMILIES, FIELDS, GAINS, PICKUPS, model_to_family, validate

MODEL = "claude-opus-5-5"
USUAL_CONFIDENCE = 0.5

FIELD_OPTIONS = {p: opts for p, opts, _ in FIELDS}
CONF_KEY = {p: c for p, _, c in FIELDS}

SYSTEM = """You research the guitar rig used on a specific recording, for a tone-matching tool.
Search the web. Prefer sources about this exact song (interviews, rig rundowns, studio notes).
If the song-specific rig is not documented, fall back to the artist's usual rig of that era and
mark the field "scope": "usual". Never guess: a field without a source URL must be "unknown".

Return ONLY one JSON object, no prose, in this shape:
{
  "artist": str, "song": str,
  "fields": {
    "guitar.pickup": {"value": one of %(pickup)s or "unknown", "source": url or null, "scope": "song"|"usual", "confidence": 0..1, "note": str},
    "amp.family":    {"value": one of %(family)s or "unknown", ...same keys},
    "amp.model":     {"value": free text or "unknown", ...same keys},
    "amp.gain":      {"value": one of %(gain)s or "unknown", ...same keys},
    "cab.size":      {"value": one of %(cab)s or "unknown", ...same keys},
    "boost":         {"value": one of %(boost)s or "unknown", ...same keys}
  }
}
Family guide: fender = Fender-style clean/blackface/tweed; vox = AC30-style class A; british = Marshall/Orange
style; american_hg = Mesa Rectifier/Mark, 5150/6505, Soldano; modern_hg = Diezel, Engl, Friedman, modern
tight high-gain. boost "tube_screamer" = Tube Screamer-style overdrive used as a boost; "none" = no pedal before the amp.
""" % {"pickup": PICKUPS, "family": FAMILIES, "gain": GAINS, "cab": CABS, "boost": BOOSTS}


# ---------------------------------------------------------------- 1. 제목 정보


def parse_title(title: str) -> tuple[str, str]:
    """'Artist - Song (Official Video)' 같은 제목에서 (아티스트, 곡명). 사용자가 확인한다."""
    t = re.sub(r"[\(\[].*?(official|video|audio|lyrics|mv|remaster).*?[\)\]]", "", title, flags=re.I)
    t = t.strip(" -|")
    for sep in [" - ", " – ", " — ", " | "]:
        if sep in t:
            a, s = t.split(sep, 1)
            return a.strip(), s.strip().strip('"')
    return "", t.strip()


def fetch_link_title(url: str, timeout: float = 10.0) -> str | None:
    """링크 페이지의 <title>만 읽는다(oEmbed 우선). 오디오는 받지 않는다."""
    import urllib.parse
    import urllib.request

    try:
        if "youtube.com" in url or "youtu.be" in url:
            q = "https://www.youtube.com/oembed?format=json&url=" + urllib.parse.quote(url, safe="")
            with urllib.request.urlopen(q, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8")).get("title")
        with urllib.request.urlopen(url, timeout=timeout) as r:
            html = r.read(200_000).decode("utf-8", "ignore")
        m = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
        return m.group(1).strip() if m else None
    except Exception:
        return None


# ---------------------------------------------------------------- 2. LLM 호출


def call_claude(prompt: str, timeout: float = 180.0) -> str:
    """Claude + 웹 검색. 마지막 텍스트를 돌려준다. 실패 시 예외."""
    import anthropic

    client = anthropic.Anthropic(timeout=timeout, max_retries=2)
    messages = [{"role": "user", "content": prompt}]
    for _ in range(4):  # 서버 도구가 길어지면 pause_turn으로 멈출 수 있다 → 이어서 요청
        resp = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM,
            messages=messages,
            tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": 8}],
            output_config={"effort": "medium"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if resp.stop_reason == "refusal":
            raise RuntimeError("LLM이 요청을 거절함")
        if resp.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": resp.content})
            continue
        texts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
        return "\n".join(texts)
    raise TimeoutError("웹 검색이 끝나지 않음")


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("JSON 없음")
    return json.loads(m.group(0))


# ---------------------------------------------------------------- 3. 검증


def validate_draft(raw: dict) -> tuple[dict, list[str]]:
    """LLM 출력 → 정리된 초안과 문제 목록. 선택지 밖 값은 거부(unknown), 출처 없으면 unknown."""
    problems = []
    fields_in = raw.get("fields") if isinstance(raw, dict) else None
    if not isinstance(fields_in, dict):
        return {"artist": "", "song": "", "fields": {}}, ["fields 객체 없음"]
    out = {}
    for path in list(FIELD_OPTIONS) + ["amp.model"]:
        f = fields_in.get(path)
        if not isinstance(f, dict):
            problems.append(f"{path} 없음")
            out[path] = {"value": "unknown", "source": None, "confidence": 0.0, "scope": None, "note": ""}
            continue
        v = f.get("value", "unknown")
        src = f.get("source")
        scope = f.get("scope") if f.get("scope") in ("song", "usual") else "usual"
        try:
            conf = float(f.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
            problems.append(f"{path} confidence 형식 오류")
        conf = min(max(conf, 0.0), 1.0)
        if path in FIELD_OPTIONS and v not in FIELD_OPTIONS[path] + ["unknown"]:
            problems.append(f"{path} 선택지 밖 값 {v!r}")
            v = "unknown"
        if not (isinstance(src, str) and src.startswith("http")):
            src = None
            v = "unknown"
        if scope == "usual":
            conf = min(conf, USUAL_CONFIDENCE)
        if v == "unknown":
            conf = 0.0
        out[path] = {"value": v, "source": src, "confidence": conf, "scope": scope,
                     "note": str(f.get("note", ""))[:300]}
    draft = {"artist": str(raw.get("artist", "")), "song": str(raw.get("song", "")), "fields": out}
    return draft, problems


def _parse(text: str) -> tuple[dict | None, list[str]]:
    try:
        return validate_draft(_extract_json(text))
    except (ValueError, json.JSONDecodeError) as e:
        return None, [f"JSON 파싱 실패: {e}"]


def research(artist: str, song: str, call: Callable[[str], str] | None = None) -> dict | None:
    """두 단계 조사 초안. API 실패면 None(직접 입력 폼으로 전환)."""
    call = call or call_claude
    prompt = (f"Artist: {artist}\nSong: {song}\n"
              "Step 1: find the guitar rig used on this specific recording.\n"
              "Step 2: for fields still unknown, find the artist's usual rig of that era (scope=usual).")
    try:
        text = call(prompt)
    except Exception:  # API 실패, 시간 초과
        return None
    draft, problems = _parse(text)
    if draft is not None and not problems:
        draft["problems"] = []
        return draft

    # 스키마 불일치: 한 번 재요청
    draft2, problems2 = None, problems
    try:
        text2 = call(prompt + "\n\nYour previous answer did not match the schema: "
                     + "; ".join(problems) + ". Return only the JSON object.")
        draft2, problems2 = _parse(text2)
    except Exception:
        pass
    if draft2 is not None and not problems2:
        draft2["problems"] = []
        return draft2
    # 또 실패 → 맞는 필드만 채운 쪽(문제가 적은 쪽)
    cands = [(d, p) for d, p in [(draft, problems), (draft2, problems2)] if d is not None]
    if not cands:
        return None
    d, p = min(cands, key=lambda dp: len(dp[1]))
    d["problems"] = p
    return d


# ---------------------------------------------------------------- 4. 사용자 확인


def draft_to_session(draft: dict, confirmed: set[str] | None = None,
                     overrides: dict[str, str] | None = None,
                     capture_target: str = "rhythm", same_rig: bool = True) -> dict:
    """초안 → ② 세션 JSON. confirmed(사용자가 확정한 필드)의 확신도는 1.0.

    한 confidence 키(amp)에 여러 필드가 걸리면 그중 가장 낮은 값을 쓴다.
    """
    confirmed = confirmed or set()
    overrides = overrides or {}
    f = draft.get("fields", {})
    s: dict = {"artist": draft.get("artist", ""), "song": draft.get("song", ""),
               "capture_target": capture_target, "same_rig": same_rig,
               "guitar": {}, "amp": {}, "cab": {}}
    conf: dict[str, float] = {}
    for path in list(FIELD_OPTIONS) + ["amp.model"]:
        v = overrides.get(path, f.get(path, {}).get("value", "unknown"))
        c = 1.0 if (path in confirmed or path in overrides) else float(f.get(path, {}).get("confidence", 0.0))
        keys = path.split(".")
        if len(keys) == 2:
            s[keys[0]][keys[1]] = v
        else:
            s[keys[0]] = v
        if path in CONF_KEY and v != "unknown":
            k = CONF_KEY[path]
            conf[k] = min(conf.get(k, 1.0), c)
    if s["amp"].get("family", "unknown") == "unknown":
        fam = model_to_family(s["amp"].get("model"))
        if fam:
            s["amp"]["family"] = fam
    s["confidence"] = {k: round(max(v, 0.05), 3) for k, v in conf.items()}
    return validate(s)
