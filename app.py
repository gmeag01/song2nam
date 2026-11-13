"""Gradio 데모 앱 (설계 문서 5장).

화면 흐름(괄호는 모듈 번호): 1. 업로드·분리(①) → 2. 구간 선택(①) → 3. 세션 정보(②) → 4. 톤 찾기(⑥) → 5. .nam 생성(⑦)

    python app.py [--share] [--egdb data/EGDB/audio_DI] [--nam-input data/nam/v3_0_0.wav] [--encoder ...]

Colab에서는 --share로 공개 링크를 띄운다. 모든 중간 결과는 runs/<날짜_곡명>/에 저장되며,
"실행 폴더 불러오기"로 미리 만든 결과를 불러올 수 있다(시연 대비).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from ampcap import separate
from ampcap import session as sess_mod
from ampcap.encoder import default_encoder_path
from ampcap.graybox.amp import BLOCKS, NAMES, GrayboxAmp
from ampcap.pipeline import load_di_pool, make_run_dir
from ampcap.synth_data import DI_REF_RMS, di_reader_for_split, normalize_di

ARGS = argparse.Namespace(egdb=os.environ.get("AMPCAP_EGDB", "data/EGDB/audio_DI"),
                          nam_input=os.environ.get("AMPCAP_NAM_INPUT", "data/nam/v3_0_0.wav"),
                          encoder=os.environ.get("AMPCAP_ENCODER", str(default_encoder_path())), split_file="data/egdb_split.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCOPE_KO = {"song": "곡 단위 장비", "usual": "평소 장비"}  # ②-0 출처 범위 표시


# ---------------------------------------------------------------- 그림


def fig_waveform(stem: np.ndarray, sr: int, segments, recommended):
    import matplotlib.pyplot as plt

    mono = stem.mean(0) if stem.ndim == 2 else stem
    step = max(1, len(mono) // 4000)
    t = np.arange(0, len(mono), step) / sr
    fig, ax = plt.subplots(figsize=(10, 2.4))
    ax.plot(t, mono[::step], lw=0.4, color="#555")
    for a, b in recommended or []:
        ax.axvspan(a, b, color="#4caf50", alpha=0.2)
    for a, b in segments or []:
        ax.axvspan(a, b, color="#2196f3", alpha=0.35)
    ax.set_xlabel("초 (초록: 추천 구간, 파랑: 선택 구간)")
    ax.set_yticks([])
    fig.tight_layout()
    return fig


def fig_curve(curve):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3))
    if curve:
        c = np.array(curve)
        ax.plot(c, lw=1)
        ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    fig.tight_layout()
    return fig


def fig_theta(theta: torch.Tensor):
    """θ를 블록별 노브(0~1 막대)와 블록별 주파수 응답으로."""
    import matplotlib.pyplot as plt

    G = GrayboxAmp()
    resp = G.block_responses(theta)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    colors = plt.cm.tab10(np.arange(len(BLOCKS)))
    for (name, sl), c in zip(BLOCKS.items(), colors):
        idx = list(range(sl.start, sl.stop))
        a1.barh(idx, theta[sl].numpy(), color=c, label=name)
    a1.set_yticks(range(len(NAMES)), NAMES, fontsize=7)
    a1.invert_yaxis()
    a1.set_xlim(0, 1)
    a1.legend(fontsize=7)
    f = resp["freqs"].numpy()
    for (name, _), c in zip(BLOCKS.items(), colors):
        a2.semilogx(f[1:], resp[name].numpy()[1:], color=c, label=name)
    a2.set_xlim(30, 20000)
    a2.set_ylim(-30, 15)
    a2.set_xlabel("Hz")
    a2.set_ylabel("dB")
    a2.legend(fontsize=7)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- 단계


def do_separate(audio_file, song_name, state):
    if audio_file is None:
        raise ValueError("음원 파일을 올려 주세요")
    song = song_name or Path(audio_file).stem
    run_dir = make_run_dir(song)
    audio = separate.load_audio(audio_file)
    stem = separate.separate_guitar(audio, device=DEVICE)
    separate.save_wav(run_dir / "stem_full.wav", stem)
    rec = separate.recommend_segments(stem)
    state = {"run_dir": str(run_dir), "song": song, "recommended": rec, "segments": []}
    return (fig_waveform(stem, separate.SR, [], rec), str(run_dir / "stem_full.wav"),
            json.dumps(rec), state, f"실행 폴더: {run_dir}")


def _stem(state):
    return separate.load_audio(Path(state["run_dir"]) / "stem_full.wav")


def preview_segment(start, end, state):
    stem = _stem(state).mean(0)
    s, e = int(start * separate.SR), int(end * separate.SR)
    return (separate.SR, stem[s:e])


def add_segment(start, end, state):
    state["segments"].append([float(start), float(end)])
    return fig_waveform(_stem(state), separate.SR, state["segments"], state["recommended"]), \
        json.dumps(state["segments"]), state


def use_recommended(state):
    state["segments"] = list(state["recommended"])
    return fig_waveform(_stem(state), separate.SR, state["segments"], state["recommended"]), \
        json.dumps(state["segments"]), state


def check_segments(target, same_rig, state):
    stem = _stem(state)
    mono, rep = separate.check_and_build(stem, state["segments"], song=state["song"], same_rig=same_rig)
    separate.save_outputs(state["run_dir"], mono, rep)
    msgs = [separate.GUIDANCE[target], f"유효 연주 길이(무음 제외): {rep.active_sec:.1f}초"]
    if rep.message:
        msgs.append("⛔ " + rep.message)
    mg = separate.multi_guitar_message(rep.warnings)
    if mg:
        msgs.append(mg)
    if "clipping" in rep.warnings:
        msgs.append("경고: 클리핑이 있습니다.")
    if any(w.startswith("full_stem_used") for w in rep.warnings):
        msgs.append("구간이 없어 전체 스템을 씁니다(여러 기타가 섞인 평균 톤).")
    state["ok"] = rep.ok
    return "\n".join(msgs), state


def llm_research(link_or_title, state):
    from ampcap import research_llm as rl

    title = rl.fetch_link_title(link_or_title) if link_or_title.startswith("http") else link_or_title
    import gradio as gr

    artist, song = rl.parse_title(title or "")
    if not artist and not song:
        return "제목을 읽지 못했습니다. '아티스트 - 곡명'으로 입력하세요.", *([gr.update()] * 6), state
    draft = rl.research(artist, song)
    if draft is None:
        return "LLM 조사 실패 — 아래 폼에 직접 입력하세요.", *([gr.update()] * 6), state
    state["draft"] = draft
    f = draft["fields"]
    lines = [f"{artist} — {song}"]
    for k, v in f.items():
        lines.append(f"- {k}: {v['value']} (확신도 {v['confidence']:.2f}, {SCOPE_KO.get(v['scope'], v['scope'])}) {v['source'] or ''}")
    if draft.get("problems"):
        lines.append("스키마 검증 문제: " + "; ".join(draft["problems"]))
    vals = [f[k]["value"] for k in ("guitar.pickup", "amp.family", "amp.gain", "cab.size", "boost")]
    return "\n".join(lines), *vals, f["amp.model"]["value"], state


def save_session(target, same_rig, pickup, family, gain, cab, boost, model, state):
    from ampcap import research_llm as rl

    overrides = {"guitar.pickup": pickup, "amp.family": family, "amp.gain": gain,
                 "cab.size": cab, "boost": boost, "amp.model": model or "unknown"}
    if state.get("draft"):
        # 화면에서 확인·승인한 값 → 확신도 1.0
        s = rl.draft_to_session(state["draft"], overrides=overrides, capture_target=target,
                                same_rig=same_rig)
    else:
        s = sess_mod.validate({"capture_target": target, "same_rig": same_rig,
                               "guitar": {"pickup": pickup},
                               "amp": {"family": family, "gain": gain, "model": model},
                               "cab": {"size": cab}, "boost": boost})
    prior = sess_mod.compute_prior(s)
    rd = Path(state["run_dir"])
    (rd / "session.json").write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    (rd / "prior.json").write_text(json.dumps(prior.to_dict(), indent=2), encoding="utf-8")
    state["session"] = s
    msg = "세션 저장됨." + (f" 경고: {prior.warnings}" if prior.warnings else "")
    return msg, fig_theta(torch.tensor(prior.center, dtype=torch.float32)), state


def find_tone(steps, state):
    """⑥ 톤 찾기. 제너레이터로 손실 곡선을 실시간 표시한다."""
    from ampcap.optimize import FitConfig, fit, save_result

    if not state.get("ok", False):
        raise ValueError("구간 점검을 통과해야 합니다")
    rd = Path(state["run_dir"])
    s = state.get("session") or sess_mod.unknown_session()
    prior = sess_mod.compute_prior(s)
    stem = separate.load_audio(rd / "stem.wav", mono=True)
    pool = torch.from_numpy(load_di_pool(ARGS.egdb, split_file=ARGS.split_file)).to(DEVICE)
    theta_hat = None
    from ampcap import encoder as enc

    model = enc.try_load(ARGS.encoder, DEVICE)
    if model is not None:
        theta_hat = enc.predict_song(model, stem, sess_mod.session_vector(s))

    curve = []
    import queue
    import threading

    q: queue.Queue = queue.Queue()
    box = {}

    def run():
        try:
            box["res"] = fit(torch.from_numpy(stem).to(DEVICE), pool, prior, theta_hat=theta_hat,
                             cfg=FitConfig(steps=int(steps)), progress=lambda st, l: q.put(l))
        except Exception as e:  # noqa: BLE001
            box["err"] = e
        q.put(None)

    th = threading.Thread(target=run)
    th.start()
    while True:
        item = q.get()
        if item is None:
            break
        curve.append(item)
        if len(curve) % 20 == 0:
            yield fig_curve(curve), None, None, None, "최적화 중...", state
    th.join()
    if "err" in box:
        raise box["err"]
    res = box["res"]
    save_result(res, rd, di_pool=pool.cpu(), target_rms=float(np.sqrt(np.mean(stem ** 2))))
    cands = [str(rd / f"candidate_{k}.wav") for k in range(len(res.candidates))]
    diag = "\n".join(res.diagnostics["messages"]) or "대역별 잔차 모두 2 dB 미만"
    state["n_candidates"] = len(cands)
    yield (fig_curve(res.curve), fig_theta(res.theta_star), cands[0] if cands else None,
           cands[1] if len(cands) > 1 else None, f"θ* 손실 {float(res.losses.min()):.4f}\n{diag}", state)


def make_nam(rank, epochs, state):
    import gradio as gr

    from ampcap.distill import distill
    from ampcap.optimize import load_theta

    rank = int(rank or 0)
    rd = Path(state["run_dir"])
    n_cand = len(json.loads((rd / "theta_candidates.json").read_text(encoding="utf-8"))["candidates"])
    if not 0 <= rank < max(n_cand, 1):
        raise gr.Error(f"후보 번호는 0~{max(n_cand, 1) - 1} 사이여야 합니다.")
    theta = load_theta(rd / "theta_candidates.json", None if rank == 0 else rank)
    reader = di_reader_for_split(ARGS.egdb, "test", ARGS.split_file)
    out = distill(theta, rd, ARGS.nam_input, DI_REF_RMS, session=state.get("session"),
                  name=state["song"], epochs=int(epochs),
                  verify_di=list(normalize_di(reader.fixed_clips(4, 123))), device=DEVICE)
    msg = json.dumps({k: out[k] for k in out if k != "nam_path"}, ensure_ascii=False, indent=1)
    return out["nam_path"], msg


def load_run(run_dir, state):
    rd = Path(run_dir)
    state = {"run_dir": str(rd), "song": rd.name.split("_", 1)[-1], "recommended": [], "segments": [],
             "ok": (rd / "stem.wav").exists()}
    if (rd / "session.json").exists():
        state["session"] = json.loads((rd / "session.json").read_text(encoding="utf-8"))
    msg = [f"불러옴: {rd}"] + [p.name for p in sorted(rd.iterdir())]
    theta_fig = None
    if (rd / "theta_candidates.json").exists():
        from ampcap.optimize import load_theta

        theta_fig = fig_theta(load_theta(rd / "theta_candidates.json"))
    return "\n".join(msg), theta_fig, state


# ---------------------------------------------------------------- UI


def build():
    import gradio as gr

    with gr.Blocks(title="song2nam") as demo:
        state = gr.State({})
        gr.Markdown("# song2nam — 음원에서 NAM 캡처 만들기")
        with gr.Accordion("실행 폴더 불러오기 (미리 만든 결과)", open=False):
            run_in = gr.Textbox(label="runs/<날짜_곡명>")
            load_btn = gr.Button("불러오기")
            load_msg = gr.Textbox(label="결과", lines=6)

        with gr.Tab("1. 업로드·분리 (①)"):
            audio = gr.Audio(type="filepath", label="음원 파일")
            song_name = gr.Textbox(label="곡 이름(선택)")
            sep_btn = gr.Button("기타 분리 (Demucs htdemucs_6s)")
            wave = gr.Plot(label="스템 파형과 추천 구간")
            stem_audio = gr.Audio(label="기타 스템", type="filepath")
            rec_txt = gr.Textbox(label="추천 구간 [[시작, 끝], ...]")
            sep_msg = gr.Markdown()

        with gr.Tab("2. 구간 선택 (①)"):
            target = gr.Radio(["rhythm", "lead"], value="rhythm", label="capture_target")
            same_rig = gr.Checkbox(True, label="same_rig (모든 기타 트랙이 같은 기타·앰프 구성)")
            with gr.Row():
                seg_s = gr.Number(label="시작(초)")
                seg_e = gr.Number(label="끝(초)")
            with gr.Row():
                prev_btn = gr.Button("미리듣기")
                add_btn = gr.Button("구간 추가")
                rec_btn = gr.Button("추천 구간 모두 사용")
            prev_audio = gr.Audio(label="미리듣기")
            wave2 = gr.Plot()
            seg_txt = gr.Textbox(label="선택한 구간")
            chk_btn = gr.Button("점검하고 저장")
            chk_msg = gr.Textbox(label="점검 결과", lines=4)

        with gr.Tab("3. 세션 정보 (②)"):
            link = gr.Textbox(label="곡 링크 또는 '아티스트 - 곡명' (LLM 조사, 선택)")
            llm_btn = gr.Button("LLM 장비 조사")
            llm_msg = gr.Textbox(label="조사 초안 (출처·확신도)", lines=8)
            with gr.Row():
                pickup = gr.Dropdown(sess_mod.PICKUPS + ["unknown"], value="unknown", label="픽업")
                family = gr.Dropdown(sess_mod.FAMILIES + ["unknown"], value="unknown", label="앰프 계열")
                gain = gr.Dropdown(sess_mod.GAINS + ["unknown"], value="unknown", label="게인")
            with gr.Row():
                cab = gr.Dropdown(sess_mod.CABS + ["unknown"], value="unknown", label="캐비닛")
                boost = gr.Dropdown(sess_mod.BOOSTS + ["unknown"], value="unknown", label="부스트")
                model = gr.Textbox(label="앰프 모델(자유 텍스트)")
            sess_btn = gr.Button("확인·저장")
            sess_msg = gr.Markdown()
            prior_fig = gr.Plot(label="사전 범위 중심 θ")

        with gr.Tab("4. 톤 찾기 (⑥)"):
            steps = gr.Slider(100, 2000, value=800, step=50, label="최적화 스텝")
            fit_btn = gr.Button("톤 찾기 (⑥)")
            curve = gr.Plot(label="손실 곡선")
            fit_msg = gr.Textbox(label="진단", lines=4)
            with gr.Row():
                cand_a = gr.Audio(label="후보 A (θ*)")
                cand_b = gr.Audio(label="후보 B")
            theta_fig = gr.Plot(label="θ: 앰프 노브와 블록별 주파수 응답")

        with gr.Tab("5. .nam 생성 (⑦)"):
            rank = gr.Number(0, label="후보 번호 (0 = θ*)", precision=0, minimum=0)
            epochs = gr.Slider(10, 300, value=100, step=10, label="NAM 학습 epoch")
            nam_btn = gr.Button(".nam 생성 (⑦)")
            nam_file = gr.File(label="model.nam")
            nam_msg = gr.Textbox(label="입력 레벨·검증(ESR)", lines=8)
            gr.Markdown("생성된 .nam에는 캐비닛이 포함됩니다. NAM 플러그인의 IR 로더를 끄고, "
                        "DI 트랙의 연주 RMS를 input_level.json의 di_ref_rms_dbfs에 맞추세요.")

        load_btn.click(load_run, [run_in, state], [load_msg, theta_fig, state])
        sep_btn.click(do_separate, [audio, song_name, state], [wave, stem_audio, rec_txt, state, sep_msg])
        prev_btn.click(preview_segment, [seg_s, seg_e, state], prev_audio)
        add_btn.click(add_segment, [seg_s, seg_e, state], [wave2, seg_txt, state])
        rec_btn.click(use_recommended, [state], [wave2, seg_txt, state])
        chk_btn.click(check_segments, [target, same_rig, state], [chk_msg, state])
        llm_btn.click(llm_research, [link, state], [llm_msg, pickup, family, gain, cab, boost, model, state])
        sess_btn.click(save_session, [target, same_rig, pickup, family, gain, cab, boost, model, state],
                       [sess_msg, prior_fig, state])
        fit_btn.click(find_tone, [steps, state], [curve, theta_fig, cand_a, cand_b, fit_msg, state])
        nam_btn.click(make_nam, [rank, epochs, state], [nam_file, nam_msg])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--egdb", default=ARGS.egdb)
    ap.add_argument("--nam-input", default=ARGS.nam_input)
    ap.add_argument("--encoder", default=ARGS.encoder)
    ap.add_argument("--split-file", default=ARGS.split_file)
    a = ap.parse_args()
    ARGS.egdb, ARGS.nam_input, ARGS.encoder, ARGS.split_file = a.egdb, a.nam_input, a.encoder, a.split_file
    build().queue().launch(share=a.share)

