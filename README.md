# song2nam (ampcap)

상업 음원 하나와 기타 세션 정보만으로 [Neural Amp Modeler](https://neural-amp-modeler.readthedocs.io)에서 바로 쓸 수 있는 `.nam` 캡처를 만드는 프로젝트입니다.

일반 NAM 캡처는 DI와 앰프 출력을 맞춘 페어 데이터로 학습합니다. 이 프로젝트는 DI 없이 음원의 기타 스템 통계만으로 미분 가능한 그레이박스 앰프의 설정을 찾고, 그 앰프로 NAM 학습 입력 신호(NAM 문서의 `input.wav`, v3.0.0)를 리앰핑해 `.nam`을 증류합니다.

용어는 설계 문서 13장 용어집을 따릅니다.

자세한 설계는 [docs/2026-11-09-ampcap-design.md](docs/2026-11-09-ampcap-design.md)를 보세요.

## 파이프라인

```
음원 → ① 분리·구간 선택 → ② 세션 처리 → ⑤ 인코더 θ̂ → ⑥ 곡별 파라미터 최적화 θ* → ⑦ .nam 증류
```

## 구조

```
ampcap/          파이프라인 모듈
  graybox/       ③ 미분 가능한 그레이박스 앰프 G
app.py           Gradio 데모
docs/            설계 문서
notebooks/       실험 노트북
tests/           테스트
runs/            실행 기록 (runs/<날짜_곡명>/)
data/            데이터셋 (저장소에 포함하지 않음)
```

## 설치

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 데이터

EGDB DI와 NAM 학습 입력 신호는 공식 Google Drive에서 직접 받습니다. 다시 실행하면 이어 받습니다.

```bash
python -m ampcap.get_data --data data
```

240곡 전체 목록은 Colab에서만 받을 수 있습니다(Drive API, Google 로그인). 다른 환경에서는 50곡까지 받고, 나머지는 [EGDB 페이지](https://ss12f32v.github.io/Guitar-Transcription/)의 Drive 링크에서 받으라고 안내합니다. 데이터셋과 음원, 분리된 스템은 저작권과 용량 문제로 저장소에 포함하지 않습니다.

## 테스트

```bash
pytest tests/
```

## 실행

```bash
# 음원 → .nam 한 번에
python -m ampcap.pipeline song.wav --session session.json --egdb data/EGDB/audio_DI --nam-input data/nam/v3_0_0.wav

# Gradio 데모
python app.py
```

## Colab

학습과 실험은 Colab GPU에서 돌립니다. `notebooks/`의 노트북은 Drive의 `MyDrive/song2nam/`에 `data/`, `runs/`를 두고 씁니다.

| 노트북 | 내용 |
|---|---|
| `00_setup_and_tests.ipynb` | 설치, `pytest tests/`, ⑦ ESR 테스트, EGDB 분할, 대응표 청취 |
| `01_train_encoder.ipynb` | ④ 합성 데이터로 ⑤ 인코더 학습 (체크포인트 이어 학습) |
| `02_song_to_nam.ipynb` | 음원 → .nam 전체 경로, (b) 세션만 경로, Gradio 앱(`--share`) |
| `03_experiment1.ipynb` | ⑨ 실험 1 파라미터 복원 실험 |

필요한 파일: `data/EGDB/audio_DI/*.wav`(EGDB DI), `data/nam/v3_0_0.wav`(NAM 학습 입력 신호 `input.wav` v3.0.0을 이 이름으로 저장). 둘 다 `00_setup_and_tests.ipynb`의 0번 셀(`ampcap.get_data`)이 받습니다. ②-0 LLM 조사는 `ANTHROPIC_API_KEY`가 필요합니다.
