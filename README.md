# song2nam (ampcap)

상업 음원 하나와 기타 세션 정보만으로 [Neural Amp Modeler](https://neural-amp-modeler.readthedocs.io)에서 바로 쓸 수 있는 `.nam` 캡처를 만드는 프로젝트입니다.

일반 NAM 캡처는 DI와 앰프 출력을 맞춘 페어 데이터로 학습합니다. 이 프로젝트는 DI 없이 음원의 기타 스템 통계와 세션 정보로 미분 가능한 그레이박스 앰프 G의 파라미터 θ를 찾고, 그 앰프로 NAM 학습 입력 신호(NAM 문서의 `input.wav`, v3.0.0)를 리앰핑해 `.nam`을 증류합니다.

자세한 설계와 용어(13장 용어집)는 [docs/2026-11-09-ampcap-design.md](docs/2026-11-09-ampcap-design.md)를 보세요.

## 파이프라인

```
음원 → 분리·구간 선택 → 세션 처리 → 인코더 θ̂ → 곡별 파라미터 최적화 θ* → .nam 증류
```

## 구조

```
ampcap/          파이프라인 모듈
  graybox/       미분 가능한 그레이박스 앰프 G
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

EGDB 240곡의 파일 ID와 MD5는 `ampcap/egdb_manifest.json`에 있어 Colab, Kaggle, 로컬 어디서나 같은 파일을 받고 해시로 확인합니다. 실패한 파일은 [EGDB 페이지](https://ss12f32v.github.io/Guitar-Transcription/)의 Drive 링크에서 받으라고 안내합니다. 데이터셋과 음원, 분리된 스템은 저작권과 용량 문제로 저장소에 포함하지 않습니다.

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

음원으로 `.nam` 만들기: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gmeag01/song2nam/blob/main/notebooks/02_song_to_nam.ipynb)

셀을 위에서부터 차례로 실행하고, 폼에서 세션 정보를 고르면 됩니다. 세션 정보 각 항목의 뜻과 고르는 법은 [docs/session-guide.md](docs/session-guide.md)에 있습니다.

학습과 실험은 Colab GPU에서 돌립니다. `notebooks/`의 노트북은 Drive의 `MyDrive/song2nam/`에 `data/`, `runs/`를 두고 씁니다.

| 노트북 | 내용 |
|---|---|
| `00_setup_and_tests.ipynb` | 설치, `pytest tests/`, 증류 ESR 테스트, EGDB 분할, 대응표 대표 설정 청취 |
| `01_train_encoder.ipynb` | 합성 데이터로 인코더 학습 (체크포인트 이어 학습) |
| `02_song_to_nam.ipynb` | 음원 → .nam (폼으로 세션 정보 입력, 데이터 내려받기 포함) |
| `03_experiment1.ipynb` | 실험 1 (파라미터 복원 실험) |

필요한 파일: `data/EGDB/audio_DI/*.wav`(EGDB DI), `data/nam/v3_0_0.wav`(NAM 학습 입력 신호 `input.wav` v3.0.0을 이 이름으로 저장). 둘 다 `00_setup_and_tests.ipynb`의 0번 셀(`ampcap.get_data`)이 받습니다. LLM 장비 조사는 `ANTHROPIC_API_KEY`가 필요합니다.
