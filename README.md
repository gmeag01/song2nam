# song2nam (ampcap)

상업 음원 하나와 기타 세션 정보만으로 [Neural Amp Modeler](https://neural-amp-modeler.readthedocs.io)에서 바로 쓸 수 있는 `.nam` 캡처를 만드는 프로젝트입니다.

일반 NAM 캡처는 DI와 앰프 출력을 맞춘 페어 데이터로 학습합니다. 이 프로젝트는 DI 없이 음원의 기타 스템 통계만으로 미분 가능한 그레이박스 앰프의 설정을 찾고, 그 앰프로 NAM 표준 입력을 재앰핑해 `.nam`을 증류합니다.

자세한 설계는 [docs/2026-11-09-ampcap-design.md](docs/2026-11-09-ampcap-design.md)를 보세요.

## 파이프라인

```
음원 → ① 분리·구간 선택 → ② 세션 처리 → ⑤ 인코더 θ̂ → ⑥ 통계 손실 미세조정 θ* → ⑦ .nam 증류
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

EGDB 데이터셋을 받아 `data/` 아래에 둡니다. 데이터셋과 음원, 분리된 스템은 저작권과 용량 문제로 저장소에 포함하지 않습니다.

## 테스트

```bash
pytest tests/
```

## 실행

```bash
python app.py
```
