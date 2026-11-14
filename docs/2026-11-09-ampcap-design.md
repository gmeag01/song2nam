# 음원 기반 NAM 캡처 생성기 (ampcap) 설계 문서

- 작성일: 2026-11-09
- 제출 기한: 2026-11-20 (학술제 발표와 시연)
- 개발 인원: 1명
- 컴퓨팅: Colab/Kaggle 무료 GPU(학습), 로컬 RTX 4060 Ti(시연, 보조 학습)

## 1. 목적

리앰핑용 학습 신호를 녹음하지 않고, **상업 음원 하나와 기타 세션 정보만으로** NAM에서 바로 쓸 수 있는 `.nam` 파일을 만든다.

기존 NAM 캡처는 같은 신호의 DI와 앰프 출력을 샘플 단위로 맞춘 페어로 학습한다. 음원에는 앰프 출력만 있고 DI가 없으므로, 이 프로젝트는 비페어(unpaired) 조건에서 톤을 추정하는 문제를 푼다.

### 1.1 결과물

1. 발표 자료
2. 시연: Gradio 앱으로 ⑥ 곡별 파라미터 최적화(앱 화면 "톤 찾기") 과정을 보여 주고, 미리 녹음한 DI 리프에 생성된 `.nam`을 DAW에서 적용해 원곡 스템과 비교
3. GitHub 저장소: 코드, 실험 노트북, 시연용 실행 기록

보고서는 보류한다.

### 1.2 기존 연구 대비 차별점

| | 입력 | 페어 데이터 | 결과물 | 세션 정보 사용 | 해석 가능한 파라미터 |
|---|---|---|---|---|---|
| NAM 리앰핑 캡처 (DI–앰프 출력 페어) | 실제 앰프 리앰핑 | 필요 | .nam | 아니요 | 아니요 |
| Wright 외 (ICASSP 2023) | 녹음된 기타 톤 | 불필요 (GAN) | 전용 신경망 | 아니요 | 아니요 |
| Chen 외 (DAFx 2024) | 녹음된 기타 톤 + 정렬 안 된 클린 | 불필요 (GAN) | 전용 신경망 | 아니요 | 아니요 |
| DDSP Guitar Amp (preprint) | 실제 앰프 페어 | 필요 | DDSP 기반 앰프 모델 | 아니요 | 예 |
| DeepAFx-ST, ST-ITO | 레퍼런스 오디오 | 불필요 | 이펙트 파라미터 | 아니요 | 예 |
| BIAS X Music-to-Tone (상용) | 곡 또는 기타 트랙 | 공개 안 됨 | 자사 프리셋 | 공개 안 됨 | 프리셋 노브 |
| 이 프로젝트 | 상업 음원(믹스) | 불필요 | .nam (NAM 기본 형식) | 예 | 예 |

발표 표현: 앰프 모델링, 그레이박스 DSP, 비페어 톤 학습은 각각 연구되어 왔고, 이 프로젝트는 이를 결합해 상업 음원과 세션 정보만으로 NAM에서 바로 쓸 수 있는 캡처(.nam)를 만드는 파이프라인을 구현했다. "최초"라는 표현은 쓰지 않고 "확인한 범위에서는"으로 한정한다.

## 2. 범위

### 2.1 필수

| 번호 | 모듈 |
|---|---|
| ① | 분리와 구간 선택 |
| ② | 세션 처리와 대응표 |
| ③ | 미분 가능한 그레이박스 앰프 G (직접 구현) |
| ④ | 합성 데이터 생성기 |
| ⑤ | 파라미터 인코더 |
| ⑥ | 곡별 파라미터 최적화 (통계 매칭 손실) |
| ⑦ | .nam 증류 |
| ⑨ | 평가 |
| | Gradio 데모 앱 |

⑧은 쓰지 않는 번호다. 모듈 번호를 바꾸지 않으려고 비워 둔다.

### 2.2 시간이 되면 (우선순위 순)

1. ②-0 LLM 장비 조사
2. (b) 세션만 경로: ⑤⑥ 없이 사전 범위 중심을 θ로 쓰는 비교 기준과 비교 실험
3. (a) 후보 화면: 후보 2개(A = θ*, B) 제시 (⑥의 시작점 여러 개 결과를 그대로 사용)
4. ⑥에서 EGDB 클립을 연주 종류(코드/단음)로 고르는 기능

### 2.3 범위 밖 (발표의 향후 계획)

- 후속 기능: 페달 회로 설계
- 스테레오 위치 기반 기타 분리, 상용 리드/리듬 분리 API
- 추가 자동 점검 항목(구간 사이 톤 일치, 피킹 세기 분포)
- 공개 캡처를 검색해 시작점으로 쓰는 비교 기준
- GAN 기반 비페어 학습, 조건부 신경망 앰프
- 재분리 증강 (반주와 섞은 뒤 Demucs로 다시 분리)
- IR 라이브러리 기반 캐비닛

## 3. 전체 구조

```
[음원 파일] ─▶ ① 분리·구간 선택 ─▶ 기타 스템 구간 ─────────────┐
[곡 링크] ─▶ ②-0 LLM 장비 조사(선택) ─▶ 사용자 확인 ─┐          │
                                                    ▼          ▼
                          [세션 JSON] ─▶ ② 세션 처리 ─▶ 사전 범위 + 세션 벡터
                                                    │          │
                                                    ▼          ▼
                                          ⑤ 인코더 ─▶ θ̂ ─▶ ⑥ 곡별 파라미터 최적화 ─▶ θ*, 후보들
                                                                         │
                                                                         ▼
                                                     ⑦ 증류: G(·; θ*) ─▶ model.nam

학습 단계(한 번): ③ G + ④ 합성 데이터 ─▶ ⑤ 인코더 학습
평가: ⑨
```

모듈끼리는 파일(wav, json, 체크포인트, .nam)로만 주고받아서 각각 따로 테스트할 수 있게 한다.

핵심 개념은 두 단계다. ⑥에서는 원곡 DI 없이 통계만 비교해 G의 설정 θ*를 찾는다. ⑦에서는 θ*를 넣은 G를 교사 모델로 삼아 NAM 학습 입력 신호를 리앰핑하고, 샘플 단위로 정렬된 이 입력–출력 페어로 NAM을 학습한다(증류). NAM은 G(θ*)를 흉내 내므로 최종 품질의 상한은 G의 표현력과 ⑥의 정확도로 정해진다.

## 4. 모듈 설계

### 4.1 ① 분리와 구간 선택

**처리 순서**

1. 음원을 Demucs `htdemucs_6s`로 분리해 스테레오 기타 스템을 얻는다. 리드와 리듬 기타는 하나의 스템에 함께 들어간다.
2. 짧은 구간 단위로 좌우 상관도와 Side/Mid 에너지 비를 계산해, 단일 기타 점수가 높은 구간을 추천 구간(`recommend_segments`)으로 파형 위에 표시한다. 판정은 참고용이다.
3. `capture_target`(rhythm/lead)에 맞는 안내 문구를 보여 준다.
4. 사용자가 미리 들어 보며 시작·끝 시간으로 구간을 여러 개 고른다.
5. 자동 점검: 유효 연주 길이(무음 제외, `active_sec`, 기본 15초 이상), 무음 프레임 제거, 클리핑 경고, 여러 기타 혼입 경고(`multi_guitar_suspected`). `same_rig`가 true이면 경고 대신 참고 문구를 띄운다(진행은 막지 않음).
6. 짧은 크로스페이드로 이어 붙여 48kHz 모노로 저장한다.

**좋은 구간의 조건**: 기타가 한 대만 나옴, 다른 악기가 적음, 구간 안에서 톤이 같음, 피킹 세기가 다양함.

**전체 스템 대체(fallback)**: 단일 기타 구간이 없으면 전체 스템을 쓰고 결과에 "여러 기타가 섞인 평균 톤"이라고 표시한다.

**출력**: `stem.wav`, `segments.json`. `stem_full.wav`(Demucs 스테레오 기타 스템. 같은 실행 폴더에서 다시 돌릴 때 분리를 건너뛰는 데 씀)도 늘 남는다

```json
{"song": "...", "segments": [[42.0, 58.5], [131.2, 140.0]],
 "active_sec": 22.4, "warnings": ["multi_guitar_suspected"]}
```

경고 코드: `multi_guitar_suspected`(여러 기타 혼입, `same_rig`가 true이면 `:same_rig`가 붙음), `clipping`, `full_stem_used:여러 기타가 섞인 평균 톤`(전체 스템 대체. 이 경고만 한국어 설명이 붙는다).

### 4.2 ②-0 LLM 장비 조사 (선택)

1. 곡 링크의 제목 정보에서 아티스트와 곡명을 뽑고 사용자가 확인한다. 링크에서 오디오를 내려받지 않는다. 분석할 오디오는 사용자가 파일로 올린다.
2. LLM과 웹 검색으로 두 단계 조사: 곡 단위 장비(scope `song`), 아티스트 평소 장비(scope `usual`). 곡 단위 정보에 출처가 있으면 우선한다. 평소 장비로 채운 필드는 확신도를 낮게(예: 0.5) 매긴다.
3. 출력은 4.3의 스키마와 같은 선택지 값만 허용하고, 필드마다 출처 URL과 확신도를 받는다. 출처가 없는 필드는 `unknown`.
4. 사용자가 출처와 함께 초안을 보고 수정·승인한다. 확정한 필드의 확신도는 1.0으로 바꾼다.

**환각 대응**: 출처 없는 필드는 `unknown`, 선택지 밖 값은 검증에서 거부, 사용자 확인 필수.

**오류 처리**: API 실패나 시간 초과 시 직접 입력 폼으로 전환. 스키마에 안 맞으면 한 번 재요청하고, 또 실패하면 맞는 필드만 채운다.

사용할 LLM API는 구현 계획 단계에서 정한다.

### 4.3 ② 세션 처리와 대응표

**입력 스키마**

```json
{
  "artist": "...", "song": "...",
  "capture_target": "rhythm",
  "same_rig": true,
  "guitar": {"pickup": "humbucker"},
  "amp":    {"family": "british", "model": "JCM800", "gain": "high"},
  "cab":    {"size": "4x12"},
  "boost":  "tube_screamer",
  "confidence": {"amp": 1.0, "boost": 0.5}
}
```

| 필드 | 값 | 영향을 주는 θ |
|---|---|---|
| `capture_target` | rhythm / lead | ① 안내 문구 |
| `same_rig` | true / false | ① 여러 기타 혼입 경고를 참고 문구로 낮출지 |
| `guitar.pickup` | single / p90 / humbucker / unknown | 입력 게인 |
| `amp.family` | fender / vox / british / american_hg / modern_hg / unknown | 입력 하이패스, 프리앰프 게인·바이어스, 톤스택, 파워앰프 |
|  | fender: Fender 계열 클린, vox: Vox AC 계열, british: Marshall·Orange 계열, american_hg: Mesa·5150·Soldano 계열 미국 하이게인, modern_hg: Diezel·ENGL·Friedman 계열 모던 하이게인 | |
| `amp.gain` | low / mid / high / unknown | 프리앰프 게인 범위 |
| `amp.model` | 자유 텍스트 | 기록용. 알려진 모델명은 family로 변환 |
| `cab.size` | 1x12 / 2x12 / 4x12 / unknown | 캐비닛 블록 |
| `boost` | none / tube_screamer / unknown | 부스트 블록 (none이면 드라이브 0 고정). tube_screamer는 Tube Screamer 계열 오버드라이브. 예전 값 `ts`도 읽는다 |
| `confidence` | 필드별 0~1, 기본 1.0 | 범위를 좁히는 정도 |

**대응표**: 세션 값별로 θ의 어느 차원을 어느 범위로 좁히는지 적은 표. 앰프 계열 5개, 게인 3단계, 캐비닛 3종, 픽업 3종, 부스트 2종. 지금 값은 회로·측정 자료로 다시 검증한 v2(2026-11-12)다. 근거가 약하거나 확인할 수 없는 칸이 남아 있다. v2는 모든 앰프 계열에서 `ts_mid_db` 상한을 0dB 이하로 두고(수동 톤스택은 미드를 올리지 못한다. θ 범위 ±12dB는 그대로), 입력 하이패스(`in_hp_hz`)를 픽업이 아니라 앰프 계열에서 정한다. 대응표 값이나 사전 범위 규칙을 바꾸면 대응표 버전(`MAPPING_VERSION`, 현재 2)을 올린다.

**사전 범위 계산**

사전 범위는 확률분포가 아니라 차원별 [하한, 상한]과 중심값이다. ⑥에서는 사전 범위를 벗어난 거리의 제곱에 벌점을 주고(λ_prior 항), ④에서만 이 범위 안에서 θ를 균등하게 뽑는다(균등 사전분포). 결과는 `prior.json`에 대응표 버전(`mapping_version`)과 함께 저장한다. 범위 충돌 경고 코드는 `range_conflict:<θ 이름>:<세션 값들>`.

1. 24차원 모두 [0, 1]에서 시작한다.
2. 입력된 세션 값마다 대응표 범위를 적용한다. 여러 세션 값이 같은 차원을 건드리면 교집합을 쓴다. 교집합이 비면 게인 우선 규칙으로 `amp.gain` 범위를 쓰고(예: fender×high는 Fender 드라이브 채널이나 크랭크한 트위드로 본다), `amp.gain`이 없을 때만 합집합으로 넓힌다. 어느 경우든 경고를 남긴다.
3. confidence가 1 미만이면 그만큼 전체 범위 쪽으로 넓힌다.
4. 차원마다 [하한, 상한]과 중심값을 출력한다.

**세션 벡터**(세션 조건 벡터): 필드별 `unknown` 포함 원-핫을 이어 붙인 21차원(픽업 4, 앰프 계열 6, 게인 4, 캐비닛 4, 부스트 3)에 confidence를 곱한다. 단 `unknown` 칸은 confidence와 관계없이 1이다. 자유 텍스트는 모델에 넣지 않는다.

### 4.4 ③ 그레이박스 앰프 G(x; θ)

**규칙**: 48kHz 모노, PyTorch로 직접 구현, 배치 처리 지원. θ는 0~1 정규화 벡터(24차원)이고 블록 안에서 실제 단위로 바꾼다(주파수는 로그, 게인은 dB 선형).

| 블록 | 구성 | 파라미터 | 개수 |
|---|---|---|---|
| 입력단 | 하이패스 → 게인 | 하이패스 주파수(40~200Hz), 입력단 게인(−12~+12dB) | 2 |
| 부스트(오버드라이브) | 피킹 EQ → 드라이브 게인 → tanh → 레벨 | 미드 중심(500~1500Hz), 미드 게인(0~12dB), 드라이브(0~30dB), 출력 레벨 | 4 |
| 프리앰프(게인 스테이지 2단) | 단마다 게인 → DC 바이어스 → tanh → DC 제거 → 로우패스 | 단마다 게인(0~40dB), 클리핑 비대칭(DC 바이어스), 로우패스(2~12kHz) | 6 |
| 톤스택 | 로우셸프 → 피킹 → 하이셸프 | 저역 게인, 미드 주파수, 미드 게인, 고역 게인(각 ±12dB) | 4 |
| 파워앰프 | 드라이브 게인 → tanh → 새그 → 하이셸프(프레즌스) | 드라이브(0~20dB), 새그 양, 프레즌스 | 3 |
| 캐비닛 | 하이패스 → 피킹(공진) → 로우패스 → 피킹(음수 게인, 딥) | 하이패스, 공진 피크, 로우패스(3~7kHz), 딥 주파수, 딥 깊이 | 5 |

출력 음량은 θ에 넣지 않고 스템 RMS에 맞춰 정규화한다.

**구현 결정**

1. biquad 계수는 오디오 EQ 쿡북(RBJ) 공식을 쓴다.
2. 학습 중 필터는 주파수 샘플링 방식의 FIR 근사(frequency sampling: 계수로 주파수 응답을 구해 FFT로 곱하기)로 계산한다. 순환 컨볼루션을 피하려고 FFT 길이를 충분히 크게 잡고 패딩한다.
3. 최종 렌더링(⑦)과 검증에서는 torchaudio `lfilter`(시간 영역)를 쓴다.
4. 새그는 엔벨로프 추적(지수 커널 FIR 근사) 결과로 게인을 줄인다.
5. 클리핑 블록의 2배 오버샘플링은 ⑥ 곡별 파라미터 최적화와 ⑦ 최종 렌더링에서 켠다. ⑥을 48kHz로 맞추면 하이게인에서 앨리어싱 때문에 ⑦(96kHz) 렌더링과 대역별로 수 dB 차이가 나서(2026-11-10 합성 DI 측정: 저역 +5.5dB, 6~16kHz −4.3dB) θ*가 ⑦ 렌더링에서 ⑥과 다른 주파수 응답을 낸다. ⑥은 FFT 필터를 쓰므로 2배 비용만 든다. ④ 합성 데이터와 ⑤ 인코더 학습은 속도 때문에 48kHz로 두고, 그 차이는 ⑥이 보정한다.
6. 차단 주파수와 Q의 범위는 나이퀴스트 근처와 극단적 Q를 피하도록 제한한다.

**대비책**: 2026-11-12까지 필터 테스트(7장)가 통과하지 않으면 NablAFx(MIT)의 필터 함수를 출처를 밝히고 가져온다.

### 4.5 ④ 합성 데이터 생성기

샘플 하나의 생성 과정:

1. 세션 필드 값을 무작위로 정하고 ②로 사전 범위를 만든다. 샘플의 약 20%는 θ 전체 범위에서 뽑는다.
2. 사전 범위 안에서 θ를 균등하게 뽑는다.
3. 세션에 조건 드롭아웃(필드를 일정 확률로 `unknown`으로 가림)과 confidence 증강(confidence를 무작위로 낮춤)을 적용한다.
4. EGDB DI에서 3초를 잘라 DI 레벨을 무작위로 바꾼다(DI 레벨 증강, `SynthConfig.di_gain_range_db`. θ의 입력단 게인과 별개).
5. y = G(x; θ)를 렌더링한다.
6. pedalboard로 무작위 EQ, 컴프, 짧은 리버브 증강을 건다. 이어서 분리 잔여물 근사 증강(무작위 로우패스 6~16kHz, SNR 25~45dB 노이즈, 각각 확률 0.5)을 건다(`augment_separation`, ④만. 실험 1 가짜 곡에는 걸지 않는다).

데이터는 학습 중 GPU에서 바로 만들고 저장하지 않는다.

**EGDB 분할** (곡 단위로 나누고, 서로 겹치지 않게 한다. 분할 결과는 `egdb_split.json`)

| 분할 | 용도 |
|---|---|
| train | ④ 합성 데이터 (⑤ 학습) |
| val | ⑤ 검증 |
| pool | ⑥ 곡별 파라미터 최적화에 쓰는 DI 풀(학습에 쓰지 않은 DI 모음) |
| test | ⑨ 실험 1의 가짜 곡 생성과, 정답 θ를 넣은 G(G(θ_true))와 .nam을 비교할 때 넣는 미사용 DI. ⑦ 증류 확인용 DI |

### 4.6 ⑤ 인코더 (파라미터 추정 인코더)

```
스템 3초 → log-Mel(128 밴드) → CNN 4~6층 → 시간 축 풀링(주파수 축은 펼침) → 오디오 벡터
세션 벡터(21) ─────────────────────────────────────────────┐
[오디오 벡터 ; 세션 벡터] → MLP → sigmoid → θ̂ (24)
```

- 풀링: 시간 축만 평균+최대로 줄이고 주파수 축은 펼친다(`EncoderConfig.pool="time"`). 주파수·시간 전역 풀링은 컷오프·공진 주파수의 위치를 지우기 때문이다. 대응표 v1 인코더(`encoder_v1_best.pt`)는 전역 풀링(`pool="global"`, 기본값)으로 학습했고, 기본값을 그대로 두어 계속 불러올 수 있다.
- 손실: θ MSE + MR-STFT 손실(multi-resolution STFT loss, Yamamoto 외 2020)(G(x; θ̂) vs y)
- 실제 곡: 3초 창 여러 개의 θ̂ 중앙값
- 학습: Colab GPU에서 몇 시간 이내 목표. 넘으면 층 수나 Mel 크기를 줄인다.
- 역할: ⑥의 시작점과 정칙화 기준점(λ_enc 항). 인코더가 없어도 파이프라인은 동작한다.

**체크포인트**

- 위치: `checkpoints/encoder_v<θ버전>_m<대응표 버전>_<날짜>.pt` (`.gitignore` 대상). Colab에서는 Google Drive에 저장한다.
- 내용: `state_dict`, 옵티마이저 상태와 스텝(이어서 학습용), 모델 설정(Mel 밴드 수, 층 수, 풀링), `theta_version`, `mapping_version`, 검증 손실.
- 저장 주기: 일정 스텝마다 덮어쓰고, 검증 손실이 가장 낮은 것은 `_best`로 따로 둔다. Colab 런타임이 끊기면 마지막 체크포인트에서 이어서 학습한다.
- θ 버전: 4.4의 블록 구성, 범위, G 내부 고정 상수를 바꾸면 `theta_version`을 올린다. 불러올 때 G의 버전과 다르면 오류를 내고, 인코더 없이 시작점 `prior_center`·`prior_sample_i`만으로 진행한다. 설정과 가중치 모양이 맞지 않을 때도 같다. 대응표 버전은 이름과 기록으로만 구분하고 불러올 때 강제하지 않는다(대응표 v1 인코더를 대비책으로 남긴다).
- 배포: 최종 `encoder_v<θ버전>_m<대응표 버전>_best.pt`는 저장소에 커밋해 클론만으로 받게 한다(`.gitignore` 예외, 중간 체크포인트는 제외). 수십 MB를 넘으면 Release 첨부로 바꾼다. 사용자가 .nam을 만들 때 필요한 오디오는 아래와 같다.
  - DI 풀: EGDB 클립은 저장소·Release에 넣지 않고, 데이터 내려받기 스크립트(`ampcap.get_data`)가 EGDB 공식 Google Drive에서 `audio_DI`를 직접 내려받는다(`gdown`). EGDB README에는 사용 허락("feel free to use EGDB")만 있고 재배포 조건이 없기 때문이다. 내려받은 뒤 4.5의 분할 규칙대로 pool 클립을 고른다. Drive 다운로드가 막히면 EGDB 공식 링크를 안내해 직접 받게 한다.
  - NAM 학습 입력 신호 v3.0.0(`input.wav`): 파일 자체에 라이선스 표기가 없고(NAM 코드의 MIT는 저장소 소프트웨어 대상, wav는 Google Drive 별도 배포) 재배포하지 않는다. 데이터 내려받기 스크립트가 NAM 공식 문서에 걸린 Google Drive 링크에서 직접 내려받는다(`gdown`). 막히면 공식 문서 링크를 안내한다. 버전은 NAM 학습 코드가 자동 판별한다.

### 4.7 ⑥ 곡별 파라미터 최적화 (통계 매칭 손실)

네트워크 가중치는 바꾸지 않고 곡마다 θ만 최적화한다(inference-time optimization, ST-ITO와 같은 계열). 코드 이름은 `fit`, 앱 화면 이름은 "톤 찾기".

**준비**

- 타깃: ①의 스템 구간(무음 제외)에서 통계를 한 번 계산해 둔다.
- DI 풀: EGDB pool 분할의 3초 클립 수십 개. 기본은 무작위 선택.

**통계 4종** (모두 진폭 스펙트럼 기반(위상 무시), 전체 음량 차이 제거)

1. 장기 평균 스펙트럼(LTAS 변형): 모든 프레임의 log-Mel 평균(문헌의 LTAS는 선형 파워 스펙트럼 평균). 톤스택, 캐비닛, EQ를 반영한다.
2. 레벨별 스펙트럼: 프레임을 신호 내 음량 분위수로 4구간으로 나눈 구간별 평균 스펙트럼. 왜곡은 세게 칠수록 배음이 늘고 EQ는 세기와 무관하다는 차이로 게인과 EQ를 구분한다.
3. 레벨 구간별 스펙트럼 평탄도: 왜곡의 양을 나타낸다.
4. 다이내믹스 통계: 프레임별 크레스트 팩터의 평균과 분산, 프레임 음량 분포의 폭. 다이내믹 레인지 압축 정도를 나타낸다.

**손실**

```
L(θ) = Σ wₖ · L1(통계ₖ(G(DI; θ)), 통계ₖ(스템))
     + λ_prior · (사전 범위를 벗어난 거리)²
     + λ_enc · ‖θ − θ̂‖²        (인코더가 있을 때)
     + λ_anchor · Σ_드라이브 ((θ − θ₀) / σ)²   (드라이브 앵커, θ₀ = 그 시작점)
```

각 항은 시작점의 손실값으로 나눠 정규화하고 같은 가중치로 시작한다. 가중치와 λ는 합성 데이터로 조정한다. λ_prior는 작게 두어 세션 정보가 틀려도 통계 매칭 손실이 사전 범위 밖을 가리키면 θ가 범위를 벗어날 수 있게 한다.

**게인 식별** (실험 1 v1에서 스펙트럼은 맞지만 게인 θ가 정답에서 멀어진 데 대한 대응, 2026-11-13)

- 교환 게인 고정: 사이에 선형 블록만 있는 게인 쌍은 dB 합만 출력에 남는다(`in_gain_db`↔`boost_drive_db`: 하이패스·피킹 EQ만 사이에 있음, `boost_level_db`↔`pre1_gain_db`: 바로 붙어 있음). 블록 지향 시스템 식별의 표준 정규화처럼 쌍마다 앞쪽 게인을 시작점 값에 고정한다. 뒤쪽이 고정(예: `boost=none`의 `boost_drive_db`)이면 앞쪽이 그 비선형의 유일한 드라이브이므로 풀어 둔다.
- 드라이브 앵커: 비선형 4곳의 입력 게인(`boost_drive_db`, `pre1_gain_db`, `pre2_gain_db`, `pa_drive_db`)에 시작점 쪽으로 당기는 가우시안 벌점을 건다. 사전 범위 벌점은 범위 안에서 0이라 평평한 방향으로 흘러가는 것을 막지 못한다. 기본값 λ_anchor = 0.1, σ = 0.1(정규화 단위, 드라이브 공통). σ는 인코더 파라미터별 검증 오차가 저장되면 그 값으로 바꾼다.
- 게인 식별 진단: 통계 특징의 θ 민감도 JᵀJ를 고유분해해 고유값이 작은 방향(손실이 보지 못하는 θ 조합)이 게인 블록에 몰리는지 본다(`stat_sensitivity`). 실험 1 가짜 곡 몇 개의 정답 θ에서 Colab으로 돌린다.

**최적화**

- θ = sigmoid(u)로 두고 u를 Adam으로 최적화한다.
- 매 스텝 DI 클립 8개를 렌더링해 통계를 계산한다.
- 수백~천 스텝, 손실 정체 시 종료.
- 시작점 여러 개(사전 범위 중심 `prior_center`, 사전 범위 샘플 몇 개 `prior_sample_i`, 인코더 θ̂ `encoder`)를 한 배치로 병렬 최적화한다. 마지막에 고정 평가 클립 16개로 다시 평가한다. θ*(= 후보 0, 코드 `theta_star`)는 최소 이동 선택으로 고른다: 손실이 최저의 1.02배 안에 드는 시작점 중 시작점에서 θ RMS 거리로 가장 덜 움직인 것, 같으면 `encoder`. 이 최종 비교에서는 시작점마다 따로 정규화하지 않고, 모든 시작점에 같은 기준(통계 항별로 시작점 평균 손실)을 쓴다. 시작점별 정규화로 고르면 처음부터 잘 맞던 시작점(예: 인코더 θ̂)보다 손실이 많이 준 시작점이 θ*가 되기 때문이다. 후보는 손실 순으로 고르되 서로 θ RMS 거리가 0.08보다 큰 상위 결과(최대 4개)를 (a) 후보로 쓴다. 후보끼리는 θ 거리로 중복을 제거한다.
- 손실이 NaN이 되거나 발산하면 최선 θ로 되돌리고 학습률을 낮춰 재시작한다.

**진단 출력**: 통계별 손실 항(`terms`)과 장기 평균 스펙트럼의 대역별 잔차(dB. 2dB 이상이면 "고역(6000-16000Hz) 부족 3.1 dB" 같은 문장)를 저장해 화면과 발표 자료에 쓴다.

**출력**: `theta_candidates.json`, `loss_curve.png`, `candidate_k.wav`

### 4.8 (b) 세션만 경로 (선택)

②의 사전 범위 중심값(또는 샘플 몇 개)을 θ로 쓰고 ⑤⑥을 건너뛰어 ⑦로 간다. 결과에 "음원 분석 없이 세션 정보만으로 추정"이라고 표시한다. 코드 옵션은 `--session-only`. 대응표는 ②와 공유하므로 추가 작업은 분기와 표시, 비교 실험이다.

### 4.9 ⑦ 증류

1. θ*를 G에 고정하고 `lfilter`와 2배 오버샘플링을 켠다.
2. NAM 학습 입력 신호 v3.0.0(NAM 문서의 `input.wav`, 커뮤니티 표기 `v3_0_0.wav`)을 G에 통과시켜 output.wav를 만든다. 클리핑되지 않게 최대 레벨을 맞추고, 사용한 입력 레벨(디지털 배율, dBu 아님)을 `input_level.json`에 기록한다.
3. neural-amp-modeler로 학습한다. 지연 보정(NAM `delay`)은 0. 학습 시간이 길면 더 가벼운 모델 크기(Lite / Feather / Nano)를 쓴다.
4. `.nam` 메타데이터에 이름, 세션 정보, "음원에서 생성됨"을 기록한다. NAM 표준 필드는 `name`, `modeled_by`("ampcap (음원에서 생성됨)"), `gear_type`(부스트가 있으면 `amp_pedal_cab`, 없으면 `amp_cab`), `gear_model`, `tone_type`. `input_level_dbu`/`output_level_dbu`는 하드웨어 dBu 값이라 비워 둔다. 세션, θ, 입력 레벨은 표준 밖 키 `metadata.ampcap`에 기록한다.
5. 학습에 쓰지 않은 DI로 G와 .nam을 비교한다(ESR). 오차가 크면 경고하고 더 큰 모델로 재학습할 수 있게 한다.

생성된 .nam에는 캐비닛까지 포함되므로, 사용할 때 NAM 플러그인의 IR 로더는 끈다.

### 4.10 ⑨ 평가

**지표**

| 상황 | 지표 |
|---|---|
| 정답이 있는 비교 | ESR(주 지표, 0.01 미만 목표: NAM 학습기 `_esr_comment`의 "Great!" 기준을 차용. 0.035 미만은 "Not bad!". 프로젝트 ESR은 pre-emphasis 없음), MR-STFT 거리, Mel 스펙트럼 L1 거리, θ 오차(실제 단위: dB, Hz), 식별 가능 게인 조합의 dB 합 오차 |
| 정답이 없는 비교 | 통계별 L1 거리(`stat_distance`, ⑥과 같은 통계 4종), 장기 평균 스펙트럼의 대역별 dB 차이 그래프, log-Mel 기반 FAD 대체(`fad_mel`, VGGish 임베딩 아님, 참고용), MUSHRA식 0~100 청취 평가 점수(숨은 기준·앵커는 실험 3에서 결정) |

**실험 1. 파라미터 복원 실험 (핵심)**

앰프 계열 5개 × 정답 θ 4개 = 가짜 곡 20개를 EGDB test 분할의 DI로 만들고 믹스 처리 증강을 건다. 결과는 `runs/exp1/exp1_results.jsonl`(대응표 v1), `runs/exp1_v2/exp1_results.jsonl`(대응표 v2)에 쌓는다(`runs/<날짜_곡명>/` 규약의 예외). 행마다 `mapping_version`을 기록하고, 기존 행과 대응표 버전이 다르면 이어서 실행하지 않는다(필드가 없는 행은 v1). 대응표 v2는 끝까지 지키는 부분집합(통계 4종 × 시작점 3 × 세션 3 + 세션만 기준)만 다시 돌리고, 통계 묶음 비교는 v1 결과로 보고한다. θ 오차는 바이어스 기준형(`pre1_bias ≥ 0`, 음수이면 두 바이어스를 함께 뒤집음)으로 바꾼 뒤 잰다. ⑥ 통계는 극성에 불변이라 두 형태를 구분할 수 없다. 톤스택은 노브 복원을 주장하지 않고 대역별 dB 차이로 보고한다. 행마다 θ*(`theta_star`), 스텝 수(`steps`), 통계별 원래 손실(`terms`), 시작점에서 움직인 거리(`move`), 드라이브 앵커 세기(`lambda_anchor`)도 기록한다. θ 오차는 개별 파라미터 오차와 함께 식별 가능 게인 조합(`in+boost_drive`, `boost_level+pre1`, `pre2`, `pa_drive`, 6개 합 `total`)의 dB 합 오차(`gain_sum_err`)로도 보고한다. 이 지표는 v2 실행 전에 정했다. λ_anchor는 all4·세션 3종에서 0 / 0.1 / 0.3을 결과 폴더를 나눠 돌리고 ESR과 게인 합 오차로 고른다(통계 손실로 고르지 않는다). `fit_loss`는 정규화된 값이라 조건 사이에 비교하지 않는다. 실험 1은 .nam이 아니라 G(θ*)를 G(θ_true)와 비교한다(⑦ 증류 오차는 따로 잰다). 파이프라인으로 .nam을 만들고, test 분할에서 가짜 곡에 쓰지 않은 미사용 DI를 G(θ_true)와 .nam에 각각 넣어 비교한다.

| 비교 | 조건 |
|---|---|
| 시작점 | 사전 범위 중심(`prior_center`) / 사전 범위 중심+샘플 4개(`prior_multi`, (곡, 세션)마다 시드 고정) / 인코더(`encoder` = θ̂에서 시작 + λ_enc θ̂ 정칙화. 다른 두 조건에는 λ_enc 항이 없다) |
| 통계 | 장기 평균 스펙트럼만 / + 레벨별 / 4종 모두 |
| 세션 정보 | 없음(`none`) / 정확(`correct`) / 오지정(`wrong`) |
| 기준 | (b) 세션만 경로 (선택 항목 완성 시) |

**실험 2. 실제 곡**: 세션 정보가 알려진 곡 3~5개. 통계별 L1 거리와 대역별 dB 차이, 청취 비교. ②-0을 만들었다면 LLM 조사 결과의 정확도도 확인한다.

**실험 3. 청취 평가**(MUSHRA식): 원곡 스템과 생성 .nam 출력을 번갈아 들려주고 점수를 받는다. 인원이 적으면 참고 수치로 표기한다.

### 4.11 구현 시 정한 값 (2026-11-09)

설계에 범위가 적혀 있지 않던 항목은 아래처럼 정했다. 모두 초기 가정이며 G로 렌더링해 들어 보고 조정한다. 전체 표는 `ampcap/graybox/amp.py`의 `PARAMS`, 대응표는 `ampcap/session.py`의 `MAPPING`.

| 항목 | 값 |
|---|---|
| 부스트 출력 레벨 | −12~+12dB |
| 프리앰프 클리핑 비대칭(DC 바이어스, `pre*_bias`) | −0.5~+0.5 |
| 톤스택 | 로우셸프 120Hz, 미드 250~2500Hz(Q 0.8), 하이셸프 3.2kHz |
| 프레즌스 | 하이셸프 4kHz, −12~+12dB |
| 캐비닛 | 하이패스 40~200Hz, 공진 피크 주파수 60~200Hz(+6dB, Q 2 고정), 로우패스 3~7kHz(같은 필터 2번, 4차), 딥 1~5kHz, 깊이 0~18dB(Q 4. 음수 게인 피킹이라 엄밀한 notch는 아님, 코드 `cab_notch_*`) |
| 새그 | |y|의 1차 로우패스(50ms) 엔벨로프로 게인을 1/(1+2·sag·env)로 줄임 |
| DI 기준 음량 | ④⑥⑦ 모두 DI 클립을 RMS 0.05로 맞춰 G에 넣는다(입력단 게인 θ의 기준점). ⑦은 NAM 학습 입력 신호를 이 RMS로 맞춘 배율을 입력 레벨(`input_gain_db`, 디지털 배율)로 기록 |
| 2배 오버샘플링 | ⑥ 곡별 파라미터 최적화(FFT 필터)과 ⑦ 렌더링(lfilter)에서 G 전체를 96kHz로 돌린다(클리핑 블록 포함). ④⑤는 48kHz |
| ⑤ MR-STFT 손실 | G(x; θ̂)와 증강 전 렌더 y(4.5의 5단계)를 비교 |
| ②-0 LLM API | Anthropic Claude(`claude-opus-5-5`) + 웹 검색 도구 |
| 전체 실행 스크립트 | `ampcap/pipeline.py` (`python -m ampcap.pipeline`) |
| ③ 중립 θ 통과 테스트 | 캐비닛 공진은 끌 수 없어 블록별(캐비닛 제외 전체 포함)로 진폭 스펙트럼을 비교 |
| 데이터 내려받기 스크립트 | `ampcap/get_data.py` (`python -m ampcap.get_data`). gdown 공개 폴더 목록은 50개까지라 EGDB `audio_DI`(240곡) 목록은 Colab에서 Drive API(Google 로그인)로 받고, 파일은 gdown으로 받는다. NAM 입력은 NAM 코드의 v3.0.0 MD5로 확인 |
| 인코더 체크포인트 | θ 버전은 `graybox/amp.py`의 `THETA_VERSION`(현재 1), 대응표 버전은 `session.py`의 `MAPPING_VERSION`(현재 2). 학습 중 파일은 `encoder_v1_m2_<시작 날짜>.pt`, 최고 모델은 날짜 없이 `encoder_v1_m2_best.pt`. Colab은 Drive `checkpoints/`에 저장하고, 01 노트북은 같은 θ 버전·대응표 버전의 가장 최근 파일에서 이어 학습한다. 파이프라인·앱의 기본 인코더는 저장소의 `checkpoints/encoder_v1_m2_best.pt`, 없으면 대응표 v1 인코더 `checkpoints/encoder_v1_best.pt` |

## 5. 데모

**형태**: Gradio 앱을 로컬(4060 Ti)에서 실행하고, Colab에서도 띄울 수 있게 한다.

**화면 흐름** (화면 번호는 진행 순서, 괄호 안이 모듈 번호)

1. 음원 업로드와 분리(①): 스템 파형과 추천 구간 표시
2. 구간 선택(①): 시작·끝 입력, 미리듣기, 구간 추가, 점검 결과
3. 세션 정보(②): LLM 조사 초안(출처, 확신도) 확인·수정, 또는 직접 입력
4. 톤 찾기(⑥): 손실 곡선 실시간 표시, 후보 A/B 재생, θ를 앰프 노브 그림과 블록별 주파수 응답으로 시각화
5. .nam 생성과 다운로드(⑦)

**시연 진행**

- 현장: 업로드, 구간 선택, LLM 조사, ⑥ 톤 찾기, 후보 듣기
- 미리 준비: 같은 곡으로 증류한 .nam, 원곡과 같은 리프를 Hi-Z 입력으로 녹음한 DI
- DAW 프로젝트의 트랙: 원곡 스템 / DI + 최종 .nam / DI + 다른 후보 / DI + (b) 세션만 경로 결과 / DI 원음(처리 없음). 솔로·뮤트로 전환한다.
- NAM 플러그인의 IR 로더는 끄고, DI 트랙의 연주 RMS를 ⑦이 `input_level.json`에 기록한 `di_ref_rms_dbfs`에 맞춘다.
- 시연 곡 2~3개는 모든 중간 결과를 미리 만들어 두고, 문제가 생기면 불러온다.

**실행 기록**: 실행마다 `runs/<날짜_곡명>/`에 `stem.wav`, `segments.json`, `session.json`, `theta_candidates.json`, `loss_curve.png`, `candidate_k.wav`, `model.nam`을 저장한다. 그 밖에 `stem_full.wav`(Demucs 스테레오 기타 스템), `prior.json`, `input_level.json`, `output.wav`(⑦ 렌더)가 생긴다. `runs/`에서는 분석·시각 자료에 쓰는 수치 파일만 저장소에 커밋한다: 실험 결과 `exp1_results.jsonl`, 인코더 학습 기록 `runs/encoder_*/`의 `history.json`·`status.json`, 곡별 `session.json`·`theta_candidates.json`·`distill.json`. 그 밖의 파일(오디오, `.nam`, 그림, 로그)은 `.gitignore`로 빠진다.

## 6. 오류 처리

| 상황 | 대응 |
|---|---|
| 지원하지 않는 형식, 다른 샘플레이트 | 48kHz로 자동 변환, 실패 시 안내 |
| Demucs GPU 메모리 부족 | 음원을 나눠 처리 |
| 구간이 기준보다 짧음 | 다음 단계 진행을 막고 안내 |
| LLM 실패, 시간 초과 | 직접 입력 폼으로 전환 |
| LLM 출력 스키마 불일치 | 한 번 재요청, 실패 시 맞는 필드만 채움 |
| 대응표 범위 충돌 | 게인 우선 규칙(`amp.gain` 범위, 없으면 합집합)과 경고 기록 |
| 실험 1 결과의 대응표 버전 불일치 | 이어 실행 거부, 다른 결과 폴더 안내 |
| ⑥ 손실 NaN, 발산 | 최선 θ로 복원, 학습률 낮춰 재시작 |
| ⑦ ESR이 큼 | 경고, 더 큰 모델로 재학습 선택지 제공 |

## 7. 테스트

| 모듈 | 테스트 | 통과 기준 |
|---|---|---|
| ③ 블록 | scipy `freqz`, `sosfilt`와 주파수 응답·출력 비교 | 허용 오차 이내 |
| ③ | 중립 θ(`neutral_theta`)에서 입력 통과 | 입력과 출력이 거의 같음 |
| ③ | `torch.autograd.gradcheck`(double) | 수치 미분과 일치 |
| ③ | 주파수 샘플링 방식 계산과 `lfilter` 비교 | 거의 같음 |
| ③ | 처리 속도 측정 | ④의 실시간 생성이 가능한 속도 |
| ② | 전부 unknown이면 전체 범위, 겹침·충돌 계산(게인 우선 규칙), 잘못된 값 거부, 모든 계열 `ts_mid_db` 상한 ≤ 0 | 모두 통과 |
| ① | 무음 섞인 사인파로 길이·무음 점검, 크로스페이드 접합부에서 클릭 없음 | 모두 통과 |
| ⑥ 통계 | 전체 음량 변화에 불변 | 차이 거의 0 |
| ⑥ 통계 | 게인 증가 시 세게 친 구간만 더 밝아지고, 트레블 증가 시 모든 구간이 비슷하게 밝아짐 | 성립 (실패 시 통계 설계 재검토) |
| ⑥ 최적화 | 같은 DI와 정답 θ로 렌더링한 타깃 스템에서 θ 복원 검사 | 거의 정확히 복원 |
| ⑥ 최적화 | 정답 θ 시작점과 먼 시작점을 함께 돌려 θ* 선택 공정성 확인 | 정답 θ 시작점이 θ* |
| ③ / ⑥ | ⑥ 렌더(FFT + 2배 오버샘플링)와 ⑦ 렌더(lfilter + 2배 오버샘플링) 비교(3초, 하이게인) | ESR 1e-3 미만 |
| ⑥ 최적화 | 교환 게인 고정(`boost=none`이면 `in_gain_db`는 풂), 손실이 같으면 `encoder` 선택 | 통과 |
| ⑥ / ⑨ | 게인 식별 진단에서 `in_gain_db` +1dB / `boost_drive_db` −1dB 방향의 통계 변화, 게인 합 오차가 교환에 불변 | 단독 변화의 5% 미만 / 0 |
| ⑨ | 실험 1 이어 실행 시 `prior_multi` 시작점 재현 | 같은 샘플 |
| ⑤ | 작은 배치 과적합 | 손실이 거의 0 |
| ⑤ | 대응표 v1 형식 체크포인트(전역 풀링) 불러오기, 모양 불일치 시 인코더 없이 진행 | 통과 |
| ⑨ | 대응표 버전이 다른 결과에 이어 실행 거부, 바이어스 기준형 | 통과 |
| ⑦ | 미사용 DI로 G와 .nam 비교 | ESR 0.01 미만(NAM 학습기 "Great!" 기준 차용)을 1차 목표 |
| ②-0 | 모의 응답으로 스키마 검증, 실제 곡 몇 개 수동 확인 | 통과 |
| 전체 | 음원 하나로 .nam까지 한 번에 실행 | 오류 없이 완료 |

⑥ 통계의 게인/트레블 테스트는 ⑥을 만들자마자 가장 먼저 실행한다.

## 8. 일정

| 날짜 | 작업 | 마감점 |
|---|---|---|
| 11/9(월)~10(화) | 저장소 구조, Colab·로컬 환경, Demucs 실행, 실제 DI–앰프 출력 페어로 NAM 학습 1회(학습기 동작 확인), EGDB 확보. 시연 곡 2~3개 선정과 DI 리프 녹음 | 스템 추출과 .nam 생성 확인 |
| 11/11(수)~12(목) | ③ G 구현과 테스트, ② 세션 처리와 대응표 | 11/12: 필터 테스트 미통과 시 NablAFx 필터 함수로 대체 |
| 11/13(금)~14(토) | ⑥ 통계와 최적화, ⑦ 증류, 전체 실행 스크립트 | 11/14: 음원 → .nam 경로 완성. 미완성이면 이후 일정을 이 경로 다듬기로 전환 |
| 11/15(일)~16(월) | ④ 합성 데이터, ⑤ 인코더 | 인코더 시작점이 사전 범위 시작점보다 나은지 |
| 11/17(화) | ②-0 LLM 장비 조사, 실험 1 실행 시작 | |
| 11/18(수) | 실험 결과 정리, Gradio 앱, 시간이 되면 (b)와 (a) 화면 | 비교표와 그래프 |
| 11/19(목) | 시연 준비(DAW 프로젝트, 미리 만든 결과), 발표 자료, GitHub 정리 | 리허설 1회 |
| 11/20(금) | 예비일, 제출 | |

**일정이 밀릴 때 빼는 순서**: (a) 후보 화면 → (b) 세션만 경로 → ②-0 LLM 조사 → 실험 1의 일부 비교 조건

**끝까지 지키는 것**: 음원 → .nam 경로, 실험 1의 시작점 비교와 세션 정보 유무 비교, DAW 시연

**④⑤가 늦어질 때**: 사전 범위 샘플 여러 개를 시작점으로 쓰는 ⑥으로 대체한다.

## 9. 코드 구조

```
ampcap/
  separate.py        ① 분리, 구간 점검, 추천 구간
  research_llm.py    ②-0 LLM 장비 조사
  session.py         ② 스키마 검증, 대응표, 사전 범위, 세션 벡터
  graybox/
    filters.py       biquad 계수, 주파수 샘플링 방식 필터, lfilter 래퍼
    blocks.py        입력단, 부스트, 프리앰프, 톤스택, 파워앰프, 캐비닛
    amp.py           G(x; θ), θ 분할과 단위 변환
  synth_data.py      ④
  encoder.py         ⑤
  stats.py           ⑥ 통계 4종
  optimize.py        ⑥ 곡별 파라미터 최적화, 후보 선택
  distill.py         ⑦
  get_data.py        EGDB audio_DI, NAM 학습 입력 신호 내려받기 (4.6 배포)
  evaluate.py        ⑨ 지표와 실험
  pipeline.py        음원 → .nam 전체 실행 (4.11)
app.py               Gradio 데모
notebooks/           실험과 그래프
tests/               7장의 테스트
runs/                실행 기록
checkpoints/         ⑤ 인코더 체크포인트 (최종 _best만 커밋)
```

## 10. 한계

- 결과물은 앰프 하나가 아니라 마이크, 룸, 믹스 처리까지 포함한 "녹음된 기타 톤" 캡처에 가깝다.
- 딜레이, 리버브 같은 시간 기반 효과(공간계)는 담지 않는다.
- 표현력은 G의 블록 구조에 묶인다. 퍼즈, 옥타버, 모듈레이션은 다루지 않는다.
- 정확도는 실제 앰프를 리앰핑한 캡처(DI–앰프 출력 페어)보다 낮을 것으로 예상한다. 장점은 녹음 과정 없이 만든다는 편의성이다.
- 정량 평가는 주로 합성 데이터에 기반한다.
- 대응표 v2도 근거가 약하거나 확인할 수 없는 칸이 남아 있다(5150·Rectifier·Diezel·ENGL·Friedman·Orange 등). 세션 정보의 효과는 근거가 있는 Fender/Marshall/Vox 칸과 나눠 보고한다.
- 프리앰프 게인의 실제 근거는 두 단의 합계(예: SLO 51~71dB, JCM800 45~60dB)에만 있다. 대응표는 단별 상자라 합계 제약을 표현하지 못하고, 넣지 않는다.
- `pa_sag`는 tanh 뒤의 이득 감소라 실제 sag(헤드룸 감소)보다 약 10배 깊다(sag=1 ≈ −8.3dB). 출력이 RMS 정규화되므로 θ*의 `pa_sag`는 "sag"가 아니라 믹스 버스 압축까지 포함한 "느린 압축 전체"로 읽는다.
- 픽업 공진(2~5kHz)을 담는 θ가 없다. 픽업 차이는 `pre1_lp_hz`, 셸빙, 캐비닛 딥이 흡수하고, `ts_mid_db ≤ 0`이라 Q > 1의 좁은 공진 피크는 재현하지 못한다.
- 캐비닛 공진은 +6dB / Q 2로 고정이다. 측정 근거는 2.7dB @110Hz 1건뿐이라 저역 혹을 과장할 수 있다.
- 톤스택 범위가 앰프 전체 밝기(브라이트 캡, 출력 임피던스, 스피커)를 떠안는다. 셸빙과 pre·cab 필터가 밝기를 나눠 가질 수 있어 톤 노브 값은 복원 대상으로 주장하지 않는다.
- Tube Screamer의 720Hz 병렬 하이패스(`x + (g−1)·HP720(x)`)가 없어 험프 대비가 실제(14~15dB)보다 작고(약 12dB), 저역이 클리퍼로 들어가지 않는 "타이트함"을 표현하지 못한다.
- 모델 dB 규약은 두 가정(잭 피크 0.3V, DI 크레스트 팩터 4~6)에 기댄다. 측정 크레스트 팩터는 5.3~9.2라 pre1 경계가 0~4dB 낮게 잡혔을 수 있다(불확실성 ±4dB).
- ⑥의 λ_prior 항은 시작점이 모두 사전 범위 안에 있고 λ_prior가 작아 실제로는 거의 작동하지 않는다. 세션 정보는 주로 시작점과 고정 차원(하한 = 상한)으로 작용하고, `boost=none`은 부스트 드라이브·미드·레벨을 강제로 고정한다.
- 통계로 구별되지 않는 θ 방향이 있다. 정확히 교환되는 게인 두 쌍은 고정했지만, tanh 포화로 생기는 근사 교환(하이게인에서 앞단 게인의 효과가 거의 사라짐)과 프레즌스·캐비닛 필터와 믹스 EQ는 남는다. 이 방향의 θ* 값은 해석하지 않는다.
- 드라이브 앵커와 최소 이동 선택은 시작점이 틀리면 θ*를 그 근처에 묶는다. 세션 정보 오지정 조건의 결과를 따로 본다.
- 실험 1은 낙관적인 시험이다. 가짜 곡에는 음원 분리(Demucs) 잔여물과 마스터링 리미터가 없다. 실제 곡의 오차는 이보다 클 것으로 본다.
- 위의 G 상수 수정(캐비닛 공진, TS 하이패스, sag 형태)과 `pre1_bias` 범위 변경은 θ 버전을 올려야 하므로 11/20 뒤로 미룬다.

## 11. 주요 결정 기록

| 결정 | 이유 |
|---|---|
| 그레이박스 + 인코더 접근법을 중심으로 채택 | 한 달 미만, 1인 일정에서 끝까지 동작 가능, 해석 가능, 후속 기능(페달 회로 설계)과 연결 |
| GAN 대신 통계 매칭 손실 | GAN 학습 불안정과 디버깅 부담 |
| 스테레오 기반 분리 제외 | 곡마다 스테레오 구성이 달라 일반화 어려움 |
| 상용 분리 API 제외 | 유료, 외부 의존 |
| G 직접 구현 | 발표 질의 대응, 의존성 충돌 회피. 검증은 scipy 기준 테스트로 보완 |
| Google DDSP 미사용 | TensorFlow 기반, 앰프 블록 부재, 오래된 버전 고정 |
| 톤스택은 셸빙·피킹 필터, 캐비닛은 파라메트릭 필터 | 일정, 라이선스, 인코더가 맞히기 쉬운 파라미터 수 |
| ④⑤를 필수에 포함하되 ⑥ 경로를 먼저 완성 | 성능과 발표의 핵심 AI 모델이지만 실패 위험을 분리 |
| 시연은 녹음된 DI 리프를 DAW에서 재생 | 현장 연주 변수 제거, 같은 리프로 원곡과 직접 비교 |
| 인코더 체크포인트에 θ 버전 기록, 최종본은 저장소에 커밋 | 다른 사람이 재학습 없이 사용, G 변경 시 잘못된 가중치 사용 방지 |
| ⑥에서 교환 게인 고정·드라이브 앵커·최소 이동 선택(G·통계는 그대로) | 실험 1 v1에서 게인이 평평한 방향으로 흘러감. θ 버전·인코더·통계 정의를 바꾸지 않고 일주일 안에 할 수 있는 대응 |
| EGDB·NAM 학습 입력 신호는 재배포하지 않고 공식 경로에서 내려받게 함 | 두 파일 모두 재배포 조건이 명시되지 않음. 사용자 편의는 같고 라이선스 문제가 없음 |

## 12. 참고 자료

- Wright, Välimäki, Juvela. Adversarial Guitar Amplifier Modelling with Unpaired Data. ICASSP 2023. https://arxiv.org/abs/2211.00943
- Chen, Choi, Liao 외. Improving Unsupervised Clean-to-Rendered Guitar Tone Transformation Using GANs and Integrated Unaligned Clean Data. DAFx 2024. https://arxiv.org/abs/2406.15751
- Yeh 외. DDSP Guitar Amp: Interpretable Guitar Amplifier Modeling. Preprint. https://arxiv.org/abs/2408.11405
- Comunità, Steinmetz, Reiss. NablAFx: A Framework for Differentiable Black-box and Gray-box Modeling of Audio Effects. https://arxiv.org/abs/2502.11668
- Comunità, Steinmetz, Reiss. Differentiable black-box and gray-box modeling of nonlinear audio effects. Frontiers in Signal Processing 5, 2025. https://doi.org/10.3389/frsip.2025.1580395
- ST-ITO: Controlling Audio Effects for Style Transfer with Inference-Time Optimization. ISMIR 2024. https://arxiv.org/abs/2410.21233
- Steinmetz, Bryan, Reiss. Style transfer of audio effects with differentiable signal processing (DeepAFx-ST). Journal of the Audio Engineering Society, 2022, pp. 708–721. https://arxiv.org/abs/2207.08759
- DDSP: Differentiable Digital Signal Processing. ICLR 2020. https://arxiv.org/abs/2001.04643
- Chen, Hsiao, Hsieh, Jang, Yang. Towards Automatic Transcription of Polyphonic Electric Guitar Music: A New Dataset and a Multi-Loss Transformer Model. ICASSP 2022. https://arxiv.org/abs/2202.09907 (EGDB 원 논문. 이 프로젝트가 쓰는 `audio_DI`, 240 tablatures. https://ss12f32v.github.io/Guitar-Transcription/)
- Towards Generalizability to Tone and Content Variations in the Transcription of Amplifier Rendered Electric Guitar Audio. https://arxiv.org/abs/2504.07406 (EGDB를 BiasFX2로 다시 렌더링한 확장 데이터셋 EGDB-PG. 이 프로젝트는 쓰지 않음)
- Yamamoto, Song, Kim. Parallel WaveGAN (MR-STFT 손실). ICASSP 2020. https://arxiv.org/abs/1910.11480
- Pakarinen, Yeh. A Review of Digital Techniques for Modeling Vacuum-Tube Guitar Amplifiers. Computer Music Journal 33(2), 2009.
- McDermott, Simoncelli. Sound Texture Perception via Statistics of the Auditory Periphery. Neuron 71(5), 2011.
- Hinton, Vinyals, Dean. Distilling the Knowledge in a Neural Network. 2015. https://arxiv.org/abs/1503.02531
- Neural Amp Modeler 문서. https://neural-amp-modeler.readthedocs.io
- NAM 학습 코드의 ESR 등급(`_esr_comment`). https://github.com/sdatkinson/neural-amp-modeler/blob/main/nam/train/core.py
- Positive Grid BIAS X (Music-to-Tone). https://www.positivegrid.com/products/bias-x
- HT-Demucs. Rouard, Massa, Défossez. Hybrid Transformers for Music Source Separation. ICASSP 2023. (facebookresearch/demucs 저장소는 2025-01 아카이브)
- torchaudio `lfilter` 문서. https://docs.pytorch.org/audio/stable/generated/torchaudio.functional.lfilter.html

## 13. 용어집

문서, 코드, 발표에서 아래 이름만 쓴다. 새 용어를 만들면 여기에 먼저 적는다.

| 용어 | 뜻 | 영어 표준 용어 | 코드 |
|---|---|---|---|
| 세션 정보 | 사용자가 넣는 기타·앰프·캐비닛·부스트 정보(4.3 스키마). "장비 정보"라고 쓰지 않는다 | rig metadata | `session`, `session.json` |
| 세션 벡터 | 세션 정보의 21차원 조건 벡터(원-핫 × confidence, unknown 칸은 1) | conditioning vector | `session_vector`, 배치 키 `session_vec` |
| 조건 드롭아웃 / confidence 증강 | ④에서 필드를 unknown으로 가림 / confidence를 무작위로 낮춤 | conditioning dropout | `dropout_session` |
| 대응표 | 세션 값 → θ 차원별 범위 표 | — | `MAPPING` |
| 대응표 버전 | 대응표 값과 사전 범위 규칙의 버전. θ 버전과 별개. 실험 1 결과·`prior.json`·인코더 체크포인트 이름에 기록 | — | `MAPPING_VERSION`, `mapping_version` |
| 게인 우선 규칙 | 사전 범위 교집합이 비면 `amp.gain` 범위를 쓰는 규칙(없으면 합집합) | — | `compute_prior` |
| 사전 범위 | 차원별 [하한, 상한]과 중심값. 확률분포가 아니다(④에서만 균등 샘플링). 문헌은 "prior"라는 말을 쓰지 않는다 | parameter ranges (DeepAFx-ST), prior range | `Prior`, `prior.json` |
| 곡별 파라미터 최적화 | ⑥. 가중치는 그대로 두고 θ만 곡마다 최적화. 앱 화면 이름은 "톤 찾기" | inference-time optimization | `fit`, `FitConfig`, `FitResult` |
| 통계 매칭 손실 | ⑥의 통계 4종 L1 손실. 근거: McDermott & Simoncelli 2011(통계로 소리를 합성), 프로젝트는 같은 원리로 θ를 최적화 | summary-statistics matching loss | `stat_losses` |
| 장기 평균 스펙트럼 | 프레임 log-Mel 평균. 문헌의 LTAS(Pestana 외 2013)는 선형 파워 스펙트럼 평균이라 계산이 다르다 | LTAS (log-Mel 변형) | `ltas` |
| 레벨별 스펙트럼 | 음량 분위수 4구간별 평균 스펙트럼(프로젝트 정의) | level-conditioned spectrum | `level` |
| 시작점 | ⑥ 최적화 초기값 | initialization, multi-start | `prior_center`, `prior_sample_i`, `encoder`; 실험 1 조건 `prior_multi` |
| θ* / 후보 | 최저 손실 θ(= 후보 0) / 서로 다른 상위 결과 | best solution / candidates | `theta_star` |
| (b) 세션만 경로 | ⑤⑥ 없이 사전 범위 중심을 θ로 쓰는 비교 기준 | session-only baseline | `--session-only`, `session_only` |
| 파라미터 추정 인코더 | ⑤. 스템과 세션 벡터 → θ̂ | parameter estimation network | `encoder.py` |
| 증류 | ⑦. 그레이박스 G(θ*)를 교사로, NAM WaveNet이 같은 입력에 대한 G의 출력을 따라 하도록 학습(응답 기반 증류, DSP→신경망 에뮬레이션). Hinton 2015의 원 정의는 신경망 교사다 | (response-based) knowledge distillation; neural emulation | `distill.py` |
| 파라미터 복원 실험 | 실험 1. 정답 θ로 만든 가짜 곡에서 θ를 되찾음 | parameter recovery | `run_experiment1`, `make_fake_songs` |
| 입력단 게인 | θ의 입력단 게인 | input gain | `in_gain_db` |
| DI 레벨 증강 | ④에서 DI 음량을 무작위로 바꾸는 범위 | input level augmentation | `SynthConfig.di_gain_range_db` |
| 입력 레벨 | ⑦에서 NAM 입력을 DI 기준 음량에 맞춘 디지털 배율(dBu 아님) | input level | `input_gain_db`, `input_level.json` |
| 부스트(오버드라이브) | 부스트 블록. 값 `tube_screamer` = Tube Screamer 계열 | boost / overdrive | `boost_*`, 블록 `booster` |
| 톤스택 | 저역·미드·고역 EQ 블록. `ts_` 접두사는 톤스택만 뜻한다 | tone stack | `ts_*` |
| 클리핑 비대칭(DC 바이어스) | 프리앰프 단마다 tanh 앞에 더하는 DC 바이어스 | asymmetric clipping (offset / DC bias), Comunità 외 2025 §3.2.2 | `pre1_bias`, `pre2_bias` |
| 딥 | 캐비닛의 음수 게인 피킹 컷(엄밀한 notch 아님) | peaking cut | `cab_notch_hz`, `cab_notch_db` |
| 리앰핑 | DI를 앰프에 다시 통과시켜 녹음 | reamping | — |
| 캡처 | `.nam` 모델 파일(NAM 공식 문서는 "model") | capture / model | — |
| NAM 학습 입력 신호 | NAM 문서의 `input.wav`(v3.0.0), 커뮤니티 표기 `v3_0_0.wav` | NAM input file | `nam_input_path` |
| ESR 목표 0.01 | NAM 학습기의 "Great!" 기준 차용, pre-emphasis 없음 | error-to-signal ratio | `ESR_TARGET`, `esr` |
| mel FAD 대체 | log-Mel 위 Fréchet 거리(VGGish 아님) | — | `fad_mel` |
| MUSHRA식 청취 평가 | 0~100 척도만 차용 | MUSHRA-like | `mushra_summary` |
| DI 풀(pool 분할) | ⑥에서 쓰는 EGDB DI 분할 | held-out DI pool | `SPLITS["pool"]` |
| 마감점 | 8장 일정의 확인 시점(모델 체크포인트와 구분) | milestone | — |
| 그레이박스 | 해석 가능한 파라미터 θ를 가진 미분 가능 DSP 블록의 고정 연쇄. 블록 구조는 고정하고 θ만 학습한다. 정의는 Comunità 외 2025 §3.2를 따른다(Aalto 그룹의 grey-box는 회로 해석까지 쓰는 더 넓은 뜻) | gray-box | `graybox/`, `G` |
| MR-STFT 손실 | 여러 STFT 해상도에서 스펙트럼 크기 차이를 더한 손실(Yamamoto 외 2020). 논문마다 MR-STFT, MRSL로 이름이 다르다 | multi-resolution STFT loss | `mrstft` |
| 주파수 샘플링 방식 FIR 근사 | 필터 계수로 주파수 응답을 구해 FFT 영역에서 곱하는 학습용 필터 계산 | frequency sampling | `mode="fft"` |
| 톤 찾기 | ⑥ 곡별 파라미터 최적화의 앱 화면 이름(발표·UI 전용) | — | 앱 탭 "4. 톤 찾기 (⑥)" |
| 추천 구간 | ①에서 단일 기타 점수가 문턱 이상인 구간. 참고용 | candidate segments | `recommend_segments` |
| 단일 기타 점수 | 창마다 좌우 상관도와 Side/Mid 에너지 비로 매긴 "기타 한 대" 가능성 | — | `stereo_analysis` |
| 유효 연주 길이 | 무음 프레임을 뺀 구간 길이 합(기본 15초 이상) | active duration | `active_sec` |
| 여러 기타 혼입 경고 | 기타 여러 대가 섞였을 수 있다는 경고. `same_rig`이면 참고 문구로 낮춤 | — | `multi_guitar_suspected` |
| 전체 스템 대체 | 고른 구간이 없을 때 전체 스템을 쓰는 것("여러 기타가 섞인 평균 톤") | fallback | `full_stem_used` |
| 곡 단위 장비 / 평소 장비 | ②-0에서 그 곡에서 쓴 장비 / 아티스트가 평소 쓰는 장비 | — | `scope: song / usual` |
| LLM 장비 조사 | ②-0. LLM과 웹 검색으로 세션 정보 초안 작성 | — | `research_llm.py` |
| 레벨 구간 | 연주 프레임 음량 분위수로 나눈 4구간 | level bins (quantile) | `StatConfig.tau_db` |
| 다이내믹스 통계 | 크레스트 팩터 평균·분산, 프레임 음량 분포 폭 | dynamics statistics | `dyn` |
| 대역별 dB 차이(잔차) | 장기 평균 스펙트럼의 4대역(저역·중역·중고역·고역) 평균 차이(예측 − 타깃) | band-wise level difference | `band_diff_db`, `describe_residual` |
| 분리 잔여물 근사 증강 | ④에서 믹스 처리 증강 뒤에 거는 무작위 로우패스·노이즈 | — | `augment_separation` |
| 교환 게인 / 교환 게인 고정 | 사이에 선형 블록만 있어 dB 합만 출력에 남는 게인 쌍 / ⑥에서 쌍의 앞쪽을 시작점 값에 고정하는 것 | gain exchange (block-oriented models) | `EXCHANGEABLE`, `exchangeable_mask` |
| 드라이브 앵커 | ⑥에서 비선형 4곳의 입력 게인을 시작점 쪽으로 당기는 가우시안 벌점 | — | `lambda_anchor`, `anchor_sigma` |
| 최소 이동 선택 | 손실이 최저의 (1 + ε) 안이면 시작점에서 덜 움직인 결과를 θ*로 고르는 규칙 | — | `select_eps`, `move` |
| 식별 가능 게인 조합 / 게인 합 오차 | 출력이 결정하는 게인 dB 합 / 그 합의 정답 대비 오차 | identifiable combination | `GAIN_SUMS`, `gain_sum_err` |
| 게인 식별 진단 | 통계 특징의 θ 민감도 JᵀJ 고유분해 | Fisher information / sensitivity analysis | `stat_sensitivity` |
| 바이어스 기준형 | θ 오차를 잴 때 `pre1_bias ≥ 0`이 되게 두 바이어스를 함께 뒤집은 θ | — | `canonical_bias` |
| 믹스 처리 증강 | ④와 실험 1에서 무작위 EQ·컴프·짧은 리버브를 거는 것 | mix augmentation | `augment_mix` |
| 가짜 곡 | 실험 1에서 정답 θ와 EGDB test DI로 만든 평가용 곡(20개) | synthetic test song | `make_fake_songs` |
| 정답 θ / G(θ_true) | 가짜 곡을 만든 θ / 그 θ를 넣은 G | ground-truth parameters | `theta_true` |
| 미사용 DI | 가짜 곡이나 ⑦ 학습에 쓰지 않은 test 분할 DI | held-out DI | `eval_clips` |
| 중립 θ | 선형 블록이 평탄해지는 θ(EQ 0dB, 비대칭 0, 드라이브 최소) | neutral setting | `neutral_theta` |
| DI 기준 음량 | ④⑥⑦에서 DI를 맞추는 RMS 0.05 | reference input RMS | `DI_REF_RMS` |
| θ 버전 | G의 블록 구성·범위·고정 상수 버전. 체크포인트와 맞지 않으면 인코더를 쓰지 않음 | — | `THETA_VERSION`, `theta_version` |
| 시연 곡 | 모든 중간 결과를 미리 만들어 두는 2~3곡 | — | — |
| (a) 후보 화면 | ⑥ 후보 A(θ*)/B를 들어 보는 화면(선택 기능) | — | 앱 `cand_a`, `cand_b` |
| 시간 기반 효과(공간계) | 딜레이·리버브. G가 다루지 않음 | time-based effects | — |
