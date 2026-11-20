# 세션 정보 입력 안내

`.nam`을 만들 때 곡의 기타 톤에 대해 아는 것을 알려 주는 항목들입니다. Colab 노트북(`notebooks/02_song_to_nam.ipynb`)에서는 드롭다운으로 고르고, 노트북이 이 값으로 `session.json`을 만듭니다.

모르는 항목은 `unknown`으로 두세요. 틀린 값보다 낫습니다.

- 세션 정보는 톤 찾기의 시작점과 탐색 범위를 정하고, 인코더에도 들어갑니다.
- 최종 톤은 음원의 통계에 맞춰 정합니다.
- 다만 틀린 값이 시작점을 엉뚱한 곳에 두면 결과가 나빠집니다(아래 "주의할 점").

스키마와 대응표의 기준은 설계 문서 4.3, 코드는 `ampcap/session.py`(`FIELDS`, `MAPPING`)입니다. 이 문서의 수치는 대응표 버전 2 기준입니다.

## 한눈에 보기

| 항목 | 고를 수 있는 값 | 바뀌는 것 |
|---|---|---|
| 곡 정보 `artist`, `song` | 자유 입력 | 기록용. 실행 폴더 이름과 `.nam` 메타데이터 |
| 캡처 대상 `capture_target` | `rhythm` / `lead` | 구간 선택 화면의 안내 문구 |
| 같은 장비 `same_rig` | 예 / 아니요 | 여러 기타 혼입 경고를 참고 문구로 낮출지 |
| 픽업 `guitar.pickup` | `single` / `p90` / `humbucker` / `unknown` | 입력단 게인 범위 |
| 앰프 계열 `amp.family` | `fender` / `vox` / `british` / `american_hg` / `modern_hg` / `unknown` | 프리앰프 게인·클리핑 비대칭, 톤스택, 파워앰프, 입력 하이패스 범위 |
| 앰프 모델 `amp.model` | 자유 입력 | 기록용. 알려진 이름이면 앰프 계열을 자동으로 채움 |
| 게인 `amp.gain` | `low` / `mid` / `high` / `unknown` | 프리앰프 1·2단 게인 범위 |
| 캐비닛 `cab.size` | `1x12` / `2x12` / `4x12` / `unknown` | 캐비닛 하이패스·공진·로우패스 범위 |
| 부스트 `boost` | `none` / `tube_screamer` / `unknown` | 부스트(오버드라이브) 블록 |
| 확신도 `confidence` | 항목별 0 ~ 1 (기본 1.0) | 위 범위를 얼마나 믿을지 |

## 항목별 설명

### 곡 정보 (`artist`, `song`)

기록용입니다. 톤 계산에는 쓰지 않습니다.

### 캡처 대상 (`capture_target`)

곡에서 어떤 기타 파트의 톤을 만들지 고릅니다.

- `rhythm`: 코드·리프를 치는 반주 기타
- `lead`: 솔로·멜로디 기타

구간을 고를 때 보여 주는 안내 문구만 바뀝니다.

실제로 어떤 파트의 톤이 나오는지는 고른 구간으로 정해집니다. 리드 톤을 원하면 솔로 구간만 고르세요.

### 같은 장비 (`same_rig`)

곡에 기타가 여러 대 나올 때, 모두 같은 기타·앰프로 녹음했는지 고릅니다.

- 예: 한 사람이 같은 장비로 여러 번 겹쳐 녹음했을 때. 여러 기타가 섞여도 톤은 비슷하므로 혼입 경고를 참고 문구로 낮춥니다.
- 아니요: 기타리스트가 둘 이상이거나 파트마다 장비가 다를 때. 섞인 구간은 평균 톤이 되므로 경고를 그대로 보여 줍니다.

모르면 "아니요"가 안전합니다.

### 픽업 (`guitar.pickup`)

기타의 픽업 종류입니다. 픽업 출력이 클수록 앰프에 들어가는 신호가 커서 입력단 게인 범위가 달라집니다.

| 값 | 대표 기타 | 입력단 게인 범위 |
|---|---|---|
| `single` | Fender Stratocaster, Telecaster | −10 ~ 0 dB |
| `p90` | Gibson Les Paul Junior/Special, SG Special | −4 ~ +4 dB |
| `humbucker` | Gibson Les Paul, SG, ES-335, 대부분의 메탈 기타 | −2 ~ +11 dB |

### 앰프 계열 (`amp.family`)

앰프의 회로 성격입니다. θ 범위를 가장 많이 좁히는 항목입니다.

| 값 | 대표 앰프 | 소리 성격 | 프리앰프 게인 (1단 / 2단) |
|---|---|---|---|
| `fender` | Twin Reverb, Deluxe Reverb, Bassman, Princeton, Hot Rod | 맑은 클린, 깊은 저역 | 0 ~ 18 / 0 ~ 16 dB |
| `vox` | AC30, AC15 | 반짝이는 고역, 클래스 A 크런치 | 0 ~ 26 / 0 ~ 20 dB |
| `british` | Marshall JCM800, Plexi, JTM45, Orange | 미드가 앞으로 나오는 크런치 ~ 하드록 | 12 ~ 36 / 10 ~ 34 dB |
| `american_hg` | Mesa Rectifier, Mark 시리즈, Peavey 5150/6505, Soldano, EVH | 두꺼운 하이게인, 메탈 | 22 ~ 40 / 18 ~ 40 dB |
| `modern_hg` | Diezel, ENGL, Friedman, Revv, Fortin, Kemper·Axe-FX 프로필 | 타이트한 모던 하이게인 | 24 ~ 40 / 20 ~ 40 dB |

톤스택, 파워앰프 드라이브, 새그(파워앰프 압축감), 입력 하이패스 범위도 함께 바뀝니다. 계열별 전체 값은 `ampcap/session.py`의 `MAPPING["amp.family"]`에 있습니다.

### 앰프 모델 (`amp.model`)

앰프 이름을 알면 적습니다. 기록용입니다.

앰프 계열을 `unknown`으로 두고 아래 이름 중 하나를 적으면 계열을 자동으로 채웁니다. 대소문자·공백·하이픈은 무시하고, 부분 일치로 찾습니다.

- fender: twin, deluxe, bassman, princeton, fender, hotrod, bluesjr
- vox: ac30, ac15, vox
- british: jcm800, jcm900, plexi, jtm45, superlead, 1959, 1987, orange, rockerverb, jvm, silverjubilee, marshall
- american_hg: rectifier, dualrec, 5150, 6505, markiv, markiic, mesa, evh, soldano, slo100
- modern_hg: diezel, vh4, herbert, engl, powerball, uberschall, fortin, revv, friedman, axefx, kemper

### 게인 (`amp.gain`)

앰프에서 얼마나 찌그러진 소리인지입니다.

곡을 들어 보고 고릅니다. 같은 Marshall이라도 곡마다 다릅니다.

| 값 | 들리는 소리 | 예 | 프리앰프 게인 (1단 / 2단) |
|---|---|---|---|
| `low` | 클린, 세게 칠 때만 살짝 깨짐 | 펑크 커팅, 재즈, 블루스 클린 | 0 ~ 14 / 0 ~ 12 dB (부스트 드라이브도 0 ~ 10 dB로 제한) |
| `mid` | 크런치. 코드가 하나하나 들림 | AC/DC, 그런지 리듬, 클래식 록 | 10 ~ 28 / 6 ~ 24 dB |
| `high` | 하이게인. 서스테인이 길고 코드가 뭉침 | 메탈, 하드코어, 모던 록 리드 | 22 ~ 40 / 18 ~ 40 dB |

앰프 계열과 범위가 겹치지 않으면 게인 값을 우선합니다(게인 우선 규칙).
예를 들어 `fender` + `high`는 Fender를 세게 밀어붙인 소리로 봅니다.

### 캐비닛 (`cab.size`)

스피커 캐비닛의 크기입니다. 저역의 양과 고역이 잘리는 위치가 바뀝니다.

| 값 | 대표 | 캐비닛 하이패스 | 공진 | 로우패스 |
|---|---|---|---|---|
| `1x12` | 콤보 앰프(Deluxe Reverb, AC15 등) | 60 ~ 160 Hz | 70 ~ 130 Hz | 3.5 ~ 6.5 kHz |
| `2x12` | 큰 콤보(Twin Reverb, AC30), 2x12 캐비닛 | 50 ~ 130 Hz | 75 ~ 140 Hz | 제한 없음 |
| `4x12` | Marshall·Mesa 스택 | 40 ~ 100 Hz | 65 ~ 130 Hz | 3 ~ 6 kHz |

모르면 앰프 계열을 보고 짐작합니다.

- 하이게인·Marshall 스택은 보통 `4x12`
- Fender·Vox 콤보는 `1x12`나 `2x12`

짐작이면 확신도를 낮추세요.

### 부스트 (`boost`)

앰프 앞에 오버드라이브 페달을 걸었는지 고릅니다.

| 값 | 뜻 | 바뀌는 것 |
|---|---|---|
| `none` | 페달 없이 앰프만 | 부스트 드라이브·미드를 0으로 고정 |
| `tube_screamer` | Tube Screamer 계열 오버드라이브(Ibanez TS808·TS9, Maxon OD808 등). 클리핑 앞에서 미드를 크게 올리고 저역을 줄임 | 미드 600 ~ 1000 Hz를 8 ~ 12 dB 올림, 드라이브 6 ~ 30 dB |
| `unknown` | 모름 | 제한 없음 (0 ~ 30 dB) |

부스트 블록은 Tube Screamer 계열을 기준으로 만들었습니다. Klon, OCD처럼 성격이 다른 오버드라이브와 퍼즈, 디스토션 페달(Big Muff, RAT 등)은 `tube_screamer`로 고르지 말고 `unknown`으로 두세요. `unknown`이면 부스트 범위가 열려 있어 음원에 맞는 값을 찾습니다.

### 확신도 (`confidence`)

확신도는 `guitar`, `amp`, `cab`, `boost` 네 항목에 따로 매깁니다. `amp` 확신도 하나가 앰프 계열과 게인에 함께 적용됩니다.

| 값 | 언제 |
|---|---|
| 1.0 | 아티스트·엔지니어 인터뷰처럼 그 곡에 대한 확실한 출처가 있을 때 |
| 0.7 ~ 0.8 | 그 곡에 대한 출처는 있지만 일부가 짐작일 때 |
| 0.5 | 아티스트가 평소에 쓰는 장비이거나, 장비 소개 사이트(Equipboard 등)의 재구성일 때 |
| 0.3 이하 | 거의 짐작일 때. 이럴 땐 `unknown`도 괜찮습니다 |

확신도 c가 1보다 작으면 범위를 그만큼 전체 범위 쪽으로 넓힙니다.

0 ~ 1로 정규화한 범위 [하한, 상한]은 [하한 × c, 상한 × c + (1 − c)]가 됩니다.

## 예시: Pearl Jam "Even Flow" (리듬)

```json
{
  "artist": "Pearl Jam", "song": "Even Flow",
  "capture_target": "rhythm",
  "same_rig": false,
  "guitar": {"pickup": "humbucker"},
  "amp": {"family": "british", "model": "JCM800", "gain": "mid"},
  "cab": {"size": "4x12"},
  "boost": "tube_screamer",
  "confidence": {"guitar": 1.0, "amp": 0.8, "cab": 0.5, "boost": 0.5}
}
```

- Stone Gossard가 이 곡에 "Les Paul과 Marshall"을 썼다고 말했습니다. 그래서 `humbucker`와 `british`이고, 기타 확신도는 1.0입니다.
- JCM800, Tube Screamer는 장비 소개 사이트가 재구성한 정보라 부스트 확신도는 0.5입니다.
- Marshall 스택이라 `4x12`로 짐작했고, 확신도는 0.5입니다.
- 기타리스트가 둘(Gossard는 Les Paul, McCready는 Strat)이라 `same_rig`는 false입니다.

## 주의할 점

- 게인을 장비만 보고 `high`로 고르지 마세요.
  - 2026-11-14에 Even Flow를 `british` + `high` + `tube_screamer`로 돌렸더니, 실제 기타로 쳤을 때 게인이 과하고 소리가 뭉개졌습니다.
  - Marshall에 부스트를 걸었어도 곡 자체는 크런치(`mid`)였습니다.
- 부스트와 높은 게인을 함께 확신하지 마세요.
  - 둘 다 드라이브를 올립니다. 어느 한쪽이 짐작이면 그쪽 확신도를 낮추세요.
- 모르는 값을 억지로 채우지 마세요.
  - 짐작한 값은 확신도를 0.5 이하로 두거나 `unknown`으로 두세요.
