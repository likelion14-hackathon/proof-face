# skin-metrics

얼굴 사진 1장에서 **색소침착, 홍조, 모공** 세 가지를 0~100점으로 산출하는 피부 분석
엔진이다. 딥러닝이 아니라 빛과 색을 계산하는 물리 기반 방식이 실제로 점수를 내며, 그 점수는
한국인 965명의 실측 데이터에 맞춰 보정돼 있다.

> **의료기기가 아니다.** 출력은 미용 참고 정보일 뿐이며 진단 목적으로 사용할 수 없다.
> **This system is not a medical device.** Outputs are cosmetic reference information only
> and must not be used for diagnostic purposes.

---

## 목차

- [빠른 시작](#빠른-시작)
- [사진 한 장이 점수가 되기까지 (쉬운 설명)](#사진-한-장이-점수가-되기까지-쉬운-설명)
- [설계 개요](#설계-개요)
- [Phase 1: 물리 기반 파이프라인 (기술 상세)](#phase-1-물리-기반-파이프라인-기술-상세)
- [Phase 2: 딥러닝 (기술 상세)](#phase-2-딥러닝-기술-상세)
- [레퍼런스 보정](#레퍼런스-보정-skin_metricscalibrate)
- [점수 해석, 신뢰도, 보정](#점수-해석-신뢰도-보정)
- [CLI 레퍼런스](#cli-레퍼런스)
- [HTTP API](#http-api)
- [Docker](#docker)
- [AWS EC2 배포](#aws-ec2-배포)
- [테스트](#테스트)
- [정확도를 더 올리려면: 데이터 확보 가이드](#정확도를-더-올리려면-데이터-확보-가이드)
- [다음 단계](#다음-단계)

---

## 빠른 시작

```bash
# 코어 (Phase 1 물리 파이프라인 + 테스트)
uv sync

# 실이미지 얼굴 검출 (MediaPipe)
uv sync --extra detection

# Phase 2 딥러닝
uv sync --extra dl

# HTTP API (이미지 URL 입력, 분석 결과 출력)
uv sync --extra api --extra detection

# 분석 (venv 활성화 시 uv 없이 skin-metrics 직접 실행 가능)
# 첫 실행: --download-model 로 FaceLandmarker 모델(약 3.8MB) 자동 다운로드
skin-metrics analyze data/face.jpg --download-model --output report.json

# 이후 실행: 모델 캐시 재사용 (플래그 불필요)
skin-metrics analyze data/face.jpg --output report.json

# 그레이카드나 흰 종이가 프레임에 있으면 그 영역 지정 (보정 신뢰도 상승)
skin-metrics analyze data/face.jpg --reference-bbox 10,10,40,40 --output report.json

# 두 시점 비교 (같은 사람, 같은 조건 촬영 권장)
skin-metrics compare data/before.jpg data/after.jpg

# Phase 2 딥러닝 스캐폴드 (라벨 없이 더미로 학습 루프 검증)
skin-metrics train --dummy --mode ranking --epochs 3

# HTTP API 서버 (docs: http://127.0.0.1:8000/docs)
skin-metrics serve --download-model
curl -X POST http://127.0.0.1:8000/analyze \
  -H 'content-type: application/json' \
  -d '{"image_url": "https://example.com/face.jpg"}'

# 다이어리용 0~10 점수 (피부 톤, 모공, 붉은기). 202 + request_id 반환
curl -X POST http://127.0.0.1:8000/analyze/diary \
  -H 'content-type: application/json' \
  -d '{"image_url": "https://example.com/face.jpg"}'

# 결과는 Redis {request_id}:diary 에 저장된다. 디버깅용 조회:
curl http://127.0.0.1:8000/results/<request_id>:diary
```

> `uv`는 `~/Library/Python/3.9/bin`에 설치된다. PATH에 없으면 전체 경로로 부르거나,
> `source .venv/bin/activate` 로 venv를 활성화하면 `skin-metrics` 콘솔 스크립트를 uv 없이
> 쓸 수 있다.

---
## 사진 한 장이 점수가 되기까지 (쉬운 설명)

같은 얼굴을 폰으로 찍고 태블릿으로 찍으면 피부색이 다르게 나온다. 카메라마다 밝기와 색을
손보는 방식이 다르기 때문이다. 그런데도 이 시스템은 사진 1장만 받아 **색소침착, 홍조,
모공** 세 가지를 0~100점으로 내놓는다.

이 섹션이 답하는 것은 두 가지다. 사진을 어떤 식으로 분석하는가, 그리고 그 결과를 어떻게
믿을 만하게 만들었는가.

먼저 오해 하나를 걷어낸다. 이 엔진은 딥러닝이 아니다. 사진 속 픽셀의 밝기와 색을 물리와
색상 공식으로 계산하는 방식이다. 그리고 의료기기가 아니다. 미용 참고용 지표다.

### 1부. 사진을 어떻게 읽는가, 6단계

전체 흐름은 이렇다. 얼굴을 찾고, 크기를 맞추고, 색을 되돌리고, 쓸 부위만 오려내고,
물리량을 계산하고, 점수로 바꾼다.

**1) 얼굴 찾기.** MediaPipe(구글의 얼굴 인식 도구)로 얼굴 위에 점 468개를 찍는다. 눈과 코,
입 윤곽이 좌표로 잡히므로 사진의 어디가 이마이고 어디가 볼인지 알 수 있다.

**2) 얼굴 크기 맞추기.** 피부 거칠기는 정해진 픽셀 간격으로 재기 때문에, 얼굴이 크게 찍힌
사진과 작게 찍힌 사진을 그냥 비교하면 안 된다. 기기별 눈 사이 거리부터가 다르다.

| 기기 | 원래 눈 사이 거리 |
|---|---|
| 디지털카메라 | 1140px |
| 태블릿 | 972px |
| 폰 | 889px |

그래서 눈과 눈 사이 거리를 512px로 통일한다. 단 축소만 하고 확대는 절대 하지 않는다.
확대하면 텍스처가 매끄러워져 실제보다 모공이 적은 피부로 읽히기 때문이다.

**3) 색 되돌리기.** JPEG 사진에는 감마(화면에서 보기 좋도록 밝기를 비틀어 저장한 것)가
걸려 있다. 이를 풀어 빛의 세기 그대로의 숫자, 즉 선형 RGB로 되돌린다. 화이트밸런스(조명
색을 중립으로 맞추는 보정)는 기본적으로 카메라가 이미 한 보정을 믿는다. 그 이유는 2부의
B-1에 있다. 그레이카드나 흰 종이를 함께 찍고 그 위치를 알려주면 정확도가 올라간다.

**4) 쓸 부위만 오려내기.** 이마, 왼볼, 오른볼, 코, 턱 5군데만 쓴다. 그 안에서도 번들거리는
하이라이트(밝고 채도 낮은 픽셀), 어두운 그림자(하위 5%), 머리카락과 눈썹과 입술은 지운다.
남은 픽셀이 60% 미만인 부위는 아예 버린다.

**5) 부위마다 물리 계산.** 지표마다 재는 대상이 다르다.

- 색소침착: 주변보다 어두운 점을 찾아 반점 개수, 면적 비율, 진하기, 그리고 톤이 얼마나
  고른지(밝기 표준편차)를 잰다.
- 홍조: 피부가 빨강을 덜 흡수하고 초록을 더 흡수하는 정도(홍반 지수)와 CIELab a*값
  (빨강과 초록 축의 색 좌표)을 쓴다.
- 모공: 피부 표면의 잔결을 텍스처 분석(GLCM, LBP, 라플라시안 고주파. 픽셀 무늬의 규칙성과
  거칠기를 수치로 바꾸는 방법)으로 잰다. 모공은 1mm가 안 되는 구멍이지만 사진에 실제로
  찍히기 때문에, 세 지표 중 실측 일치도가 가장 높다.

**6) 점수 매기기.** 계산값을 그대로 주지 않는다. 한국인 965명 데이터의 분포에 대고
"이 사람은 몇 등쯤"인지를 0~100 백분위로 바꾼다. 비교 대상은 피부 톤 타입(Fitzpatrick,
피부가 햇빛에 타고 붉어지는 정도로 나눈 분류)별로 다른 분포다. 보정 상태와 유효 부위
개수에 따라 confidence(신뢰도, 0~1)도 함께 나온다.

여기까지를 한 문장으로 줄이면, **얼굴에서 잴 수 있는 곳만 골라 물리량을 재고 그 값을
한국인 분포 위의 등수로 바꾼다**가 된다.

### 2부. 정확도는 어떻게 높였는가

원칙은 하나다. 느낌이 아니라 실측 데이터로 채점했다.

채점표는 AI-Hub 《028. 한국인 피부상태 측정 데이터》다. 965명, 정면 2,895장, ROI 11,580행에
피부 수분 측정 장비(Corneometer) 수치와 전문의 등급이 붙어 있다. 사람 단위로 학습용 858명과
검증용 107명을 나누고, 검증용 사람은 한 번도 보지 않은 상태에서 점수의 순위가 실측과 얼마나
맞는지(Spearman 상관, 1.0이면 순위가 완전히 일치)를 쟀다.

채점 기준이 생기자, 눈으로 봐서는 멀쩡하고 코드로도 정상 동작하던 계산에서 그럴듯한데 사실
틀린 버그가 줄줄이 드러났다. 정확도는 대부분 새 기법이 아니라 아래 일곱 가지를 걷어내면서
올라갔다.

**B-1. 화이트밸런스가 오히려 답을 지우고 있었다.** 가장 큰 문제였다. gray-world는 사진
전체의 평균은 회색일 것이라 가정하고 색을 되돌리는 방법이다. 그런데 얼굴 사진은 화면
대부분이 피부라서 평균이 곧 피부색이다. 그 평균을 회색으로 만들면 재려던 피부색이 통째로
사라진다. 실제로 a*는 15.9에서 0.5로, b*는 26.6에서 -1.2로 붕괴했다. 전문의 색소 등급과의
상관은 gray-world가 -0.307로 방향까지 반대였고, 보정을 끄면 +0.437이었다. 그래서 기본값을
끔으로 바꿨다.

**B-2. 기준 눈금이 실제 값과 두 자릿수 어긋나 있었다.** 거칠기 특징의 기준 평균이 0.10으로
적혀 있었는데 실제 값은 0.0009였다. 965명 데이터로 다시 쟀다.

**B-3. 카메라가 바뀌면 뒤집히는 특징을 뺐다.** 피부의 절대 색(멜라닌 지수, ITA)은 카메라마다
재현이 달라 기기별로 상관의 부호까지 뒤집힌다.

| 특징 | 디지털카메라 | 태블릿 | 폰 |
|---|---|---|---|
| melanin_index (절대 색) | +0.52 | -0.09 | +0.03 |
| 반점 개수 (형태) | +0.60 | +0.73 | +0.63 |

옛 공식은 위쪽의 불안정한 값에 가중치 40%를 주고, 세 기기 모두에서 튼튼한 아래쪽은 아예
쓰지 않았다. 반점의 모양과 개수 같은 형태 특징만 남기자 직전 라운드 대비 실측 일치도가
전문의 등급 기준 +0.139에서 +0.422로, 장비 스팟 개수 기준 +0.208에서 +0.578로 올랐다.
대신 색소 점수의 의미가 톤이 어둡다에서 반점이 많다로 좁아졌다.

**B-4. 항상 0만 내놓던 특징이 있었다.** 홍조 계산에 쓰던 hemoglobin 값은 알고리즘 특성상
평균이 정확히 0이 될 수밖에 없다(965명 전체에서 최대 1.5e-12). 그런 값이 홍조 공식의 30%를
차지하며 신호를 희석하고 있었다. 제거했다.

**B-5. 지표는 그 지표를 실제로 잰 부위에서만 재야 한다.** 모공 장비는 좌우 볼에서만 세고
이마와 코와 턱에서는 센 적이 없다. 그래서 모공은 볼에서만 집계한다. 다른 부위를 섞으면
실측 일치도가 떨어진다(볼만 +0.591, 볼+코 +0.575, 5부위 전부 +0.545). T존이 신호가 없어서가
아니라 — 코만 써도 +0.391이 나온다 — 볼에서 잰 라벨에 대고 채점하기 때문이다. 부위 제한은
`composite.<지표>.rois`로 지표마다 따로 선언한다.

**B-6. 좋아졌을 때만 채택하는 게이트를 뒀다.** 새로 피팅한 가중치는 검증셋에서 기존보다
최소 +0.05 좋아야 쓴다. 지도학습(릿지 회귀)도 마찬가지다. 색소는 r=+0.302에 오차 개선
2.4%로 탈락했다. 특히 디지털카메라로 학습한 모델을 폰 사진에 쓰면 오차가 그냥 평균값
답하기보다 217% 나빴다. 사용자가 어떤 기기로 찍을지 모르는 서비스에서는 쓸 수 없다는
뜻이다. 성능이 안 되는 모델이 구체적인 숫자를 찍으면 아는 게 없으면서 아는 척하는 꼴이
된다. 탈락은 실패가 아니라 설계된 동작이다. 반대로 모공은 이 게이트를 통과했고(r=+0.488,
오차 13.8% 개선), **세 기기 전부에서** 개선됐다(폰 +9.4%, 태블릿 +10.8%, DSLR +21.3%).

**B-7. 시도했다가 버린 것도 근거를 남겼다.** 얼굴 크기 정규화를 끄고 원본 해상도로 보는
방법은 오히려 나빠졌다(+0.053에서 +0.003). 폰의 노이즈 제거와 샤프닝 처리가 피부와
무관한 가짜 텍스처로 읽히기 때문이다. 옆모습 사진 추가도 정면 재보정에 못 미쳐 기각했다.
모공 공식에 특징을 4개째 넣는 것도 +0.001에 그쳐 3개에서 멈췄다.

### 3부. 무엇이 좋아졌는가

고친 결과는 점수의 분포에서 먼저 드러난다.

보정 전 점수에는 변별력이 거의 없었다. 검증용 321명 중 홍조는 215명이 최상위 10%에 몰렸다.
지금은 세 지표 모두 0~100에 고르게 퍼진다.

실측과의 순위 일치도(1.0이 완벽)는 다음과 같다.

| 지표 | 실측 기준 | 처음 | 현재 |
|---|---|---|---|
| 모공 | 장비 모공 개수 | — | **+0.575** |
| 색소침착 | 장비 스팟 개수 | +0.168 | **+0.578** |
| 색소침착 | 전문의 등급 | +0.096 | **+0.422** |
| 홍조 | 이 코호트에 실측 장비값 없음 | 분포만 정규화 | 분포만 정규화 |

가장 강한 지표는 모공이다. 특징 하나(`lbp_uniformity`)만으로 -0.581이 나오고, 세 기기에서
모두 같은 방향이다(폰 -0.563, 태블릿 -0.567, DSLR -0.650). 이유는 단순하다. **모공은 사진에
실제로 찍힌다.** 1mm가 안 되는 구멍이지만 주변 표면과 빛을 다르게 산란시키므로 고주파
텍스처로 그대로 드러난다.

나이를 맞히고 있는 것 아니냐는 의심은 실측으로 걸러냈다. 나이를 통계적으로 제거해도
-0.581 중 -0.557이 남는다. 즉 나이가 아니라 모공을 보고 있다.

> 여기 있던 **수분력 지표는 모공으로 교체됐다.** 수분은 두 라운드에 걸쳐 -0.028에서
> +0.320까지 끌어올렸지만 거기가 한계였다. 수분 장비(Corneometer)는 각질층의 전기 용량을
> 재는데 그 물리량은 표면 광학에 거의 남지 않기 때문이다. 같은 볼, 같은 텍스처 특징으로
> 모공을 재면 +0.575가 나온다. 잴 수 있는 것을 재는 쪽을 택했다. 함께 검토한 탄력
> 지표(Cutometer R2)는 채택하지 않았다. 이미지 특징으로 +0.329까지는 가지만 나이 하나가
> -0.434로 더 잘 맞히고, 나이를 제거하면 +0.171로 무너진다. 사실상 나이 추정기다.

홍조는 이 데이터셋에 대응하는 실측 장비값이 없어 순위 검증은 아직이며, 코호트 분포 기준의
상대 비교로 제공한다. 실측 라벨이 확보되는 즉시 붙일 수 있도록 보정 코드에 자리를 만들어
두었다.

처음의 문제로 돌아가면, 폰과 태블릿이 같은 얼굴을 다르게 찍는다는 사실은 카메라가 바뀌어도
살아남는 특징만 골라내는 방식으로 넘었다. 각 단계의 구현과 수치 근거는 아래 기술 섹션에
이어진다.

---
## 설계 개요

파이프라인은 다섯 단계다.

```
1. calibration   색 보정
2. detection     얼굴 검출, ROI 분할, 마스킹
3. features      물리 지표 계산
4. scoring       정규화와 점수화
5. SkinReport    결과 스키마
```

- **의존성 분리**: 무거운 패키지(`mediapipe`, `torch`, `timm`, `albumentations`)는
  `pyproject.toml`의 optional extra(`detection`, `dl`)로 분리하고 코드에서 지연 import 한다.
  덕분에 Phase 1 로직과 단위 테스트는 코어 의존성만으로 실행된다.
- **색공간 규약**: 모든 이미지는 `(H, W, 3)` float RGB다. `linear`은 scene-linear `[0,1]`,
  `sRGB`는 디스플레이 인코딩 `[0,1]`이며 파이프라인 내부 연산은 선형 RGB에서 이뤄진다.
- **방어적 수치 계산**: 0으로 나누기, 음수나 0에 대한 log는 전 함수에서 epsilon clip으로
  방어한다.
- **신뢰도 전파**: 색 보정 방식, 유효 ROI 비율, 헤모글로빈 분리 성공 여부가 `confidence`로
  전파된다.

### 디렉토리

```
skin_metrics/
├── calibration/color.py     # sRGB 선형화, 화이트밸런스, CCM, D65 CIELab
├── detection/face.py        # MediaPipe FaceMesh, 5 ROI, 아티팩트 마스킹
├── features/
│   ├── pigmentation.py      # ITA, 멜라닌 지수, 반점 검출, 톤 균일도
│   ├── erythema.py          # 홍반 지수, a*, Tsumura 헤모글로빈 ICA
│   └── pores.py             # GLCM과 LBP, 스케일링, 광택, 미세주름 (표면 텍스처)
├── scoring/
│   ├── normalize.py         # 경험적 분위수 백분위, 지도학습 예측 적용, compare()
│   └── schema.py            # pydantic SkinReport, MetricScore, FaceScale
├── calibrate/               # 오프라인 보정 (런타임에서 import 안 됨)
│   ├── aihub.py             # AI-Hub 028 코퍼스 인덱싱 (이미지와 실측 라벨 조인)
│   ├── extract.py           # 멀티프로세스 재개가능 특징 추출, CSV 출력
│   └── fit.py               # anchor, 릿지, 분위수 격자 피팅 + 채택 게이트
├── models/                  # Phase 2: dataset(+dummy), network, train
├── api/                     # HTTP API (FastAPI)
│   ├── app.py               # /healthz, /analyze 엔드포인트 + lifespan
│   ├── fetch.py             # 이미지 URL 다운로드 (SSRF, 크기 가드)
│   ├── schemas.py           # 요청과 응답 pydantic 모델
│   └── settings.py          # SKIN_METRICS_API_* 환경변수 설정
├── pipeline.py              # 이미지 입력에서 SkinReport 출력까지 오케스트레이션
├── config.py / config.yaml  # 설정 로더(2파일 병합), 사람이 관리하는 임계값과 정책
├── calibration_profile.yaml # 생성 파일: 피팅된 anchor, 레퍼런스, 지도학습 계수
└── cli.py                   # typer: analyze, compare, train, serve, calibrate
tests/                       # 합성 이미지, 랜드마크, 테이블 기반 단위 테스트
Dockerfile                   # 멀티스테이지: api(기본), full(Phase 2 포함)
docker-compose.yml           # 로컬 실행 + trainer 프로파일
.dockerignore                # deny-all + allow-list (로컬 사진과 리포트 유출 차단)
```

---

## Phase 1: 물리 기반 파이프라인 (기술 상세)

### 1. 색 보정, `calibration/color.py`

| 함수 | 내용 |
|---|---|
| `linearize_srgb(img)` | sRGB EOTF 역변환(감마 제거). `s<=0.04045 ? s/12.92 : ((s+0.055)/1.055)^2.4` |
| `encode_srgb(lin)` | 역변환(선형에서 sRGB로). 왕복 오차 1e-4 미만 |
| `white_balance_grayworld(img, mask)` | 채널 평균을 회색으로 정규화. `mask`로 게인 추정에 쓸 픽셀 제한. 약한 fallback이라 `success=False` |
| `white_balance_from_reference(img, bbox)` | 프레임 내 중립 패치(그레이카드, 흰 종이) 평균으로 채널 게인 산출. 너무 어둡거나 클리핑되면 거부 |
| `estimate_ccm(detected, reference)` | 24패치 컬러체커로 최소제곱 3x3 색보정 행렬 산출. `M = lstsq(detected, reference)`, RMS 잔차 반환 |
| `apply_ccm(img, M)` | `rgb @ M` 적용 |
| `rgb_to_lab(img)` | colour-science로 sRGB primaries와 D65 whitepoint 기준 CIELab. `L*`는 0에서 100 |
| `calibrate_image(...)` | 오케스트레이션: 선형화, CCM, WB 순. `status`는 `reference`, `grayworld`, `none` 중 하나이고 `success`를 함께 반환 |

- **보정 신뢰도 규약**: `reference`(중립 패치 성공)이거나 CCM을 적용하면 `success=True`,
  그 외에는 `False`다. 이 값이 downstream `confidence`를 낮춘다.

#### 인물 사진에 gray-world를 쓰면 안 되는 이유

gray-world는 장면 전체 평균이 무채색이라는 가정이다. 얼굴이 프레임을 채우면 장면 평균이 곧
피부색이므로, 게인을 적용하는 순간 피부의 색도가 나눠져 사라진다. 실측 코호트에서 확인된
결과다.

| WB 모드 | ROI 평균 a* | ROI 중앙값 b* | ITA |
|---|---|---|---|
| `grayworld` (구 기본값) | **0.5** | **-1.2** | **93도** (90도에서 포화) |
| `none` (카메라 AWB) | 15.9 | 26.6 | 39.8도 |

a*(홍조)와 b*(황색)가 0으로 붕괴하면 ITA = `atan2(L*-50, b*)`가 90도에서 포화하고 부호가
무작위로 뒤집힌다. 코호트 90명 기준, 전문가 색소 등급과의 상관은 다음과 같다.

| `calibration.fallback` | spearman(-ITA, 등급) | ITA 평균과 표준편차 | Fitzpatrick 분포 (타입 1~6) |
|---|---|---|---|
| `none` (**현재 기본값**) | **+0.437** | 31.8, 6.4 | [0, 0, 55, 35, 0, 0] |
| `background` | +0.367 | 38.6, 10.4 | [4, 13, 54, 19, 0, 0] |
| `grayworld` | **-0.307** | 6.5, **83.1** | [16, 3, 32, 10, 28, 1] |

gray-world는 노이즈를 더하는 정도가 아니라 색소 신호를 뒤집는다. 기본 fallback은
`none`(카메라 AWB 신뢰)이며, 한국인 코호트에서 유일하게 타당한 Fitzpatrick 분포(타입 3~4)를
낸다. 카메라 AWB가 감당 못 하는 강한 색 조명 환경에서는
`calibration.fallback: background`(비피부 픽셀만으로 gray-world 추정)가 대안이다.

- **정확도의 핵심 레버**는 여전히 그레이카드 `--reference-bbox` 또는 컬러체커 CCM이다.
- **배경 영향**: 지표 계산은 ROI 내부만 쓰므로 배경 누끼는 불필요하다.

### 1-b. 얼굴 크기 정규화, `pipeline._normalize_face_scale`

GLCM, LBP, 미세주름 밀도는 고정 픽셀 오프셋에서 계산되므로, 얼굴이 차지하는 픽셀 수가
다르면 값을 비교할 수 없다. 코호트 기기별 눈 사이 거리(외안각)는 다음과 같다.

| 기기 | 평균 eye-span | 해상도 |
|---|---|---|
| 디지털카메라 | 1140 px | 2136x3216 |
| 스마트패드 | 972 px | 2448x3264 |
| 스마트폰 | 889 px (중앙값) | 1920x2560 |

파이프라인은 특징 추출 전에 얼굴을 eye-span 512px로 맞춘다
(`normalization.target_eye_span_px`).

- **다운스케일 전용**: 더 작은 얼굴은 그대로 둔다. 업샘플링은 없는 디테일을 만들어내지
  않으면서 텍스처만 매끄럽게 만들어 모공이 거짓으로 적어 보이게 하기 때문이다. 대신
  `under_resolved` 플래그가 서고, 경고가 붙고, 모공 confidence가 0.6배가 된다.
- **선형 광에서 리샘플**: sRGB 인코딩 상태로 축소하면 감마 곡선을 타고 ROI 평균 색이
  편향된다.
- 부수 효과로 36MP 이미지 분석이 약 11초에서 3.6초로 빨라졌다.

### 2. 얼굴과 ROI 검출, `detection/face.py`

- **랜드마크**: MediaPipe FaceMesh 468점. 두 API를 모두 지원한다.
  - 레거시 `mp.solutions.face_mesh` (MediaPipe 0.10.x)
  - Tasks API `FaceLandmarker` (MediaPipe 1.0 이상). `face_landmarker.task` 모델 필요
- **모델 파일 처리**:
  - `resolve_face_model()`: `model_path` 인자, `SKIN_METRICS_FACE_MODEL` env,
    `~/.cache/skin_metrics/face_landmarker.task` 순으로 탐색
  - `ensure_face_model()`: 없으면 Google 저장소에서 약 3.8MB 다운로드
  - CLI `--download-model` 플래그로 첫 실행 시 자동 다운로드
- **5 ROI**: 이마, 좌볼, 우볼, 코, 턱. 각 ROI는 선별된 내부 랜드마크의 convex hull을
  `cv2.fillConvexPoly`로 래스터화한다(정확한 경계 loop 불필요, 합성 데이터로도 테스트 가능).
- **아티팩트 마스킹 `mask_artifacts(...)`**:
  - 정반사와 글레어: sRGB에서 HSV로 변환 후 `V > glare_v_min(0.92)` 이면서
    `S < glare_s_max(0.15)`
  - 그림자: ROI 내 `L*`의 하위 `shadow_percentile(5%)` 컷
  - 모발, 눈썹, 입술: 랜드마크 기반 `exclusion_mask()`(눈, 눈썹, 입술 폴리곤을 dilate하여 제외)
- **유효 픽셀 게이트**: `valid/region < min_valid_ratio(0.60)`이면 해당 ROI를 `None` 처리한다.

### 3. 물리 지표, `features/`

**색소침착 `pigmentation.py`**

- `ita(L, b) = arctan((L-50)/b)*180/pi`. `b`가 0에 가까우면 부호 보존 epsilon으로 방어
- `melanin_index(R_red) = 100*log10(1/R_red)`. `R`을 `[eps,1]`로 clip
- `spot_detection(L, mask, sigma=15)`: 마스크 정규화 가우시안 국소 평균 대비 음의 편차가
  임계(`contrast_thresh`) 이상이면 반점. 연결성분(skimage `label`)으로 면적률, 개수,
  평균 대비도 산출
- `evenness(L, mask)`: ROI 내 `L*` 표준편차(톤 균일도)
- `estimate_fitzpatrick(ita, boundaries)`: ITA 컷오프(Del Bino)로 타입 1~6 추정

**홍조 `erythema.py`**

- `erythema_index(R_r, R_g) = 100*(log10(1/R_g) - log10(1/R_r))`
- `mean_a_star(lab, mask)`: CIE `a*` 평균과 90퍼센타일
- `hemoglobin_map(rgb, mask)`: Tsumura 색소분리
  1. 광학 밀도 `-log(reflectance)`로 변환
  2. FastICA(n=2)로 멜라닌과 헤모글로빈 2성분 분리
  3. ICA mixing 열벡터와 기준 흡광 방향의 코사인 유사도로 헤모글로빈 성분 식별. 부호는
     헤모글로빈이 올라가면 값도 올라가도록 결정(순서와 부호 모호성 해소)
  4. 퇴화 입력(픽셀 부족, 상수)은 `separation_ok=False`로 안전 반환

**모공 `pores.py`**: 표면 텍스처 특징 모음이다. 모공은 사진에 실제로 찍히는 구조이므로
간접 추정이라는 단서가 붙지 않는다(같은 함수들이 예전에 수분력을 추정할 때는 붙어 있었다).

- `texture_stats`: GLCM(contrast, correlation, energy)과 LBP 균일도(정수 양자화).
  이 중 `lbp_uniformity`가 시스템 전체에서 가장 강한 단일 특징이다(held-out -0.581)
- `scaling_index`: Laplacian 고주파 에너지(표면 요철)
- `specular_ratio`: 정반사 픽셀 비율(광택). 추출은 하지만 모공 공식에는 안 들어간다
- `micro_wrinkle_density`: Frangi와 Hessian ridge 필터로 선형 구조 밀도 산출. 모공은 둥근
  구멍이고 이 필터는 그런 걸 억제하도록 만들어졌으므로 역시 공식 밖이다

모공 공식은 셋을 쓴다: `lbp_uniformity` -0.78, `scaling_index` +0.13, `glcm_contrast` -0.09.
가중치가 음수인 것은 버그가 아니라 suppressor다(자세한 것은 아래 보정 섹션).

### 4. 정규화와 점수화, `scoring/`

- **집계**: 파이프라인이 유효 ROI별 지표를 유효 픽셀 수 가중 평균으로 얼굴 단위 집계한다.

점수는 항상 score driver를 레퍼런스 분포의 백분위로 변환해 나온다. driver를 얻는 경로가 두
가지다.

| 경로 | 조건 | driver |
|---|---|---|
| **보정됨** (calibrated) | `supervised.<metric>` 모델 존재 | 릿지 회귀가 예측한 실측 장비값 (`score_sign`으로 방향 정렬) |
| 미보정 (fallback) | 모델 없음 | anchor로 z-score한 서브피처의 가중합 |

- **`predict_instrument(model, roi_features)`**: 모델이 학습된 ROI에만 적용된다
  (`applies_to_rois`). 코 부위 특징으로 볼에서 잰 모공 개수를 예측하는 건 외삽이므로 제외한다.
  얼굴 단위 값은 부위별 예측의 단순 평균이다. 장비가 부위 면적과 무관하게 부위당 1회씩
  측정했으므로, 점수를 조회할 레퍼런스 분포도 같은 방식으로 만들어야 한다(면적 가중을 하면
  다른 분포에 대고 조회하게 된다).
- **`score_from_raw(raw, metric, fitz, config)`**: 해당 Fitzpatrick 타입 레퍼런스의 경험적
  분위수 격자(0~100, 101점)에서 선형 보간해 백분위를 산출한다. 격자가 없으면 기존 정규 CDF로
  폴백한다. 스팟 개수와 예측 등급 분포는 눈에 띄게 치우쳐 있어서 가우시안 가정은 꼬리를
  잘못 배치한다.
- **`compare(current, baseline, min_delta)`**: 지표별 변화량, 방향, 유의 여부(시계열).
- **출력 스키마 `schema.py`** (pydantic):

```python
class MetricScore(BaseModel):
    score: float                  # 0-100 condition index (높을수록 뚜렷)
    confidence: float             # 0-1
    raw_features: dict
    is_estimate: bool             # 현재 세 지표 모두 False (전부 실제로 찍히는 것)
    calibrated: bool              # 실측 라벨로 학습된 모델이 낸 점수인가
    predicted_value: float | None # 예측된 실측 장비값 (보정된 경우)
    predicted_units: str | None   # 예: "count", "grade 0-5"

class SkinReport(BaseModel):
    pigmentation / erythema / pores: MetricScore
    roi_breakdown: dict            # ROI별 valid_ratio 와 지표
    calibration_status: Literal["reference","grayworld","none"]
    fitzpatrick_estimate: int      # 1-6
    face_scale: FaceScale          # eye_span_px, scale_factor, under_resolved
    calibration_profile: str | None  # 어떤 코호트로 보정됐는지
    warnings: list[str]
    disclaimer: str                # 의료기기 아님 고지 (자동 포함)
```

- **confidence 계산**(`pipeline.py`): `calibration(reference 1.0, grayworld 0.6, none 0.6)`에
  `(유효 ROI 수 / 5)`를 곱한다. 홍조는 헤모글로빈 분리 실패 시 0.7배, 모공은
  `under_resolved`면 0.6배가 된다. `none`은 더 이상 실패 경로가 아니라 기본 경로이고
  레퍼런스 코호트도 이 조건에서 촬영하고 피팅했기 때문에 grayworld보다 불리하게 두지 않는다.

---
## Phase 2: 딥러닝 (기술 상세)

> 상태: 완전히 도는 스캐폴드다. 실측 라벨이 있어야 의미 있는 절대값을 낸다. 라벨이 없으면
> 더미 데이터로 학습 루프 전체를 즉시 검증할 수 있다. `dl` extra가 필요하다.

### `models/dataset.py`

- **`Sample`**: ROI 크롭 이미지, Phase 1 물리 피처 벡터(12차, `PHYSICS_FEATURE_NAMES`),
  멀티태스크 라벨(3개), Fitzpatrick, 조명 버킷으로 구성된다.
- **`DummyLabelGenerator`**: 라벨을 물리 벡터의 (노이즈 섞인) 결정적 함수로 생성하므로 모델이
  실제로 학습 가능하고, 라벨과 이미지 파일 없이 루프를 검증할 수 있다. 조명 버킷이 이미지
  밝기에 반영되어 도메인 적대 헤드에 신호를 준다.
- **`SkinDataset.from_csv`**: 실측 라벨 CSV 로더다(누락 타깃은 NaN이라 ranking 모드와 호환).
- **Augmentation**(albumentations 2.x): 색상 변환은 공격적으로(ColorJitter, 밝기와 대비,
  RGBShift, GaussNoise, JPEG 압축), 기하 변환은 약하게(flip, mild Affine) 건다.

### `models/network.py`

- **백본**: timm `efficientnet_b0` (`pretrained` 옵션. 오프라인과 테스트에서는 `False`)
- **물리 브랜치**: `PhysicsMLP`(Linear, ReLU, LayerNorm, Linear)로 임베딩한 뒤 백본 특징과
  concat
- **헤드 3개**: 색소, 홍조, 모공 회귀. homoscedastic uncertainty weighting
  (`L = sum(exp(-s_i)*L_i + s_i)`, Kendall 2018)으로 멀티태스크 loss를 균형 잡는다
- **조명 불변성**: Gradient Reversal Layer와 조명 버킷 분류기(domain-adversarial)로 공유
  특징이 조명에 불변하도록 학습한다

### `models/train.py`

- **손실**:
  - `mode="regression"`: Huber(Smooth-L1)에 uncertainty weighting (이상치에 강건)
  - `mode="ranking"`: pairwise margin ranking loss (절대값 대신 A가 B보다 건조하다는 쌍 비교)
  - 도메인 적대 cross-entropy 추가
- **검증 지표**: MAE, Pearson r, Spearman rho, 그리고 반드시 Fitzpatrick 타입별 분리 리포트
- **엔트리포인트**: `run_training(data_csv, config, mode, epochs, use_dummy, pretrained, ...)`

---

## 레퍼런스 보정 (`skin_metrics/calibrate/`)

0~100 점수가 의미를 가지려면 실제 사람들의 분포가 필요하다. 그 분포와 지도학습 모델은 AI-Hub
공개 데이터셋 《028. 한국인 피부상태 측정 데이터》로 피팅했다.

| 항목 | 규모 |
|---|---|
| 피험자 | 965명 (train 858, val 107. 피험자 단위로 분리) |
| 이미지 | 정면 2,895장 (3기기 x 965명) |
| ROI 행 | 11,580 (이마, 좌우볼, 턱) |
| 실측 라벨 | 장비 모공 개수, 전문가 색소와 모공과 주름 등급, 장비 스팟 개수, 수분 측정 장비(Corneometer) 값, Cutometer 탄력 |

```bash
# 1) 코호트 전체에서 물리 특징 추출 (재개 가능, 약 27분 / 6워커)
skin-metrics calibrate extract --data-root "028. 한국인 피부상태 측정 데이터" --workers 6

# 2) anchor, 레퍼런스 분포, 지도학습 모델 피팅 후 calibration_profile.yaml 기록
skin-metrics calibrate fit --dry-run     # 검증 수치만 출력
skin-metrics calibrate fit               # 프로파일 기록
```

### 설정 파일이 둘인 이유

| 파일 | 성격 |
|---|---|
| `config.yaml` | 사람이 관리하는 임계값, 정책, 특징 선택과 그 근거(주석). 기계가 덮어쓰지 않는다. 여기 적힌 composite 가중치는 피팅이 채택되지 않았을 때의 fallback이다 |
| `calibration_profile.yaml` | 생성 파일. anchor, composite 가중치, 레퍼런스 분위수 격자, 지도학습 계수, 검증 수치 |

`load_config()`가 둘을 병합한다. 프로파일이 없어도 파이프라인은 그대로 동작한다(미보정
composite 경로).

### 채택 게이트, 두 종류

**composite 가중치**: 피팅한 가중치가 held-out에서 `config.yaml`의 선언된 가중치보다
Spearman 기준 +0.05 이상 좋을 때만 채택한다. 현재 색소는 채택됐고(+0.540에서 +0.625), 모공은
기각됐다(+0.576 대 +0.575). 모공은 `config.yaml`에 선언된 가중치가 이미 피팅 결과와
동률이라(부분집합 전수 탐색으로 고른 값이라 당연하다) 게이트가 선언값을 유지한다.

> `fit_composite_weights`에는 `sign` 인자가 있고 `COMPOSITE_TARGET_SIGN`을 반드시 넘겨야
> 한다. 현재 두 지표 모두 sign이 +1이라 티가 안 나지만, 장비가 척도의 *좋은* 쪽을 재는
> 지표(예전의 수분력)에서는 이걸 빼먹으면 **항상 부호가 뒤집힌 가중치**가 나오고 게이트가
> 그걸 "반상관"으로 오해해 기각한다. 조용히 실패하므로 테스트로 고정해 뒀다.

**지도학습 모델 (실측 장비값 예측)**: held-out 30행 이상, |pearson| 0.25 이상, MAE가 평균
예측 대비 5% 이상 개선일 때만 채택한다. 평균값만 예측하는 것과 별 차이 없는 모델이 리포트에
"모공 820개" 같은 구체적 숫자를 찍으면, 개인에 대한 정보가 거의 없는데도 있는 것처럼
보이기 때문이다. 현재 모공은 이 게이트를 통과했고(r=+0.488, MAE 265.8 대 308.5 = 13.8%
개선) 색소는 탈락했다(r=+0.302, 2.4% 개선).

어느 쪽이든 탈락은 오류가 아니다. 해당 항목은 선언된 값이나 코호트 백분위 경로로 돌아가고,
사유가 프로파일의 `validation`과 `validation_weights`에 기록된다.

### 이게 실제로 무엇을 고쳤나 (held-out 321명)

보정 전후로 동일한 held-out 이미지를 점수화한 분포다. 백분위 매핑이 올바르면 십분위마다 약
32명씩 균등해야 한다.

| 지표 | 보정 후 | 보정 전 (placeholder anchor와 reference) |
|---|---|---|
| pigmentation | 평균 48.6, 십분위 23~41 | 평균 41.3, 십분위 **0~79**, 10.7~84.4 |
| erythema | 평균 49.8, 십분위 21~43 | 평균 **90.3**, **321명 중 215명이 최상위 십분위** |
| pores | 평균 51.4, 십분위 22~42, 0.1~99.5 | (당시 지표가 아님) |

보정 전 점수는 변별력이 거의 없었다. 특히 홍조는 사실상 모두가 매우 심함으로 나왔다.
모공은 이 라운드 이후에 들어온 지표라 보정 전 값이 없다.

### 실측 일치도, 배포중인 점수가 실제로 얼마나 맞나

앞의 십분위 표는 분포가 맞다는 뜻이지 순위가 맞다는 뜻이 아니다. 실측과의 순위
일치도(Spearman)는 별도로 측정해야 한다. held-out 321명 기준이다.

| 지표 | 대상 | 최초(placeholder) | 1차 보정 | **현재** |
|---|---|---|---|---|
| pores | 장비 모공 개수 | (지표 없음) | (지표 없음) | **+0.575** |
| pigmentation | 장비 스팟 개수 | +0.168 | +0.208 | **+0.578** |
| pigmentation | 전문가 색소 등급 | +0.096 | +0.139 | **+0.422** |
| erythema | 실측 라벨 없음 | 측정 불가 | 측정 불가 | 측정 불가 |

모공은 held-out 볼 642행 기준이다(집계 부위가 볼이므로 거기서 재는 게 맞다). 기기별로는
폰 +0.558, 태블릿 +0.576, 디지털카메라 +0.635다. 사람 단위로 좌우 볼을 평균내면 +0.591이다.
지도학습 릿지가 게이트를 통과해 실제 배포 경로는 릿지이며, 순위 성능은 composite과
사실상 동률이다(+0.572 대 +0.575).

색소도 기기별로 일관된다. 등급 기준 디지털카메라 +0.425, 태블릿 +0.554, 폰 +0.329이며,
재가중 전에는 **-0.011**, +0.313, +0.481로 기기마다 제각각이었다.

#### 색소 composite에서 절대 색상 특징을 뺀 이유

가장 큰 개선은 anchor 재추정이 아니라 특징 선택에서 나왔다. 장비 스팟 개수와의 순위 일치도를
기기별로 보면 이렇다.

| 특징 | 전체 | 디지털카메라 | 태블릿 | 폰 | 기존 가중치 |
|---|---|---|---|---|---|
| `spot_count` | **+0.578** | +0.595 | +0.726 | +0.627 | **없었음** |
| `spot_area_ratio` | +0.434 | +0.329 | +0.495 | +0.592 | 0.30 |
| `spot_mean_contrast` | +0.362 | +0.574 | +0.380 | +0.244 | 없었음 |
| `evenness` | +0.247 | +0.356 | +0.358 | +0.486 | 0.20 |
| `ita` | -0.258 | -0.676 | -0.236 | -0.210 | 0.10 |
| `melanin_index` | +0.050 | +0.519 | **-0.085** | +0.031 | **0.40** |

절대 색상(멜라닌, ITA)은 카메라를 넘으면 부호까지 뒤집힌다. 카메라별 색 재현과 센서 노이즈가
절대값을 옮기기 때문이다. 형태학 특징(반점 개수, 면적, 대비, 균일도)은 세 기기 모두에서
유지된다. 그런데 기존 composite은 가장 불안정한 `melanin_index`에 가중치 40%를 주고 가장
강한 `spot_count`는 아예 빼놓고 있었다.

형태학 특징만 남기고 가중치를 코호트에서 피팅한 결과가 위 표의 현재 값이다. 그레이카드나
컬러체커 보정(`--reference-bbox`)을 쓰면 절대 색상이 기기 독립적이 되므로 그때는 다시 넣을
가치가 있다.

> 이 때문에 색소 점수의 의미가 전반적 톤 어두움 포함에서 반점 부담으로 좁아졌다. 피부가
> 전반적으로 어두운 것 자체는 더 이상 점수를 올리지 않는다.

### 지도학습 모델 검증 결과 (held-out, 피험자 분리)

| 지표 | 타깃 | pearson | MAE (vs 평균 예측) | 채택 |
|---|---|---|---|---|
| pores | 장비 모공 개수 | +0.488 | 265.8 vs 308.5 (+13.8%) | **예** |
| pigmentation | 전문가 색소 등급 0-5 | +0.302 | 0.958 vs 0.981 (+2.4%) | 아니오 (5% 미달) |
| erythema | 이 코호트에 실측 장비값 없음 | 해당 없음 | 해당 없음 | 코호트 백분위만 |

모공만 통과했다. 통과 사유가 중요한데, **세 기기 전부에서** 평균 예측을 이겼기 때문이다
(폰 +9.4%, 태블릿 +10.8%, 디지털카메라 +21.3%). 아래에서 보듯 색소 모델을 무너뜨린 것이
바로 기기 전이였는데, 텍스처 특징에는 그 문제가 없다. 색소와 홍조는 코호트 백분위 경로로
점수가 나오고, 탈락 사유는 `calibration_profile.yaml`의 `validation` 블록에 기록된다.

> 다만 예측된 **개수 자체는 순위만큼 믿을 게 못 된다**. 평균 예측보다 13.8% 나은 수준이라
> 실제 62개인 사람에게 629개를, 1737개인 사람에게 1209개를 찍는 식으로 평균 쪽으로
> 수축한다(실사진 검증값). 그래서 리포트에 경고가 붙고, HTTP API는 0~100 점수만 내보내며
> `predicted_value`는 CLI 리포트에만 남는다.

#### 왜 탈락했나, 기기 종속성

기기별로 따로 피팅하면 색소 모델은 잘 작동한다. 그런데 기기를 넘으면 무너진다.

| 학습 (아래) / 평가 (오른쪽) | 디지털카메라 | 스마트패드 | 스마트폰 |
|---|---|---|---|
| 디지털카메라 | r=+0.600 **+17.4%** | r=+0.317 **-114.8%** | r=+0.125 **-217.4%** |
| 스마트패드 | r=+0.148 -56.1% | r=+0.488 **+9.6%** | r=+0.213 -7.4% |
| 스마트폰 | r=+0.353 +2.4% | r=+0.396 -15.4% | r=+0.357 +3.4% |
| 풀링(전체) | r=+0.453 +3.4% | r=+0.397 +0.8% | r=+0.311 +2.9% |

디지털카메라로 학습한 모델을 폰 사진에 적용하면 평균만 답하는 것보다 MAE가 217% 나쁘다.
카메라별 색 재현과 센서 노이즈가 절대 특징값을 옮기기 때문이다. 그래서 기본 프로파일은
풀링이고, 색소는 지도학습 모델 없이 간다.

이 문제는 **색상 특징을 쓰는 지표에만** 해당한다. 모공은 같은 풀링 프로파일에서 세 기기
모두 개선을 내고(위 표) 기기별 순위 일치도도 +0.558에서 +0.635 사이로 좁다. 절대 색상은
카메라를 넘지 못하고 텍스처는 넘는다는 것이, 이 프로젝트에서 반복해서 확인된 사실이다.

> 촬영 기기가 고정된 배포라면 `skin-metrics calibrate fit --device digital_camera`로 기기
> 전용 프로파일을 만들어 색소 모델(+17.4%)을 활성화할 수 있다. 그 프로파일을 다른 기기
> 사진에 쓰면 안 된다.
>
> 기기 간 전이를 되살리는 정공법은 그레이카드나 컬러체커 보정이다(`--reference-bbox`).
> 절대 색값을 기기 독립적으로 만들어 주기 때문이다.

### 모공: 이 시스템에서 가장 강한 지표

모공은 세 지표 중 실측 일치도가 가장 높다(held-out +0.575). 이유는 기법이 아니라 물리다.
모공은 지름 1mm가 안 되는 구멍이지만 주변 표면과 빛을 다르게 산란시키므로 **사진에 실제로
찍힌다**. 색소 반점과 마찬가지로 형태 특징이고, 그래서 카메라를 넘어서도 살아남는다.

**집계 부위는 볼뿐이다.** 이 코호트에서 모공 장비는 좌우 볼(facepart 5, 6)에서만 셌다.
사람 단위 held-out 일치도를 집계 부위만 바꿔 가며 재면 이렇다.

| 집계 부위 | 일치도 |
|---|---|
| **볼만 (채택)** | **+0.591** |
| 볼 + 코 | +0.575 |
| 볼 + 이마 | +0.573 |
| 5부위 전부 | +0.545 |
| 코만 | +0.391 |
| 이마만 | +0.357 |

T존에 신호가 없어서가 아니다. 코만 써도 +0.391이 나온다. 모공 부담은 얼굴 전체에 걸쳐
상관되기 때문이다. 그런데도 볼만 쓰는 이유는 채점 기준이 볼에서 잰 라벨이고, anchor와 분위수
격자도 전부 볼 통계이기 때문이다. 보정된 부위 밖으로 나가면 그 눈금이 의미를 잃는다.

**특징 세트는 전수 탐색으로 골랐다.** 추출된 모든 특징(색소, 홍조 특징 포함)을 후보로 놓고
볼에서 부분집합을 전수 탐색했다. train에서 고르고 held-out에서 채점했다.

| 세트 | held-out |
|---|---|
| `lbp_uniformity` 단독 | **-0.581** |
| 아래 3개 (채택) | **+0.575** |
| 최적 4개 세트 | +0.576 |

네 번째 특징이 +0.001밖에 못 벌어서 3개에서 멈췄다. 색소 특징은 후보에 있었지만 하나도
들어오지 않았다. 이건 색상이 아니라 텍스처 측정이고, 그래서 기기가 바뀌어도 버틴다.

**`lbp_uniformity` -0.78, `scaling_index` +0.13, `glcm_contrast` -0.09**

- `lbp_uniformity`는 텍스처가 복잡할수록 내려가므로 가중치가 음수다. 시스템 전체에서 가장
  강한 단일 특징이다.
- `glcm_contrast`는 suppressor다. `scaling_index`와 공유하는 조명성 분산을 상쇄한다. 단독
  성능이 약하다고 지우면 안 된다.

**나이 추정기가 아니라는 확인.** 텍스처 지표는 "나이 들어 보이는 정도"를 재고 있을 위험이
늘 있다. 실제로 나이와 모공 개수는 +0.240 상관이다. 그래서 나이를 통계적으로 제거하고 다시
쟀다.

| | 원래 | 나이 제거 후 |
|---|---|---|
| `lbp_uniformity` ↔ 장비 모공 개수 | -0.581 | **-0.557** |

거의 그대로 남는다. 모공을 보고 있다는 뜻이다. (비교: 함께 검토했던 탄력 지표는 원래 +0.278
에서 나이 제거 후 +0.171로 무너졌다. 나이 하나가 탄력을 -0.434로 더 잘 맞힌다. 그래서 탄력은
채택하지 않았다.)

**실사진 end-to-end 검증** — held-out 폰 사진 3장을 JPEG부터 파이프라인 전체에 통과시킨 결과:

| 실측 모공 개수 | 점수 | 예측 개수 |
|---|---|---|
| 62 | 12.1 | 629 |
| 910 | 47.4 | 922 |
| 1737 | 99.1 | 1209 |

순위는 정확하고 절대 개수는 평균으로 수축한다. 이 지표를 **점수로 쓰고 개수로는 쓰지 말라**는
경고가 리포트에 붙는 이유다.

> **여기 있던 수분력 지표는 모공으로 교체됐다.** 수분은 두 라운드에 걸쳐 버그 두 개(코까지
> 평균에 넣던 부위 문제, `COMPOSITE_TARGET_SIGN`을 피팅에 안 넘기던 부호 문제)를 고쳐
> -0.028에서 +0.320까지 갔지만 거기가 한계였다. Corneometer는 각질층의 전기 용량을 재는데
> 그 물리량은 표면 광학에 거의 남지 않기 때문이다. 같은 볼, 같은 텍스처 특징으로 모공을
> 재면 +0.575가 나온다. 그때 만든 인프라(지표별 `rois` 제한, `sign` 인자, 볼 전용 anchor)는
> 그대로 모공이 쓰고 있다.

---

## 점수 해석, 신뢰도, 보정

각 점수는 코호트 대비 0~100 백분위다.

| 지표 | 높은 점수 | 방향 | 비고 |
|---|---|---|---|
| pores | 모공 많음, 뚜렷함 | 높을수록 나쁨 | 물리 측정, 실측 일치도 **+0.58** |
| pigmentation | 색소침착 많음, 짙음 | 높을수록 나쁨 | 물리 측정, 실측 일치도 +0.63 |
| erythema | 홍조 강함 | 높을수록 나쁨 | 물리 측정, 코호트 백분위 기준 상대 비교 |

> **세 지표 모두 방향이 같다. 높을수록 나쁨이다.** 내부 계산도 전부 condition
> index(높을수록 뚜렷)이므로 뒤집기가 필요 없고, `scoring.report_inverted`는 비어 있다.
> 이 설정을 남겨 둔 이유는 척도의 *좋은* 쪽을 이름으로 삼는 지표가 다시 생기면 필요하기
> 때문이다. 예전의 `hydration`이 그런 경우였다(수분력 85점이 건조함을 뜻하면 UI가 사용자에게
> 정반대를 알려주게 된다). 그때도 뒤집기는 반드시 마지막에 일어나야 한다.
> `calibration_profile.yaml`의 검증 수치와 `calibrate/fit.py`의 `COMPOSITE_TARGET_SIGN`이
> 전부 "높을수록 뚜렷" 방향으로 되어 있기 때문이다.

- **`calibration_status`**: `reference`, `grayworld`, `none` 순으로 신뢰도가 높다.
  그레이카드나 흰 종이를 프레임에 넣고 `--reference-bbox x,y,w,h`로 지정하면 `reference`로
  올라간다.
- **`fitzpatrick_estimate`**: ITA로 자동 추정하며, 정규화가 타입별 레퍼런스로 수행된다.
- **촬영 조건**: 정면, 균일 조명, 맨얼굴, 얼굴이 프레임의 대부분을 차지하는 구도, 그림자와
  강한 광택 최소화. 이 조건에서 점수가 가장 안정적이다.

---
## CLI 레퍼런스

```bash
skin-metrics calibrate extract --data-root PATH [--out-dir DIR] [--workers N]
                               [--devices digital_camera,tablet,phone] [--limit N]
                               [--no-resume]
skin-metrics calibrate fit [--features-dir DIR] [--output PATH] [--device NAME]
                           [--profile-name NAME] [--dry-run]

skin-metrics analyze <image> [--reference-bbox x,y,w,h] [--model PATH]
                             [--download-model] [--output report.json] [--config cfg.yaml]
skin-metrics compare <img1> <img2> [--reference-bbox ...] [--output ...]
skin-metrics train [--data labels.csv | --dummy] [--mode regression|ranking]
                   [--epochs N] [--config ...]
skin-metrics serve [--host 127.0.0.1] [--port 8000] [--reload] [--download-model]
                   [--config cfg.yaml] [--allow-private-hosts]
```

- `--download-model`: 첫 실행 시 FaceLandmarker 모델(약 3.8MB)을 자동 다운로드하고 이후
  캐시를 재사용한다
- `--reference-bbox`: 중립 패치를 지정해 배경과 무관한 화이트밸런스를 적용하고 confidence를
  올린다

---

## HTTP API

`uv sync --extra api --extra detection` 후 `skin-metrics serve`로 띄운다. FastAPI와 uvicorn은
`api` extra에만 있으며, `skin_metrics.api`를 import 하지 않는 한 코어 동작에 영향이 없다.
OpenAPI 문서는 `/docs`, 스키마는 `/openapi.json`이다.

### 비동기 흐름 (Spring Boot 연동)

분석은 비동기다. POST는 `request_id`를 즉시 돌려주고, 분석이 끝나면 결과 JSON이 Redis의
`{request_id}:analyze` 또는 `{request_id}:diary` 키에 저장된다. Spring Boot는 그 키를
Redis에서 직접 읽어간다. 이 API를 다시 부를 필요가 없다.

```
1. Spring Boot 가 POST /analyze 호출
2. skin-metrics 가 202 {"request_id","redis_key"} 즉시 응답
3. skin-metrics 백그라운드에서 이미지 다운로드 후 분석
4. skin-metrics 가 Redis 에 SET {request_id}:analyze = {JSON} (TTL 1시간)
5. Spring Boot 가 Redis 에서 {request_id}:analyze 를 직접 GET
```

Redis 연결은 `SKIN_METRICS_REDIS_HOST`, `_PORT`, `_USER`, `_PASSWORD`, `_DB`, `_TLS`로 나눠
설정한다. URL 한 줄로 받지 않는 이유는 비밀번호에 `@`나 `/`가 들어가면 퍼센트 인코딩이
필요하고, 틀리면 원인과 무관한 인증 오류로 나타나기 때문이다. 자격증명이 들어가므로 저장소에
커밋하지 않고 compose 옆의 `.env`(gitignore됨)에 둔다.

Redis는 캐시가 아니라 결과 전달 경로 자체이므로 필수다. 미설정 시 API는 기동 단계에서
`RuntimeError`로 즉시 실패하고(`docker compose`는 그 전에 변수 미설정으로 멈춘다), 조용히
폴백하지 않는다. 폴백이 있으면 `.env`를 빠뜨렸을 때 API는 200을 주는데 Spring만 결과를 못
읽는 상태가 된다.

클론 직후에는 `.env.example`을 복사해서 채우면 된다.

```bash
cp .env.example .env   # Redis host/port/password를 실제 값으로 교체
```

### `POST /analyze` 와 `POST /analyze/diary`

요청 본문은 두 엔드포인트가 동일하다.

```jsonc
{
  "image_url": "https://example.com/face.jpg",   // 필수, http(s)
  "reference_bbox": [10, 10, 40, 40]             // 선택, [x, y, w, h] 중립 패치
}
```

응답은 202로 즉시 반환된다.

```jsonc
{
  "request_id": "470b634e92bd44b9abeb12accb0f0b70",
  "redis_key": "470b634e92bd44b9abeb12accb0f0b70:analyze"   // diary면 ...:diary
}
```

202라는 상태 코드 자체가 접수됨과 처리 중을 뜻하므로 본문에는 식별자만 담는다. 진행 상태는
저장 문서의 `status`(`processing` 다음 `done` 또는 `failed`)로 확인된다.

**Redis에 저장되는 문서** (JSON 문자열, TTL 기본 1시간):

```jsonc
// 처리 중
{ "status": "processing", "request_id": "...", "kind": "analyze", "submitted_at": "..." }

// 완료. /analyze 의 result (0~100)
{
  "status": "done", "request_id": "...", "kind": "analyze",
  "submitted_at": "...", "completed_at": "...",
  "result": {
    "pigmentation": 18.46,   // 높을수록 색소침착 많음
    "erythema": 41.62,       // 높을수록 붉음
    "pores": 47.40,          // 높을수록 모공 많음 (볼 기준)
    "confidence": { "pigmentation": 0.6, "erythema": 0.6, "pores": 0.36 }
  }
}

// 완료. /analyze/diary 의 result (0~10)
{
  "status": "done", "kind": "diary", /* ... */
  "result": {
    "skin_tone": 8.0,      // 피부 톤 밝기: 0=어두움, 10=매우 밝음 (ITA 선형 매핑)
    "pores": 4.7,          // 모공: 0=거의 안 보임, 10=매우 뚜렷. pores/10
    "redness": 4.2,        // 붉은기: 0=없음, 10=강함. erythema/10
    "confidence": { "skin_tone": 0.6, "pores": 0.36, "redness": 0.6 }
  }
}

// 실패 (다운로드 실패, 얼굴 미검출 등). 소비자는 이걸로 아직 처리 중인 상태와 구분
{
  "status": "failed", /* ... */
  "error": { "code": "analysis_failed", "message": "No face detected ..." }
}
```

- `result`는 점수와 confidence만 담는다. 요청 식별자와 시각은 바깥 envelope에 있다.
- `skin_tone`은 절대 색상 기반(ITA)이라 카메라와 조명에 민감하다.
  `reference_bbox`(그레이카드)를 주면 기기 독립적이 된다.
- `pores`는 볼에서만 측정한다. 모공이 가장 잘 보이는 곳은 코지만, 보정 코호트의 장비가
  볼에서만 셌기 때문에 눈금이 있는 곳이 볼뿐이다.

> **응답에서 빠진 것은 소비하는 쪽 책임이다.** `result`에는 파이프라인의 `warnings`(모공
> 예측 개수는 순위용이지 개수가 아님 / 그레이카드가 없어 절대 색상 신뢰도 낮음 / 얼굴이 작아
> 텍스처 신뢰도 낮음)와
> 의료기기 아님 고지가 들어가지 않는다. 점수의 신뢰도를 UI에서 표현하는 값은 `confidence`이고,
> 고지는 서비스에서 별도로 노출해야 한다. 두 정보가 모두 필요한 경우 CLI `analyze`가 전체
> `SkinReport`를 낸다.

### `GET /results/{key}`

Redis에 저장된 문서를 그대로 돌려주는 디버깅용 엔드포인트다
(`curl localhost:8000/results/470b634e...:analyze`). Spring Boot는 Redis를 직접 읽는 쪽이
빠르므로 이 엔드포인트에 의존하지 않는 게 맞다. 키가 없으면 404 `result_not_found`가
나온다(TTL 만료, 오타, 아직 시작 전).

### `GET /healthz`

`face_model_available`과 `detection_available`이 둘 다 `true`여야 분석이 가능하다.
`result_store`는 `"redis"`(정상) 또는 `"redis_unreachable"`(접속 불가. 제출이 503으로
거절됨)이다. 배포 후 가장 먼저 볼 값이다.

### 오류 응답

제출 시점에 판별 가능한 문제만 동기 4xx로 응답하고(아래 표), 다운로드와 분석 중의 실패는
Redis 문서의 `status: "failed"`로 전달된다(`error.code`는 같은 코드 체계).

| status | code | 상황 |
|---|---|---|
| 400 | `invalid_scheme`, `invalid_url`, `dns_error` | URL 자체가 잘못됨 |
| 403 | `blocked_host` | URL이 사설, 루프백, 링크로컬 주소로 해석됨 |
| 404 | `result_not_found` | `GET /results/{key}` 에서 키 없음이나 만료 |
| 422 | `invalid_request` | 요청 본문 검증 실패 |
| 503 | `result_store_unavailable` | Redis에 접수 기록조차 못 씀 |

백그라운드 실패로 문서에 기록되는 code: `decode_error`, `empty_body`, `image_too_large`,
`upstream_error`, `fetch_error`, `fetch_timeout`, `too_many_redirects`,
`analysis_failed`(얼굴 미검출, 전 ROI 탈락), `face_model_missing`, `detection_unavailable`,
`internal_error`.

### 보안 가드 (`api/fetch.py`)

서버가 사용자가 준 URL로 직접 요청하므로 SSRF 경계다. scheme allow-list(http, https)를
통과시킨 뒤, DNS 해석 결과가 사설이나 루프백, 링크로컬, 예약 대역이면 거부하고
(`169.254.169.254` 같은 클라우드 메타데이터 포함), 리다이렉트는 매 홉 재검증하며 횟수를
제한한다. 본문은 스트리밍하며 바이트 상한에서 중단하고, 디코딩 시 픽셀 수 상한으로 압축
폭탄을 막는다. 남는 위험은 DNS rebinding(검증과 실제 연결이 각각 해석)이며, 이를 위협모델에
포함한다면 egress 프록시에서 allow-list 하는 편이 낫다.

### 환경변수

| 변수 | 기본값 | 의미 |
|---|---|---|
| `SKIN_METRICS_API_CONFIG` | 패키지 기본 `config.yaml` | 설정 YAML 경로 |
| `SKIN_METRICS_FACE_MODEL` | 캐시 경로 | FaceLandmarker `.task` 경로 |
| `SKIN_METRICS_API_DOWNLOAD_MODEL` | `0` | 시작 시 모델 자동 다운로드 |
| `SKIN_METRICS_API_MAX_BYTES` | `20971520` (20MB) | 다운로드 바이트 상한 |
| `SKIN_METRICS_API_MAX_PIXELS` | `40000000` | 디코딩 픽셀 하드 상한 (압축폭탄 방어, 초과 시 413) |
| `SKIN_METRICS_API_ANALYSIS_MAX_PIXELS` | `16000000` | 분석 픽셀 예산. 초과 시 거절 대신 축소. 메모리가 메가픽셀당 약 63MB로 늘어나므로 이 값이 피크 메모리를 결정 |
| `SKIN_METRICS_API_FETCH_TIMEOUT` | `10.0` | 다운로드 타임아웃(초) |
| `SKIN_METRICS_API_MAX_REDIRECTS` | `3` | 리다이렉트 허용 횟수 |
| `SKIN_METRICS_API_ALLOW_PRIVATE_HOSTS` | `0` | 개발 전용. SSRF 가드 해제 |
| `SKIN_METRICS_API_MAX_CONCURRENCY` | `2` | 동시 분석 수 (파이프라인은 CPU 바운드) |
| `SKIN_METRICS_REDIS_HOST` | **필수** | Redis 호스트명. 비어 있으면 기동 실패 |
| `SKIN_METRICS_REDIS_PORT` | `6379` | Redis 포트 (Redis Cloud는 인스턴스별 포트) |
| `SKIN_METRICS_REDIS_USER` | `default` | Redis 사용자 |
| `SKIN_METRICS_REDIS_PASSWORD` | (없음) | Redis 비밀번호. `.env`에만 두며 커밋 대상이 아님 |
| `SKIN_METRICS_REDIS_DB` | `0` | DB 인덱스 |
| `SKIN_METRICS_REDIS_TLS` | `0` | TLS(`rediss`) 사용 여부 |
| `SKIN_METRICS_RESULT_TTL` | `3600` | 결과가 Redis에 남아 있는 시간(초) |

> `SKIN_METRICS_BIND`(기본 `127.0.0.1`)와 `SKIN_METRICS_PORT`(기본 `8000`)는 API가 아니라
> compose가 읽는 변수로, 호스트 쪽 바인딩 주소와 포트를 정한다.

> **응답 시간과 메모리**: 분석은 동기이고 CPU 바운드라 워커 스레드와 세마포어로 실행된다.
> 얼굴 크기 정규화가 들어간 뒤로는 입력 해상도가 시간에 거의 영향을 주지 않는다
> (6.5MP 6.4초, 40MP 8.9초). 반면 메모리는 메가픽셀당 약 63MB로 선형 증가하므로 실질적인
> 제약은 시간이 아니라 메모리다. 자세한 수치와 인스턴스 사이징은
> [AWS EC2 배포](#aws-ec2-배포)에 있다. 트래픽이 늘면 큐와 작업 ID 방식이 필요하다.

---
## Docker

```bash
docker build -t skin-metrics-api:0.1.0 .        # 기본 = api 타깃 (약 1.7GB)
docker run --rm -p 127.0.0.1:8000:8000 skin-metrics-api:0.1.0
curl localhost:8000/healthz
```

**재배포 스크립트**가 이전 스택 종료, 빌드, 재기동, 헬스체크 통과까지 한 번에 처리한다.

```bash
./redeploy.sh                          # 코드 수정 후 이것 하나면 끝
./redeploy.sh --no-cache               # 의존성까지 처음부터 다시 (약 5분)
./redeploy.sh --logs                   # 기동 후 로그 따라가기
SKIN_METRICS_PORT=8100 ./redeploy.sh   # 8000이 이미 쓰이는 경우
```

포트를 이미 쓰는 프로세스가 있으면 죽이지 않고 누가 쓰는지 알려주고 멈춘다. 이미지 정리도
`skin-metrics-api` 라벨이 붙은 것만 대상으로 해서, 다른 프로젝트의 컨테이너와 이미지와 빌드
캐시는 건드리지 않는다.

compose를 직접 쓰는 경우는 이렇다.

```bash
docker compose up -d --build      # 빌드 + 백그라운드 실행 (코드 수정 후엔 항상 --build)
docker compose up -d              # 이미지가 이미 있으면 빌드 없이 실행만
docker compose logs -f api
docker compose down

SKIN_METRICS_PORT=8100 docker compose up -d   # 8000 포트가 이미 쓰이는 경우
```

> **`up -d`는 이미지가 없을 때만 빌드한다.** 이미 `skin-metrics-api:0.1.0`이 있으면 소스를
> 고쳐도 예전 이미지를 그대로 띄운다. 코드 변경을 반영하는 플래그가 `--build`다.
> `trainer` 서비스는 `full` 프로파일이라 `up`으로는 뜨지 않는다.

FaceLandmarker 모델(약 3.8MB)이 빌드 시점에 이미지 안에 포함되므로 컨테이너는 시작할 때
네트워크가 필요 없다(빌드에는 필요하다). 밖으로 나가는 통신은 `/analyze`의 이미지 URL
다운로드뿐이다.

### 빌드 타깃 2종

| 타깃 | 내용 | 크기 | 용도 |
|---|---|---|---|
| `api` (기본) | Phase 1 + FastAPI. torch 없음 | 1.72GB | 배포용 |
| `full` | 위 구성에 `dl` extra(torch, torchvision, timm, albumentations, pandas) 추가 | 2.88GB | 컨테이너에서도 Phase 2 학습 |

```bash
docker build --target full -t skin-metrics-api:0.1.0-full .
docker run --rm skin-metrics-api:0.1.0-full skin-metrics train --dummy --mode ranking --epochs 1
# compose 로도 동일:
docker compose --profile full run --rm trainer train --dummy --mode ranking
```

두 타깃은 소스와 OS 레이어를 공유하고 가상환경만 다르다. 배포 이미지에 torch가 들어가지
않도록 기본 타깃을 `api`로 두었고, Phase 2가 필요하면 `full`이 로컬과 동일하게 동작한다.

linux에서는 torch와 torchvision을 CPU 전용 인덱스(`https://download.pytorch.org/whl/cpu`)에서
받도록 `pyproject.toml`에 설정돼 있다. PyPI 기본 휠은 nvidia CUDA 패키지를 끌고 오는데, GPU
없는 컨테이너에서 `import torch`가 SIGILL로 죽는다(`torch._preload_cuda_deps`). macOS 로컬은
영향 없이 기존 휠을 그대로 쓴다. 이 설정으로 이미지가 9.61GB에서 2.88GB가 됐다.

> `torchvision`이 `dl` extra에 명시돼 있는 이유는 `[tool.uv.sources]`가 직접 의존성에만
> 적용되기 때문이다. timm의 전이 의존성으로 두면 torch만 `+cpu`가 되어 ABI가 어긋나고
> `operator torchvision::nms does not exist` 로 학습이 실패한다.

> 빌드에는 디스크 여유가 넉넉해야 한다(15GB 이상 권장). 부족하면 빌드가 멈추면서 도커
> 데몬까지 응답하지 않을 수 있다(그 경우 Docker Desktop 재시작).

### 이미지 올리기 (레지스트리 push)

```bash
# 단일 아키텍처
docker build -t <registry>/<user>/skin-metrics-api:0.1.0 .
docker push <registry>/<user>/skin-metrics-api:0.1.0

# amd64 와 arm64 동시 (mediapipe 1.0.0 은 manylinux x86_64, aarch64 휠 모두 제공)
docker buildx build --platform linux/amd64,linux/arm64 \
  -t <registry>/<user>/skin-metrics-api:0.1.0 --push .
```

`ghcr.io/astral-sh/uv` 이미지를 쓰지 않고 PyPI의 uv를 설치하도록 되어 있어, 빌드는 Docker
Hub만 있으면 된다(일부 네트워크에서 ghcr 익명 pull이 막힌다).

### 이미지에 들어가는 것과 안 들어가는 것

`.dockerignore`가 전부 차단한 뒤 필요한 것만 허용하는 방식이라, 빌드 컨텍스트에는
`pyproject.toml`, `uv.lock`, `README.md`, `skin_metrics/` 만 들어간다. `data/`의 얼굴 사진,
`report*.json`, `tests/`, `.venv/`, `.git/`, `*.task`는 어떤 경로로도 이미지에 포함되지
않는다. 나중에 새 파일이 생겨도 기본이 차단이라 안전하다.

### AWS EC2 배포

**아키텍처 주의**: 맥에서 `docker build`로 만든 기본 이미지는 arm64라서 x86_64 EC2에 올리면
`exec format error`로 죽는다. GitHub Actions가 amd64로 빌드해 주므로 방법 A·B(ghcr pull)에는
이 문제가 없다.

#### 1. 인스턴스 선택

| 타입 | vCPU / RAM | 비고 |
|---|---|---|
| `t3.medium` | 2 / 4GB | 최소 사양. 버스터블이라 연속 요청 시 CPU 크레딧이 소진된다 |
| **`t3.large`** | 2 / 8GB | **권장 구성**. 메모리 여유가 있어 동시 요청에 안전 |
| `c6i.large` | 2 / 4GB | 비버스터블. CPU 성능이 일정해야 하는 경우 |
| `t4g.large` | 2 / 8GB | arm64(Graviton), 약 20% 저렴. 이 저장소는 arm64에서도 빌드와 검증 완료 |

- **EBS 루트 볼륨 30GB**가 필요하다. 기본 8GB로는 빌드 캐시가 들어가지 않는다.
- **보안 그룹**: 인바운드 TCP `8000`(또는 프록시를 쓸 경우 80과 443). 아웃바운드는 기본값이
  맞다. `/analyze`가 이미지 URL을 받아오려면 외부로 나갈 수 있어야 한다.

#### 2. 인스턴스 준비 (Amazon Linux 2023)

```bash
sudo dnf install -y docker git
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user      # 적용하려면 재로그인

# compose v2 플러그인 (AL2023 기본 포함 아님)
sudo mkdir -p /usr/libexec/docker/cli-plugins
sudo curl -sSL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-$(uname -m) \
  -o /usr/libexec/docker/cli-plugins/docker-compose
sudo chmod +x /usr/libexec/docker/cli-plugins/docker-compose
docker compose version
```

> Ubuntu에서는 `sudo apt install -y docker.io docker-compose-v2 git` 로 대체된다.

#### 3-A. 배포 방법 A: GitHub Actions 자동 배포 (현재 구성)

`.github/workflows/publish-image.yml`의 `deploy` 잡이 이미지를 ghcr에 올린 뒤 서버에 SSH로
붙어 컨테이너를 교체한다. Spring Boot(`notdesign-server`)와 **같은 서버·같은 시크릿**을 쓴다.

- **트리거**: `main` 푸시(경로 필터에 걸리는 변경) 또는 Actions 탭의 *Run workflow*.
- **저장소 시크릿**: `SERVER_HOST` / `SERVER_USER` / `SSH_PRIVATE_KEY`.
  `notdesign-server`와 같은 값이다.
- **서버 사전 준비는 두 가지뿐**:
  1. docker 설치 (`sudo apt install -y docker.io`)
  2. `/root/proof-face.env` — 이 저장소의 `.env`(Redis 접속 정보)를 그 이름으로 저장.
     `notdesign-server`의 `/root/proof.env`와 나란히 두면 된다.

  ghcr 로그인은 잡이 매번 자신의 `GITHUB_TOKEN`으로 하므로 **서버에 PAT을 둘 필요가 없다**
  (패키지가 private이어도 된다).
- **실행 형태**: `-p 127.0.0.1:8000:8000`(루프백 전용), `--restart unless-stopped`,
  `--memory 2g --cpus 2`, `--read-only`. Spring이 `--network host`로 도니
  `SERVICES_ANALYZE_URL=http://127.0.0.1:8000`으로 부르면 된다. 인증·레이트리밋이 없는
  상태이므로 보안 그룹에 8000을 여는 대신 루프백에 묶어 두는 쪽이 맞다.
- **배포 태그는 `:latest`가 아니라 커밋 SHA**다. 잡은 `/healthz`가 200을 낼 때까지 최대
  90초 기다리고, 그때까지 못 뜨면 컨테이너 로그 50줄을 남기고 실패한다.
- **롤백**: 직전 성공 실행을 *Re-run jobs* 하거나, 서버에서 원하는 SHA 태그로 직접
  `docker run`(방법 B).

> 메모리는 compose(4g / 동시 2건)보다 낮은 `--memory 2g` + `MAX_CONCURRENCY=1`로 잡아
> 뒀다. 같은 인스턴스에서 Spring이 함께 돌기 때문이다. 16MP 입력 1건의 피크 RSS가 약 1GB
> 이므로 여유는 있지만, 인스턴스를 키웠다면 워크플로의 `--memory`와
> `SKIN_METRICS_API_MAX_CONCURRENCY`를 같이 올린다.

#### 3-B. 배포 방법 B: GitHub Packages에서 수동 pull

이미지는 GitHub Actions가 빌드해 ghcr.io에 올린다(`.github/workflows/publish-image.yml`).
`main`에 푸시하면 자동으로 돌고, Actions 탭의 "Publish image" 에서 *Run workflow* 로 수동
실행도 된다. 러너가 amd64 네이티브라 맥에서 크로스 빌드하는 것보다 훨씬 빠르다.

```
ghcr.io/likelion14-hackathon/skin-metrics-api:latest
ghcr.io/likelion14-hackathon/skin-metrics-api:<commit-sha>   # 롤백용
```

저장소가 private이면 패키지도 private이므로 EC2에서 먼저 로그인이 필요하다. `read:packages`
스코프 PAT을 [github.com/settings/tokens](https://github.com/settings/tokens)에서
만든다(classic, `read:packages`만 체크).

```bash
echo '<PAT>' | docker login ghcr.io -u <github-사용자명> --password-stdin
```

> 패키지를 public으로 바꾸면(패키지 페이지에서 Package settings, Change visibility) EC2에서
> 로그인 없이 pull할 수 있다. 다만 이미지 안에 소스 코드가 들어 있으므로, 저장소를
> private으로 두는 이유가 있다면 위의 PAT 방식이 맞다.

EC2에서의 실행은 다음과 같다.

```bash
docker pull ghcr.io/likelion14-hackathon/skin-metrics-api:latest
docker run -d --name skin-metrics --restart unless-stopped \
  -p 0.0.0.0:8000:8000 \
  -e SKIN_METRICS_REDIS_HOST='<redis-host>' \
  -e SKIN_METRICS_REDIS_PORT='<port>' \
  -e SKIN_METRICS_REDIS_PASSWORD='<password>' \
  -e SKIN_METRICS_API_ANALYSIS_MAX_PIXELS=16000000 \
  -e MPLCONFIGDIR=/tmp/mpl \
  --read-only --tmpfs /tmp:rw,size=64m \
  --memory 4g --cpus 2 \
  ghcr.io/likelion14-hackathon/skin-metrics-api:latest
```

> Spring Boot가 같은 인스턴스에서 돌아 localhost로만 부르는 구성이면
> `-p 127.0.0.1:8000:8000`이 맞다. 외부에 전혀 노출되지 않아 인증이 없는 현재 상태에서 가장
> 안전하다.

업데이트는 `docker pull ... && docker rm -f skin-metrics && docker run ...` 이다.

> **Graviton(t4g, c7g) 배포**에서는 기본 워크플로가 amd64만 만든다는 점이 걸린다. Actions의
> *Run workflow* 에서 platforms 입력을 `linux/amd64,linux/arm64`로 주면 된다(arm64는 러너에서
> 에뮬레이션되어 빌드가 몇 배 느리다).

#### 3-C. 배포 방법 C: 저장소 클론 후 인스턴스에서 빌드

```bash
git clone https://github.com/likelion14-hackathon/proof-face.git
cd proof-face
cp .env.example .env
$EDITOR .env   # Redis host/port/password를 실제 값으로 (없으면 컨테이너가 안 뜬다)

# 0.0.0.0 바인딩이 있어야 인스턴스 밖에서 접근된다 (기본은 루프백)
SKIN_METRICS_BIND=0.0.0.0 ./redeploy.sh
```

빌드에 약 3분에서 5분 걸리고, 그 뒤 헬스체크 통과까지 자동으로 기다린다. 확인 방법은
다음과 같다.

```bash
curl http://<EC2-퍼블릭-IP>:8000/healthz
curl -X POST http://<EC2-퍼블릭-IP>:8000/analyze/diary \
  -H 'content-type: application/json' \
  -d '{"image_url":"https://example.com/face.jpg"}'
# 응답 예시: {"request_id": "...", "redis_key": "...:diary"}
curl http://<EC2-퍼블릭-IP>:8000/results/<request_id>:diary
```

`/healthz`의 `result_store`가 배포 성공 여부를 가른다. `"redis_unreachable"`이면 접속 정보는
전달됐지만 접속이 안 되는 것(방화벽, 자격증명, 인스턴스 다운)이고, 컨테이너가 아예 안 뜬다면
`docker logs`에 `SKIN_METRICS_REDIS_HOST is not set`이 찍혀 있다.

재배포는 방법 A라면 `main`에 푸시하는 것으로 끝이고, 방법 B는 새 이미지를 pull한 뒤
컨테이너를 재생성하고, 방법 C는
`git pull && SKIN_METRICS_BIND=0.0.0.0 ./redeploy.sh` 다. 두 방법 모두
`restart: unless-stopped`라 인스턴스를 재부팅해도 컨테이너가 살아난다.

#### 4. 성능과 메모리 (실측)

이 맥에서 컨테이너 CPU 2개 제한으로 잰 값이다. EC2 x86 vCPU는 더 느리므로 시간은 2배에서
3배로 잡는 게 맞다(메모리는 아키텍처와 무관하게 동일하다).

| 입력 | 소요 시간 | 피크 메모리 |
|---|---|---|
| 6.5MP | 6.4s | 647MB |
| 12MP (폰 기본) | 6.8s | 1.0GB |
| 24MP | 8.1s | 1.7GB |
| 40MP | 8.9s | 2.5GB |

메모리는 메가픽셀당 약 63MB로 선형 증가한다(내부 연산이 float64). 그래서
`SKIN_METRICS_API_ANALYSIS_MAX_PIXELS`(기본 16MP)를 넘는 이미지는 거절하지 않고 분석 직전에
축소한다. 요청 1건이 약 1.2GB, 동시 2건이 2.5GB로 묶여 compose의 `memory: 4g` 안에 들어온다.
이 예산이 없으면 40MP 사진 2장이 동시에 들어올 때 5GB를 써서 컨테이너가 OOM으로 죽는다.

축소해도 정확도 손실은 없다. 파이프라인이 어차피 모든 얼굴을
`normalization.target_eye_span_px`(512px)로 정규화하기 때문이다. 축소 후 얼굴이 너무 작아지면
기존 `under_resolved` 경고가 그대로 동작해 모공 신뢰도를 낮춘다.

#### 5. 현재 배포 구성의 전제

- **인증과 레이트리밋이 없다.** 퍼블릭 IP에 그대로 열면 누구나 호출할 수 있고, 요청 1건이
  CPU를 수 초씩 점유한다. 데모 기간에 가장 간단한 방어는 보안 그룹의 소스 IP를 팀이나 심사장
  대역으로 제한하는 것이다.
- **HTTPS가 아니다.** 프론트엔드가 HTTPS면 브라우저가 mixed content로 차단한다. 해법은 Caddy나
  nginx 리버스 프록시에 Let's Encrypt를 붙여 앞에 두거나, 프론트도 HTTP로 띄우는 것이다.
- `SKIN_METRICS_API_ALLOW_PRIVATE_HOSTS`를 EC2에서 켜면 `169.254.169.254`(인스턴스 메타데이터,
  곧 IAM 자격증명)로 요청을 보낼 수 있게 된다. 기본값 `0` 그대로면 막힌다.

### 운영 구성

- `docker-compose.yml`은 기본적으로 127.0.0.1 에만 바인딩한다(`SKIN_METRICS_BIND`로 변경).
  `/analyze` 앞에 인증과 레이트리밋이 없으므로, 외부 노출 구성에서는 리버스 프록시의 인증과
  요청 제한이 전제다.
- `SKIN_METRICS_API_ALLOW_PRIVATE_HOSTS=1`은 개발 전용이다. 켜면 컨테이너가 같은 네트워크의
  내부 서비스로 요청을 보낼 수 있게 된다(SSRF).
- 분석은 CPU 바운드다. `SKIN_METRICS_API_MAX_CONCURRENCY`와 컨테이너 CPU 한도는 같이 올라가야
  의미가 있다(compose 기본: 2 CPU, 동시 2건).
- 컨테이너는 비루트(uid 10001)로 실행되며 compose에서 `read_only: true` 로 뜬다.

---

## 테스트

```bash
uv run pytest -q          # 138 passed
```

- 합성 이미지와 합성 랜드마크 기반이라 Phase 1 테스트는 `detection`이나 `dl` extra 없이
  실행된다.
- `tests/test_models.py`는 torch 미설치 시, `tests/test_api.py`는 fastapi 미설치 시
  `importorskip`으로 자동 스킵된다.
- API 테스트는 루프백 HTTP 서버를 띄워 실제 다운로드 경로까지 태우며 외부 네트워크는 쓰지
  않는다.
- `tests/test_calibrate.py`는 합성 테이블과 합성 코퍼스로 돌아가므로 43GB 데이터셋 없이
  실행된다.
- 커버리지: 색보정 왕복과 CCM 복원과 D65 화이트와 마스크 기반 gray-world, ITA와 멜라닌과
  홍반 공식과 가드, 헤모글로빈 ICA(정상과 퇴화), 텍스처와 주름 추정, ROI 기하와 마스킹,
  정규화와 스키마, end-to-end 파이프라인, 보정 툴링(코퍼스 인덱싱, 릿지 왕복, 채택 게이트,
  분위수 매핑, 프로파일 병합), Phase 2 forward와 GRL과 학습 루프(regression, ranking).

---

## 정확도를 더 올리려면: 데이터 확보 가이드

현재 프로파일은 기존 코호트에서 짜낼 수 있는 것을 거의 다 짜냈다(전수 특징 부분집합 탐색까지
완료). 다음 단계는 전부 새 데이터가 필요하며, 지표별로 필요한 데이터와 구할 수 있는 곳이
다르다.

| 지표 | 병목 | 필요한 데이터 | 어디서 |
|---|---|---|---|
| **홍조** | 실측 라벨이 0장이라 순위 검증이 불가능 | 1) 전문의 CEA 등급(0~4) 사진 채점 2) Mexameter나 VISIA red 실측 | 1번이 최선이다. 장비가 필요 없고 기존 사진 200장에서 300장에 피부과 전문의 2명이면 된다. 2번은 VISIA 보유 피부과나 에스테틱 제휴 |
| **모공** | 현재 +0.58. 볼에만 눈금이 있어 T존을 못 쓴다 | 코와 이마에서 잰 모공 개수 실측 (Visiometer 등), 또는 타깃 폰으로 찍은 사진 + 장비 동시 측정 | 공개 데이터셋 없음. 028에는 볼 라벨뿐. 장비 대여나 에스테틱 제휴 |
| **색소** | 지도학습이 기기 종속으로 탈락 | 타깃 폰 단일 기종으로 찍은 사진과 전문가 등급 또는 장비 스팟 개수 | 공개 데이터셋 없음. 028 재활용 가능: `calibrate fit --device phone`(단, 타 기기 입력에 쓰면 안 됨) |
| **피부 톤** (`/analyze/diary`) | 절대 색상이라 카메라와 조명에 민감 | 데이터가 아니라 그레이카드. 요청에 `reference_bbox` 포함 | 촬영 UI에 그레이카드나 흰 종이 가이드 추가 |
| **전 지표 (Phase 2)** | 딥러닝 미학습 | 이미 보유. 028 라벨 38GB | `calibrate/aihub.py`의 `iter_roi_rows` 를 거쳐 `train --data labels.csv --mode regression` |

**공개 데이터셋에 기대기 어렵다.** 얼굴 정면 표준 촬영에 장비 실측이 붙은 공개 데이터는
사실상 없다. Fitzpatrick17k(1.6만 장)와 SCIN(1만 장 이상)은 병변 클로즈업이나 크라우드소싱에
진단명 라벨이라 이 파이프라인의 타깃(정면 얼굴, 중증도와 실측값)과 맞지 않고, VISIA와 CEA
임상 코호트(1,001명 규모)는 병원 보유라 비공개다. AI-Hub에서 얼굴과 피부 실측이 붙은
데이터셋은 028이 유일하다.

---

## 다음 단계

- **홍조 실측 라벨 확보**: 이 코호트에 홍반 실측값이 없어 홍조만 순위 검증이 남아 있다. 현재는
  코호트 백분위 기준의 상대 비교로 제공한다. 라벨이 생기면 `calibrate/fit.py`의 `specs`
  튜플에 한 줄 추가하는 것으로 붙는다.
- **레퍼런스 코호트 확장**: `calibration_profile.yaml`은 AI-Hub 028 코호트(대부분 타입 3~4)로
  피팅됐다. 타입 1~2와 5~6 버킷은 표본이 부족해 `default` 분포로 폴백한다. 다른 인구집단이 주
  대상이면 그 코호트로 재피팅이 필요하다.
- **모공 다음 레버**: 기존 추출 특징은 부분집합 전수 탐색으로 소진했다(4번째 특징이 +0.001).
  더 올리는 길은 모공 검출에 특화된 특징을 새로 만들어 재추출(약 27분)하거나, T존 실측
  라벨을 확보해 집계 부위를 넓히는 쪽이다. 후자가 사용자 체감에는 더 클 수 있다. 모공이 가장
  신경 쓰이는 부위는 코인데 지금은 점수에 반영되지 않기 때문이다.
- **탄력 지표는 보류**: 코퍼스에 Cutometer R2 실측이 있어 붙일 수는 있지만, 이미지 특징으로
  +0.329인 반면 나이 하나가 -0.434로 더 잘 맞히고 나이를 제거하면 +0.171로 무너진다. 사실상
  나이 추정기라 채택하지 않았다. 되살리려면 표면 광학이 아닌 다른 신호가 필요하다.
- **Phase 2 실학습**: 라벨 CSV는 이제 존재한다(`calibrate/aihub.py`). 38GB 이미지 학습이라 이
  맥에서는 수 시간에서 하루 단위다.
- **기기별 프로파일**: 서비스가 폰 전용이라면 `calibrate fit --device phone`으로 만든 폰 전용
  프로파일이 pooled보다 나을 수 있다. 비교해 볼 가치가 있다.
- **Tsumura 기준 벡터**(`erythema._HEMOGLOBIN_DIR`, `_MELANIN_DIR`)는 근사값이고 FastICA가
  수렴 실패하는 이미지가 꽤 된다. 카메라별 측정 흡광 스펙트럼으로 교체하는 것이 정공법이다.
- **리포트 시각화**: `SkinReport`를 HTML이나 차트(얼굴 오버레이, ROI별 점수, 시계열)로.
- 동일 조건에서 두 시점을 비교하는 `compare` 가 여전히 단일 절대 점수보다 신뢰도가 높다.
