# skin-metrics

카메라 이미지에서 **색소침착(pigmentation) / 홍조(erythema) / 수분력(hydration proxy)** 3개
지표를 0~100 점수로 산출하는 하이브리드 시스템입니다. 물리 기반 색상분석(Phase 1)과
딥러닝 회귀(Phase 2)를 결합한 구조입니다.

> ⚠️ **의료기기가 아닙니다.** 출력은 **미용 참고 정보**일 뿐이며 **진단 목적으로 사용할 수
> 없습니다.** / **This system is not a medical device.** Outputs are cosmetic reference
> information only and must not be used for diagnostic purposes.

---

## 목차
- [빠른 시작](#빠른-시작)
- [설계 개요](#설계-개요)
- [Phase 1: 물리 기반 파이프라인 (기술 상세)](#phase-1-물리-기반-파이프라인-기술-상세)
- [Phase 2: 딥러닝 (기술 상세)](#phase-2-딥러닝-기술-상세)
- [레퍼런스 보정](#레퍼런스-보정-skin_metricscalibrate)
- [점수 해석 · 신뢰도 · 보정](#점수-해석--신뢰도--보정)
- [CLI 레퍼런스](#cli-레퍼런스)
- [HTTP API](#http-api)
- [Docker](#docker)
- [AWS EC2 배포](#aws-ec2-배포)
- [테스트](#테스트)
- [정확도를 더 올리려면 — 데이터 확보 가이드](#정확도를-더-올리려면--데이터-확보-가이드)
- [알려진 한계 · TODO](#알려진-한계--todo)

---

## 빠른 시작

```bash
# 코어 (Phase 1 물리 파이프라인 + 테스트)
uv sync

# 실이미지 얼굴 검출 (MediaPipe)
uv sync --extra detection

# Phase 2 딥러닝
uv sync --extra dl

# HTTP API (이미지 URL → 분석)
uv sync --extra api --extra detection

# 분석 (venv 활성화 시 uv 없이 skin-metrics 직접 실행 가능)
# 첫 실행: --download-model 로 FaceLandmarker 모델(~3.8MB) 자동 다운로드
skin-metrics analyze data/face.jpg --download-model --output report.json

# 이후 실행: 모델 캐시 재사용 (플래그 불필요)
skin-metrics analyze data/face.jpg --output report.json

# 그레이카드/흰 종이가 프레임에 있으면 그 영역 지정 → 보정 신뢰도 상승
skin-metrics analyze data/face.jpg --reference-bbox 10,10,40,40 --output report.json

# 두 시점 비교 (같은 사람·같은 조건 촬영 권장)
skin-metrics compare data/before.jpg data/after.jpg

# Phase 2 딥러닝 스캐폴드 (라벨 없이 더미로 학습 루프 검증)
skin-metrics train --dummy --mode ranking --epochs 3

# HTTP API 서버 (docs: http://127.0.0.1:8000/docs)
skin-metrics serve --download-model
curl -X POST http://127.0.0.1:8000/analyze \
  -H 'content-type: application/json' \
  -d '{"image_url": "https://example.com/face.jpg"}'

# 다이어리용 0~10 점수 (피부 톤 / 당김·건조함 / 붉은기) — 202 + request_id 반환
curl -X POST http://127.0.0.1:8000/analyze/diary \
  -H 'content-type: application/json' \
  -d '{"image_url": "https://example.com/face.jpg"}'

# 결과는 Redis {request_id}:diary 에 저장됨. 디버깅은:
curl http://127.0.0.1:8000/results/<request_id>:diary
```

> `uv`는 `~/Library/Python/3.9/bin`에 설치됩니다. PATH에 없으면 전체 경로로 부르거나,
> venv를 `source .venv/bin/activate` 하면 `skin-metrics` 콘솔 스크립트를 uv 없이 쓸 수 있습니다.

---

## 설계 개요

```
image ─▶ [calibration] ─▶ [detection/ROI] ─▶ [features] ─▶ [scoring] ─▶ SkinReport
          색 보정            얼굴·ROI·마스킹      물리 지표         정규화·점수
```

- **의존성 분리**: 무거운 패키지(`mediapipe`, `torch`, `timm`, `albumentations`)는
  `pyproject.toml`의 optional extra(`detection`, `dl`)로 분리하고 **코드에서 지연 import**
  합니다. 덕분에 Phase 1 로직과 단위 테스트는 코어 의존성만으로 실행됩니다.
- **색공간 규약**: 모든 이미지는 `(H, W, 3)` float RGB. `linear`은 scene-linear `[0,1]`,
  `sRGB`는 디스플레이 인코딩 `[0,1]`. 파이프라인 내부는 **선형 RGB**에서 연산합니다.
- **방어적 수치 계산**: 0 division, 음수/0 log 입력은 전 함수에서 epsilon clip으로 방어.
- **신뢰도 전파**: 색 보정 방식·유효 ROI 비율·헤모글로빈 분리 성공 여부가 `confidence`로
  전파됩니다.

### 디렉토리

```
skin_metrics/
├── calibration/color.py     # sRGB 선형화, 화이트밸런스, CCM, D65 CIELab
├── detection/face.py        # MediaPipe FaceMesh, 5 ROI, 아티팩트 마스킹
├── features/
│   ├── pigmentation.py      # ITA, 멜라닌 지수, 반점 검출, 톤 균일도
│   ├── erythema.py          # 홍반 지수, a*, Tsumura 헤모글로빈 ICA
│   └── hydration_proxy.py   # 광택/GLCM·LBP/스케일링/미세주름 (proxy)
├── scoring/
│   ├── normalize.py         # 경험적 분위수 백분위, 지도학습 예측 적용, compare()
│   └── schema.py            # pydantic SkinReport / MetricScore / FaceScale
├── calibrate/               # 오프라인 보정 (런타임에서 import 안 됨)
│   ├── aihub.py             # AI-Hub 028 코퍼스 인덱싱 (이미지 ↔ 실측 라벨 조인)
│   ├── extract.py           # 멀티프로세스·재개가능 특징 추출 → CSV
│   └── fit.py               # anchor/릿지/분위수 격자 피팅 + 채택 게이트
├── models/                  # Phase 2: dataset(+dummy) / network / train
├── api/                     # HTTP API (FastAPI)
│   ├── app.py               # /healthz, /analyze 엔드포인트 + lifespan
│   ├── fetch.py             # 이미지 URL 다운로드 (SSRF·크기 가드)
│   ├── schemas.py           # 요청/응답 pydantic 모델
│   └── settings.py          # SKIN_METRICS_API_* 환경변수 설정
├── pipeline.py              # image -> SkinReport 오케스트레이션
├── config.py / config.yaml  # 설정 로더(2파일 병합) / 사람이 관리하는 임계값·정책
├── calibration_profile.yaml # 생성 파일: 피팅된 anchor·레퍼런스·지도학습 계수
└── cli.py                   # typer: analyze / compare / train / serve / calibrate
tests/                       # 합성 이미지·랜드마크·테이블 기반 단위 테스트
Dockerfile                   # 멀티스테이지: api(기본) / full(Phase 2 포함)
docker-compose.yml           # 로컬 실행 + trainer 프로파일
.dockerignore                # deny-all + allow-list (로컬 사진·리포트 유출 차단)
```

---

## Phase 1: 물리 기반 파이프라인 (기술 상세)

### 1. 색 보정 — `calibration/color.py`

| 함수 | 내용 |
|---|---|
| `linearize_srgb(img)` | sRGB EOTF 역변환(감마 제거). `s≤0.04045 ? s/12.92 : ((s+0.055)/1.055)^2.4` |
| `encode_srgb(lin)` | 역변환(선형→sRGB). 왕복 오차 < 1e-4 |
| `white_balance_grayworld(img, mask)` | 채널 평균을 회색으로 정규화. `mask`로 게인 추정에 쓸 픽셀 제한. **약한 fallback → `success=False`** |
| `white_balance_from_reference(img, bbox)` | 프레임 내 중립 패치(그레이카드/흰 종이) 평균으로 채널 게인 산출. 너무 어둡/클리핑 시 거부 |
| `estimate_ccm(detected, reference)` | 24패치 컬러체커 → **최소제곱 3×3 색보정 행렬**. `M = lstsq(detected, reference)`, RMS 잔차 반환 |
| `apply_ccm(img, M)` | `rgb @ M` 적용 |
| `rgb_to_lab(img)` | **colour-science**로 sRGB primaries + **D65** whitepoint 기준 CIELab. `L*∈[0,100]` |
| `calibrate_image(...)` | 오케스트레이션: 선형화 → (CCM) → WB. `status ∈ {reference, grayworld, none}` 와 `success` 반환 |

- **보정 신뢰도 규약**: `reference`(중립 패치 성공) 또는 CCM 적용 시 `success=True`,
  그 외에는 `False`. 이 값이 downstream `confidence`를 낮춥니다.

#### ⚠️ 인물 사진에 gray-world를 쓰면 안 되는 이유

gray-world는 "장면 전체 평균이 무채색"이라는 가정입니다. **얼굴이 프레임을 채우면 장면
평균이 곧 피부색**이므로, 게인을 적용하는 순간 피부의 색도가 나눠져 사라집니다.
실측 코호트에서 확인된 결과:

| WB 모드 | ROI 평균 a* | ROI 중앙값 b* | ITA |
|---|---|---|---|
| `grayworld` (구 기본값) | **0.5** | **-1.2** | **93°** (±90에서 포화) |
| `none` (카메라 AWB) | 15.9 | 26.6 | 39.8° |

a*(홍조)와 b*(황색)가 0으로 붕괴하면 ITA = `atan2(L*-50, b*)`가 ±90°로 포화하고
**부호가 무작위로 뒤집힙니다**. 코호트 90명 기준, 전문가 색소 등급과의 상관은:

| `calibration.fallback` | spearman(-ITA, 등급) | ITA 평균±sd | Fitzpatrick 분포 (타입 1~6) |
|---|---|---|---|
| `none` (**현재 기본값**) | **+0.437** | 31.8 ± 6.4 | [0, 0, 55, 35, 0, 0] |
| `background` | +0.367 | 38.6 ± 10.4 | [4, 13, 54, 19, 0, 0] |
| `grayworld` | **-0.307** | 6.5 ± **83.1** | [16, 3, 32, 10, 28, 1] |

gray-world는 노이즈를 더하는 정도가 아니라 **색소 신호를 뒤집습니다**.
기본 fallback은 `none`(카메라 AWB 신뢰)이며, 한국인 코호트에서 유일하게 타당한
Fitzpatrick 분포(타입 3~4)를 냅니다. 카메라 AWB가 감당 못 하는 강한 색 조명 환경이면
`calibration.fallback: background`(비피부 픽셀만으로 gray-world 추정)를 쓰세요.

- **정확도의 핵심 레버**는 여전히 **그레이카드 `--reference-bbox`** 또는 컬러체커 CCM입니다.
- **배경 영향**: 지표 계산은 ROI 내부만 쓰므로 배경 누끼는 불필요합니다.

### 1-b. 얼굴 크기 정규화 — `pipeline._normalize_face_scale`

GLCM·LBP·미세주름 밀도는 **고정 픽셀 오프셋**에서 계산되므로, 얼굴이 차지하는 픽셀 수가
다르면 값을 비교할 수 없습니다. 코호트 기기별 눈 사이 거리(외안각):

| 기기 | 평균 eye-span | 해상도 |
|---|---|---|
| 디지털카메라 | 1140 px | 2136×3216 |
| 스마트패드 | 972 px | 2448×3264 |
| 스마트폰 | 889 px (중앙값) | 1920×2560 |

파이프라인은 특징 추출 전에 얼굴을 **eye-span 512px**로 맞춥니다
(`normalization.target_eye_span_px`).

- **다운스케일 전용**: 더 작은 얼굴은 그대로 둡니다. 업샘플링은 없는 디테일을 만들어내지
  않으면서 텍스처만 매끄럽게 만들어 **거짓으로 촉촉하게** 보이게 하기 때문입니다.
  대신 `under_resolved` 플래그가 서고, 경고가 붙고, 수분 confidence가 ×0.6 됩니다.
- **선형 광에서 리샘플**: sRGB 인코딩 상태로 축소하면 감마 곡선을 타고 ROI 평균 색이
  편향됩니다.
- 부수 효과로 36MP 이미지 분석이 **약 11초 → 3.6초**로 빨라집니다.

### 2. 얼굴·ROI 검출 — `detection/face.py`

- **랜드마크**: MediaPipe FaceMesh 468점. **두 API 모두 지원**
  - 레거시 `mp.solutions.face_mesh` (MediaPipe 0.10.x)
  - **Tasks API `FaceLandmarker`** (MediaPipe ≥ 1.0) — `face_landmarker.task` 모델 필요
- **모델 파일 처리**:
  - `resolve_face_model()`: `model_path` 인자 → `SKIN_METRICS_FACE_MODEL` env →
    `~/.cache/skin_metrics/face_landmarker.task` 순으로 탐색
  - `ensure_face_model()`: 없으면 Google 저장소에서 **~3.8MB** 다운로드
  - CLI `--download-model` 플래그로 첫 실행 시 자동 다운로드
- **5 ROI**: 이마/좌볼/우볼/코/턱. 각 ROI는 선별된 내부 랜드마크의 **convex hull**을
  `cv2.fillConvexPoly`로 래스터화(정확한 경계 loop 불필요, 합성 데이터로도 테스트 가능).
- **아티팩트 마스킹 `mask_artifacts(...)`**:
  - **정반사/글레어**: sRGB→HSV, `V > glare_v_min(0.92)` **AND** `S < glare_s_max(0.15)`
  - **그림자**: ROI 내 `L*`의 하위 `shadow_percentile(5%)` 컷
  - **모발/눈썹/입술**: 랜드마크 기반 `exclusion_mask()`(눈·눈썹·입술 폴리곤을 dilate하여 제외)
- **유효 픽셀 게이트**: `valid/region < min_valid_ratio(0.60)`이면 해당 ROI를 `None` 처리.

### 3. 물리 지표 — `features/`

**색소침착 `pigmentation.py`**
- `ita(L, b) = arctan((L-50)/b)·180/π` — `b≈0`은 부호 보존 epsilon으로 방어
- `melanin_index(R_red) = 100·log10(1/R_red)` — `R`을 `[eps,1]`로 clip
- `spot_detection(L, mask, σ=15)` — 마스크-정규화 가우시안 국소 평균 대비 **음의 편차**를
  임계(`contrast_thresh`) 이상이면 반점. 연결성분(skimage `label`)으로 **면적률·개수·평균 대비도**
- `evenness(L, mask)` — ROI 내 `L*` 표준편차(톤 균일도)
- `estimate_fitzpatrick(ita, boundaries)` — ITA 컷오프(Del Bino)로 타입 1~6 추정

**홍조 `erythema.py`**
- `erythema_index(R_r, R_g) = 100·(log10(1/R_g) − log10(1/R_r))`
- `mean_a_star(lab, mask)` — CIE `a*` 평균 + 90퍼센타일
- `hemoglobin_map(rgb, mask)` — **Tsumura 색소분리**:
  1. 광학 밀도 `-log(reflectance)`로 변환
  2. **FastICA(n=2)** 로 멜라닌·헤모글로빈 2성분 분리
  3. ICA mixing 열벡터와 **기준 흡광 방향의 코사인 유사도**로 헤모글로빈 성분 식별,
     부호는 "헤모글로빈↑ → 값↑"이 되도록 결정 (순서·부호 모호성 해소)
  4. 퇴화 입력(픽셀 부족·상수)은 `separation_ok=False`로 안전 반환

**수분력 프록시 `hydration_proxy.py`** — 함수명·docstring·스키마 모두 **proxy/estimate 명시**
> RGB로는 수분을 직접 측정할 수 없음. 아래는 전부 표면 광학/텍스처 기반 **간접 추정치**.
- `specular_ratio_proxy` — 정반사 픽셀 비율(광택)
- `texture_features_proxy` — **GLCM**(contrast/correlation/energy) + **LBP** 균일도(정수 양자화)
- `scaling_index_proxy` — Laplacian 고주파 에너지(각질/거칠기)
- `micro_wrinkle_density_proxy` — **Frangi/Hessian** ridge 필터 → 선형 구조 밀도

### 4. 정규화·점수화 — `scoring/`

- **집계**: 파이프라인이 유효 ROI별 지표를 **유효 픽셀 수 가중 평균**으로 얼굴 단위 집계.

점수는 항상 **score driver를 레퍼런스 분포의 백분위로 변환**해 나옵니다. driver를 얻는
경로가 두 가지입니다:

| 경로 | 조건 | driver |
|---|---|---|
| **보정됨** (calibrated) | `supervised.<metric>` 모델 존재 | 릿지 회귀가 예측한 **실측 장비값** (`score_sign`으로 방향 정렬) |
| 미보정 (fallback) | 모델 없음 | anchor로 z-score한 서브피처의 **가중합** |

- **`predict_instrument(model, roi_features)`**: 모델이 **학습된 ROI에만**
  적용됩니다(`applies_to_rois`). 코 부위 특징으로 볼 Corneometer 값을 예측하는 건
  외삽이므로 제외합니다. 얼굴 단위 값은 부위별 예측의 **단순 평균**입니다 —
  장비가 부위 면적과 무관하게 부위당 1회씩 측정했으므로, 점수를 조회할 레퍼런스 분포도
  같은 방식으로 만들어야 합니다(면적 가중을 하면 다른 분포에 대고 조회하게 됨).
- **`score_from_raw(raw, metric, fitz, config)`**: 해당 **Fitzpatrick 타입 레퍼런스**의
  **경험적 분위수 격자**(0~100, 101점)에서 선형 보간해 백분위 산출. 격자가 없으면
  기존 정규 CDF로 폴백. 스팟 개수·예측 등급 분포는 눈에 띄게 치우쳐 있어서 가우시안
  가정은 꼬리를 잘못 배치합니다.
- **`compare(current, baseline, min_delta)`**: 지표별 변화량·방향·유의 여부(시계열).
- **출력 스키마 `schema.py`** (pydantic):

```python
class MetricScore(BaseModel):
    score: float                  # 0-100 condition index (높을수록 뚜렷)
    confidence: float             # 0-1
    raw_features: dict
    is_estimate: bool             # hydration은 항상 True
    calibrated: bool              # 실측 라벨로 학습된 모델이 낸 점수인가
    predicted_value: float | None # 예측된 실측 장비값 (보정된 경우)
    predicted_units: str | None   # 예: "grade 0-5"

class SkinReport(BaseModel):
    pigmentation / erythema / hydration: MetricScore
    roi_breakdown: dict            # ROI별 valid_ratio + 지표
    calibration_status: Literal["reference","grayworld","none"]
    fitzpatrick_estimate: int      # 1-6
    face_scale: FaceScale          # eye_span_px / scale_factor / under_resolved
    calibration_profile: str | None  # 어떤 코호트로 보정됐는지
    warnings: list[str]
    disclaimer: str                # 의료기기 아님 고지 (자동 포함)
```

- **confidence 계산**(`pipeline.py`): `calibration(reference 1.0 / grayworld 0.6 / none 0.6)
  × (유효 ROI 수 / 5)`. 홍조는 헤모글로빈 분리 실패 시 ×0.7, 수분은 `under_resolved`면 ×0.6.
  `none`은 더 이상 실패 경로가 아니라 **기본 경로**이고 레퍼런스 코호트도 이 조건에서
  촬영·피팅됐기 때문에 grayworld보다 불리하게 두지 않습니다.

---

## Phase 2: 딥러닝 (기술 상세)

> 상태: **완전히 도는 스캐폴드**. 실측 라벨(Corneometer/Mexameter)이 있어야 의미 있는 절대값을
> 냅니다. 라벨이 없으면 더미 데이터로 학습 루프 전체를 즉시 검증 가능. `dl` extra 필요.

### `models/dataset.py`
- **`Sample`**: ROI 크롭 이미지 + **Phase 1 물리 피처 벡터**(12차, `PHYSICS_FEATURE_NAMES`) +
  멀티태스크 라벨(3) + Fitzpatrick + 조명 버킷.
- **`DummyLabelGenerator`**: 라벨을 물리 벡터의 (노이즈 섞인) 결정적 함수로 생성 →
  **모델이 실제로 학습 가능**, 라벨·이미지 파일 없이 루프 검증. 조명 버킷이 이미지 밝기에
  반영되어 도메인 적대 헤드에 신호 제공.
- **`SkinDataset.from_csv`**: 실측 라벨 CSV 로더(누락 타깃은 NaN → ranking 모드 호환).
- **Augmentation**(albumentations 2.x): **색상 변환을 공격적으로**(ColorJitter, 밝기/대비,
  RGBShift, GaussNoise, JPEG 압축), **기하 변환은 약하게**(flip, mild Affine).

### `models/network.py`
- **백본**: timm `efficientnet_b0` (`pretrained` 옵션; 오프라인/테스트는 `False`)
- **물리 브랜치**: `PhysicsMLP`(Linear→ReLU→LayerNorm→Linear)로 임베딩 후 백본 특징과 **concat**
- **헤드 3개**: 색소/홍조/수분 회귀. **homoscedastic uncertainty weighting**
  (`L = Σ exp(−sᵢ)·Lᵢ + sᵢ`, Kendall 2018)으로 멀티태스크 loss 균형
- **조명 불변성**: **Gradient Reversal Layer** + 조명 버킷 분류기(domain-adversarial)로
  공유 특징을 조명에 불변하도록 학습

### `models/train.py`
- **손실**:
  - `mode="regression"`: **Huber(Smooth-L1)** + uncertainty weighting (이상치 강건)
  - `mode="ranking"`: **pairwise margin ranking loss** (절대값 대신 "A가 B보다 건조" 쌍 비교)
  - + 도메인 적대 cross-entropy
- **검증 지표**: **MAE, Pearson r, Spearman ρ**, 그리고 **반드시 Fitzpatrick 타입별 분리 리포트**
- **엔트리포인트**: `run_training(data_csv, config, mode, epochs, use_dummy, pretrained, ...)`

---

## 레퍼런스 보정 (`skin_metrics/calibrate/`)

0~100 점수가 의미를 가지려면 **실제 사람들의 분포**가 필요합니다. 그 분포와 지도학습
모델은 AI-Hub 공개 데이터셋 **《028. 한국인 피부상태 측정 데이터》**로 피팅했습니다.

| 항목 | 규모 |
|---|---|
| 피험자 | 965명 (train 858 / val 107, **피험자 단위로 분리**) |
| 이미지 | 정면 2,895장 (3기기 × 965명) |
| ROI 행 | 11,580 (이마·좌우볼·턱) |
| 실측 라벨 | Corneometer 수분, 전문가 색소/모공/주름 등급, 장비 스팟·모공 개수 |

```bash
# 1) 코호트 전체에서 물리 특징 추출 (재개 가능, 약 27분 / 6워커)
skin-metrics calibrate extract --data-root "028. 한국인 피부상태 측정 데이터" --workers 6

# 2) anchor·레퍼런스 분포·지도학습 모델 피팅 → calibration_profile.yaml
skin-metrics calibrate fit --dry-run     # 검증 수치만 출력
skin-metrics calibrate fit               # 프로파일 기록
```

### 설정 파일이 둘인 이유

| 파일 | 성격 |
|---|---|
| `config.yaml` | 사람이 관리하는 임계값·정책·**특징 선택** + 그 근거(주석). 기계가 덮어쓰지 않음. 여기 적힌 composite 가중치는 피팅이 채택되지 않았을 때의 fallback |
| `calibration_profile.yaml` | **생성 파일**. anchor / composite 가중치 / 레퍼런스 분위수 격자 / 지도학습 계수 / 검증 수치 |

`load_config()`가 둘을 병합합니다. 프로파일이 없어도 파이프라인은 그대로 동작합니다
(미보정 composite 경로).

### 채택 게이트 — 두 종류

**composite 가중치**: 피팅한 가중치가 held-out에서 `config.yaml`의 선언된 가중치보다
Spearman ≥ +0.05 더 좋을 때만 채택합니다. 현재 색소는 채택(+0.540 → +0.625),
수분은 기각(+0.320 → +0.315). 수분은 `config.yaml`에 선언된 가중치가 이미 피팅 결과와
동률이라(부분집합 탐색으로 고른 값이라 당연합니다) 게이트가 선언값을 유지합니다.

> ⚠️ 이 게이트는 한동안 **수분을 잘못 기각**하고 있었습니다. `fit_composite_weights`가
> `COMPOSITE_TARGET_SIGN`을 적용하지 않고 원본 타깃에 회귀해서, sign=-1인 수분에서
> 항상 부호가 뒤집힌 가중치를 내놨기 때문입니다(pooled -0.208 / 폰 -0.128 /
> 태블릿 -0.338 / 디지털카메라 -0.365 — 전부 음수였던 게 단서였습니다). 색소는
> sign=+1이라 우연히 맞아서 드러나지 않았습니다.

**지도학습 모델 (실측 장비값 예측)**: held-out ≥30행, |pearson| ≥ 0.25,
MAE가 "평균 예측" 대비 ≥5% 개선. 평균값만 예측하는 것과 별 차이 없는 모델이 리포트에
`수분 62 a.u.` 같은 구체적 숫자를 찍으면, 개인에 대한 정보가 거의 없는데도 있는 것처럼
보이기 때문입니다.

어느 쪽이든 탈락은 오류가 아닙니다. 해당 항목은 선언된 값 / 코호트 백분위 경로로
돌아가고, 사유가 프로파일의 `validation` · `validation_weights`에 기록됩니다.

### 이게 실제로 무엇을 고쳤나 (held-out 321명)

보정 전후로 **동일한 held-out 이미지**를 점수화한 분포입니다. 백분위 매핑이 올바르면
십분위마다 약 32명씩 균등해야 합니다.

| 지표 | 보정 후 | 보정 전 (placeholder anchor/reference) |
|---|---|---|
| pigmentation | 평균 48.6 · 십분위 23~41 | 평균 41.3 · 십분위 **0~79** · 10.7~84.4 |
| erythema | 평균 49.8 · 십분위 21~43 | 평균 **90.3** · **321명 중 215명이 최상위 십분위** |
| hydration | 평균 48.6 · 십분위 27~37 | 평균 41.6 · **321명 중 288명이 두 십분위에 몰림** · 25.4~53.5 |

보정 전 점수는 **변별력이 거의 없었습니다**. 특히 홍조는 사실상 모두가 "매우 심함"으로
나왔고, 수분은 25~53의 좁은 구간에 뭉쳐 있었습니다.

### 실측 일치도 — 배포중인 점수가 실제로 얼마나 맞나

**앞의 십분위 표는 분포가 맞다는 뜻이지 순위가 맞다는 뜻이 아닙니다.** 실측과의 순위
일치도(Spearman)는 별도로 측정해야 합니다. held-out 321명:

| 지표 | 대상 | 최초(placeholder) | 1차 보정 | 수분 라운드 | **현재** |
|---|---|---|---|---|---|
| pigmentation | 전문가 색소 등급 | +0.096 | +0.139 | +0.422 | **+0.422** |
| pigmentation | 장비 스팟 개수 | +0.168 | +0.208 | +0.578 | **+0.578** |
| hydration | Corneometer 수분 | -0.028 | +0.048 | +0.241 | **+0.320** |
| erythema | — | 측정 불가 | 측정 불가 | 측정 불가 | 측정 불가 |

수분은 held-out **볼 642행** 기준입니다(집계 부위가 볼이므로 거기서 재는 게 맞습니다).
기기별 +0.292(폰) / +0.389(태블릿) / +0.339(디지털카메라).

기기별로도 일관됩니다 — 등급 기준 디지털카메라 +0.425 / 태블릿 +0.554 / 폰 +0.329
(재가중 전에는 **-0.011** / +0.313 / +0.481로 기기마다 제각각이었습니다).

#### 색소 composite에서 절대 색상 특징을 뺀 이유

가장 큰 개선은 anchor 재추정이 아니라 **특징 선택**에서 나왔습니다. 장비 스팟 개수와의
순위 일치도를 기기별로 보면:

| 특징 | 전체 | 디지털카메라 | 태블릿 | 폰 | 기존 가중치 |
|---|---|---|---|---|---|
| `spot_count` | **+0.578** | +0.595 | +0.726 | +0.627 | **없었음** |
| `spot_area_ratio` | +0.434 | +0.329 | +0.495 | +0.592 | 0.30 |
| `spot_mean_contrast` | +0.362 | +0.574 | +0.380 | +0.244 | 없었음 |
| `evenness` | +0.247 | +0.356 | +0.358 | +0.486 | 0.20 |
| `ita` | -0.258 | -0.676 | -0.236 | -0.210 | 0.10 |
| `melanin_index` | +0.050 | +0.519 | **-0.085** | +0.031 | **0.40** |

**절대 색상(멜라닌·ITA)은 카메라를 넘으면 부호까지 뒤집힙니다** — 카메라별 색 재현과
센서 노이즈가 절대값을 옮기기 때문입니다. 형태학 특징(반점 개수·면적·대비·균일도)은
세 기기 모두에서 유지됩니다. 그런데 기존 composite은 **가장 불안정한 `melanin_index`에
가중치 40%**를 주고 **가장 강한 `spot_count`는 아예 빼놓고** 있었습니다.

형태학 특징만 남기고 가중치를 코호트에서 피팅한 결과가 위 표의 "현재"입니다.
그레이카드/컬러체커 보정(`--reference-bbox`)을 쓰면 절대 색상이 기기 독립적이 되므로
그때는 다시 넣을 가치가 있습니다.

> **주의**: 이 때문에 색소 점수의 의미가 "전반적 톤 어두움 포함"에서 **"반점 부담"**으로
> 좁아졌습니다. 피부가 전반적으로 어두운 것 자체는 더 이상 점수를 올리지 않습니다.

### 지도학습 모델 검증 결과 (held-out, 피험자 분리)

| 지표 | 타깃 | pearson | MAE (vs 평균 예측) | 채택 |
|---|---|---|---|---|
| pigmentation | 전문가 색소 등급 0–5 | +0.302 | 0.958 vs 0.981 (+2.4%) | ✗ (5% 미달) |
| hydration | Corneometer 수분 | +0.173 | 8.783 vs 8.867 (+0.9%) | ✗ (r 0.25 미달) |
| erythema | — (이 코호트에 실측 장비값 없음) | — | — | 코호트 백분위만 |

둘 다 게이트에서 탈락했고, 세 지표 모두 **코호트 백분위 경로**로 점수가 나옵니다.
탈락 사유는 `calibration_profile.yaml`의 `validation` 블록에 기록됩니다.

#### 왜 탈락했나 — 기기 종속성

기기별로 따로 피팅하면 색소 모델은 잘 작동합니다. 그런데 **기기를 넘으면 무너집니다**:

| 학습 ↓ / 평가 → | 디지털카메라 | 스마트패드 | 스마트폰 |
|---|---|---|---|
| 디지털카메라 | r=+0.600 **+17.4%** | r=+0.317 **-114.8%** | r=+0.125 **-217.4%** |
| 스마트패드 | r=+0.148 -56.1% | r=+0.488 **+9.6%** | r=+0.213 -7.4% |
| 스마트폰 | r=+0.353 +2.4% | r=+0.396 -15.4% | r=+0.357 +3.4% |
| 풀링(전체) | r=+0.453 +3.4% | r=+0.397 +0.8% | r=+0.311 +2.9% |

디지털카메라로 학습한 모델을 폰 사진에 적용하면 **평균만 답하는 것보다 MAE가 217% 나쁩니다**.
카메라별 색 재현·센서 노이즈가 절대 특징값을 옮기기 때문입니다. 기기를 모르는 입력을 받는
서비스에서는 지도학습 모델을 쓸 수 없다는 뜻이고, 그래서 기본 프로파일은 풀링 + 모델 없음입니다.

> **촬영 기기가 고정된 배포라면** `skin-metrics calibrate fit --device digital_camera`로
> 기기 전용 프로파일을 만들어 색소 모델(+17.4%)을 활성화할 수 있습니다.
> 그 프로파일을 다른 기기 사진에 쓰면 안 됩니다.
>
> 기기 간 전이를 되살리는 정공법은 **그레이카드/컬러체커 보정**입니다
> (`--reference-bbox`). 절대 색값을 기기 독립적으로 만들어 주기 때문입니다.

### 수분력 — 어디까지 왔고 무엇이 한계인가

수분은 **볼(`left_cheek`/`right_cheek`)에서만** 집계합니다. Corneometer는 이마·볼·턱에서
측정되었고 **코에서는 잰 적이 없는데** 예전에는 코를 포함한 5개 ROI를 전부 평균내고
있었습니다. 실측상 신호는 볼에만 있습니다 (held-out, 셀당 n=107):

| ROI | 폰 | 태블릿 | 디지털카메라 |
|---|---|---|---|
| left_cheek | **+0.284** | **+0.232** | **+0.385** |
| right_cheek | **+0.253** | +0.088 | **+0.258** |
| forehead | +0.051 | +0.187 | +0.165 |
| chin | +0.072 | -0.019 | +0.005 |
| nose | 장비 측정 부위 아님 (라벨 없음) | | |

특징 집합은 두 번에 걸쳐 바뀌었습니다. 수분 라운드에서 `glcm_contrast`(볼에서 단독
-0.002)와 `specular_inv`를 빼고 `scaling_index` 0.75 / `wrinkle_density` 0.25로 갔는데
(+0.241), 이후 **추출은 되고 있었지만 composite 후보에 든 적이 없는 특징 3개**
(`glcm_correlation`/`glcm_energy`/`lbp_uniformity`)를 포함해 볼 전용 전수 부분집합
탐색을 다시 돌린 결과가 현재 세트입니다:

**`scaling_index` +0.45 / `glcm_contrast` -0.30 / `lbp_uniformity` -0.25 → held-out +0.320**

- 개선 폭 +0.074, 피험자 단위 부트스트랩 95% CI [+0.004, +0.144] (P(개선)=0.98).
  train 기준으로 골라도 같은 계열이 뽑히므로 val 쇼핑이 아닙니다.
- `glcm_contrast`는 단독으로는 무상관이지만 **suppressor**(음수 가중치)로 돌아왔습니다 —
  `scaling_index`와 공유하는 조명성 분산을 상쇄합니다. 음수 가중치는 버그가 아닙니다.
- `wrinkle_density`는 피팅하면 ~-0.02로 수렴해 탈락했습니다.

**시도했지만 채택하지 않은 것:**

| 시도 | 결과 | 판단 |
|---|---|---|
| 측면 이미지(`--angles L,R`) 추가 | 볼이 1.7배 크게 잡혀 +0.053 → **+0.128** | 실재하는 효과지만 정면 재보정(+0.320)에 한참 못 미침. 사용자에게 추가 촬영을 요구할 가치 없음 |
| 정규화 해제(원본 해상도) | +0.053 → **+0.003** (전 특징 악화) | 반증. 폰의 노이즈 리덕션·샤프닝이 원본에서 가짜 텍스처로 읽힘 |
| 지도학습 릿지 | r=+0.173 | 게이트 탈락 |

**여전히 proxy입니다.** Corneometer는 각질층의 **전기 용량**을 재고, RGB 표면 광학에는
그 신호가 부분적으로만 담깁니다. +0.32는 **중간 수준의 순위 상관**이지 수분량 측정이
아닙니다. `is_estimate=True`와 "not a moisture measurement" 경고는 유지됩니다.
수분 점수는 **"이 코호트 대비 볼 텍스처가 얼마나 거친가"**로 읽으세요.

---

## 점수 해석 · 신뢰도 · 보정

각 점수는 코호트 대비 0~100 백분위입니다.

| 지표 | 높은 점수 | 방향 | 비고 |
|---|---|---|---|
| pigmentation | 색소침착 많음/짙음 | 높을수록 나쁨 | 물리 측정, 실측 일치도 +0.63 |
| erythema | 홍조 강함 | 높을수록 나쁨 | 물리 측정, **실측 검증 안 됨**(코호트 백분위) |
| hydration | **더 촉촉함** | **높을수록 좋음** | **proxy 추정 (`is_estimate=True`)**, 실측 일치도 +0.32 |

> ⚠️ **`hydration`만 방향이 반대입니다.** 내부적으로는 세 지표 모두 "condition index"
> (높을수록 뚜렷)로 계산되고, hydration의 driver·가중치·분위수 격자는 전부 **건조도**
> 기준입니다. 마지막 단계에서 `scoring.report_inverted`가 백분위를 뒤집습니다 —
> `hydration`이라는 이름의 필드는 수분을 뜻해야 하기 때문입니다("수분력 85점"이
> 건조함을 뜻하면 UI가 사용자에게 정반대를 알려주게 됩니다).
> 뒤집기는 **반드시 마지막에** 하세요. `calibration_profile.yaml`의 검증 수치와
> `calibrate/fit.py`의 `COMPOSITE_TARGET_SIGN`은 모두 건조도 방향으로 되어 있습니다.

- **`calibration_status`**: `reference` > `grayworld` > `none` 순으로 신뢰도.
  그레이카드/흰 종이를 프레임에 넣고 `--reference-bbox x,y,w,h`로 지정하면 `reference`로 상승.
- **`fitzpatrick_estimate`**: ITA로 자동 추정. 정규화가 **타입별 레퍼런스**로 수행됨.
- **촬영 팁**: 정면·균일 조명, 맨얼굴, 얼굴이 프레임의 대부분, 그림자·강한 광택 최소화.

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

- `--download-model`: 첫 실행 시 FaceLandmarker 모델(~3.8MB) 자동 다운로드 후 캐시 재사용
- `--reference-bbox`: 중립 패치 지정 → 배경 무관 화이트밸런스 + confidence 상승

---

## HTTP API

`uv sync --extra api --extra detection` 후 `skin-metrics serve`. FastAPI/uvicorn은
`api` extra에만 있으며, `skin_metrics.api` 를 import 하지 않는 한 코어 동작에 영향이 없습니다.
OpenAPI 문서는 `/docs`, 스키마는 `/openapi.json`.

### 비동기 흐름 (Spring Boot 연동)

분석은 **비동기**입니다. POST는 `request_id`를 즉시 돌려주고, 분석이 끝나면 결과 JSON이
**Redis**의 `{request_id}:analyze` / `{request_id}:diary` 키에 저장됩니다. Spring Boot는
그 키를 Redis에서 직접 읽어갑니다 — 이 API를 다시 부를 필요가 없습니다.

```
Spring Boot ──POST /analyze──▶ skin-metrics ──202 {"request_id","redis_key"}──▶ Spring Boot
                                    │ (백그라운드: 이미지 다운로드 → 분석)
                                    ▼
                          Redis  SET {request_id}:analyze = {...JSON}  (TTL 1시간)
                                    ▲
Spring Boot ────────────── GET {request_id}:analyze ──────────────────┘
```

Redis 연결은 `SKIN_METRICS_REDIS_URL`(`redis://user:password@host:port/db`)로 설정하며,
**자격증명이 들어가므로 저장소에 커밋하지 않고** compose 옆의 `.env`(gitignore됨)에 둡니다.
URL을 설정하지 않으면 프로세스 내부 메모리 스토어로 폴백합니다(개발·테스트 전용 —
다른 서비스에서는 결과가 보이지 않으며, `/healthz`의 `result_store`가 `"memory"`로 표시됨).

### `POST /analyze` · `POST /analyze/diary`

요청 본문은 두 엔드포인트가 동일:

```jsonc
{
  "image_url": "https://example.com/face.jpg",   // 필수, http(s)
  "reference_bbox": [10, 10, 40, 40]             // 선택, [x, y, w, h] 중립 패치
}
```

응답은 **202** 즉시 반환:

```jsonc
{
  "request_id": "470b634e92bd44b9abeb12accb0f0b70",
  "redis_key": "470b634e92bd44b9abeb12accb0f0b70:analyze",  // diary면 ...:diary
  "status": "processing",
  "version": "0.1.0"
}
```

**Redis에 저장되는 문서** (JSON 문자열, TTL 기본 1시간):

```jsonc
// 처리 중
{ "status": "processing", "request_id": "...", "kind": "analyze", "submitted_at": "..." }

// 완료 — /analyze 의 result (0~100)
{
  "status": "done", "request_id": "...", "kind": "analyze",
  "submitted_at": "...", "completed_at": "...",
  "result": {
    "pigmentation": 38.39,   // 높을수록 색소침착 많음
    "erythema": 55.52,       // 높을수록 붉음
    "hydration": 70.80,      // 높을수록 촉촉 (proxy)
    "confidence": { "pigmentation": 0.6, "erythema": 0.6, "hydration": 0.6 }
  }
}

// 완료 — /analyze/diary 의 result (0~10)
{
  "status": "done", "kind": "diary", /* ... */
  "result": {
    "skin_tone": 7.8,      // 피부 톤 밝기: 0=어두움, 10=매우 밝음 (ITA 선형 매핑)
    "dryness": 2.9,        // 당김·건조함 정도: 0=촉촉, 10=매우 건조 (= (100-hydration)/10)
    "redness": 5.6,        // 붉은기: 0=없음, 10=강함 (= erythema/10)
    "confidence": { "skin_tone": 0.6, "dryness": 0.6, "redness": 0.6 }
  }
}

// 실패 (다운로드 실패, 얼굴 미검출 등) — 소비자는 이걸로 '아직 처리 중'과 구분
{
  "status": "failed", /* ... */
  "error": { "code": "analysis_failed", "message": "No face detected ..." }
}
```

- `result`는 **점수와 confidence만** 담습니다. 요청 식별자·시각은 바깥 envelope에 있습니다.
- **`skin_tone`은 절대 색상 기반(ITA)**이라 카메라·조명에 민감합니다.
  `reference_bbox`(그레이카드)를 주면 기기 독립적이 됩니다.
- **`dryness`는 당김·건조함을 하나로** 제공합니다 — 당김은 감각이라 사진에서 분리 측정이
  불가능하고, 광학적으로는 동일한 건조 신호입니다.

> ⚠️ **응답에서 빠진 것은 소비하는 쪽 책임입니다.**
> `result`에는 파이프라인의 `warnings`(수분은 proxy 추정 / 그레이카드가 없어 절대 색상
> 신뢰도 낮음 / 얼굴이 작아 텍스처 신뢰도 낮음)와 **의료기기 아님 고지**가 들어가지
> 않습니다. 점수의 신뢰도를 UI에서 표현하려면 `confidence`를 쓰고, 고지는 서비스에서
> 별도로 노출하세요. 두 정보가 필요하면 CLI `analyze`가 전체 `SkinReport`를 냅니다.

### `GET /results/{key}`

Redis에 저장된 문서를 그대로 돌려주는 **디버깅용** 엔드포인트입니다
(`curl localhost:8000/results/470b…:analyze`). Spring Boot는 Redis를 직접 읽는 쪽이
빠르므로 이 엔드포인트에 의존하지 마세요. 키가 없으면 404 `result_not_found`
(TTL 만료·오타·아직 시작 전).

### `GET /healthz`

`face_model_available` / `detection_available` — 둘 다 `true`여야 분석 가능.
`result_store` — `"redis"`(정상) / `"redis_unreachable"`(URL은 있는데 접속 불가) /
`"memory"`(URL 미설정 — **Spring이 결과를 못 봅니다**). 배포 후 이 값부터 확인하세요.

### 오류 응답

제출 시점에 판별 가능한 문제만 동기 4xx로 응답하고(아래 표), 다운로드·분석 중의 실패는
**Redis 문서의 `status: "failed"`** 로 전달됩니다(`error.code`는 같은 코드 체계).

| status | code | 상황 |
|---|---|---|
| 400 | `invalid_scheme` / `invalid_url` / `dns_error` | URL 자체가 잘못됨 |
| 403 | `blocked_host` | URL이 사설/루프백/링크로컬 주소로 해석됨 |
| 404 | `result_not_found` | `GET /results/{key}` — 키 없음/만료 |
| 422 | `invalid_request` | 요청 본문 검증 실패 |
| 503 | `result_store_unavailable` | Redis에 접수 기록조차 못 씀 |

백그라운드 실패로 문서에 기록되는 code: `decode_error` / `empty_body` / `image_too_large` /
`upstream_error` / `fetch_error` / `fetch_timeout` / `too_many_redirects` /
`analysis_failed`(얼굴 미검출·전 ROI 탈락) / `face_model_missing` /
`detection_unavailable` / `internal_error`.

### 보안 가드 (`api/fetch.py`)

서버가 **사용자가 준 URL로 직접 요청**하므로 SSRF 경계입니다:
scheme allow-list(http/https) → DNS 해석 결과가 사설·루프백·링크로컬·예약 대역이면 거부
(`169.254.169.254` 같은 클라우드 메타데이터 포함) → 리다이렉트는 매 홉 재검증 + 횟수 제한 →
본문은 스트리밍하며 바이트 상한에서 중단 → 디코딩 시 픽셀 수 상한(압축 폭탄 방어).
남는 위험은 DNS rebinding(검증과 실제 연결이 각각 해석)이며, 이를 위협모델에 포함한다면
egress 프록시에서 allow-list 하는 편이 낫습니다.

### 환경변수

| 변수 | 기본값 | 의미 |
|---|---|---|
| `SKIN_METRICS_API_CONFIG` | 패키지 기본 `config.yaml` | 설정 YAML 경로 |
| `SKIN_METRICS_FACE_MODEL` | 캐시 경로 | FaceLandmarker `.task` 경로 |
| `SKIN_METRICS_API_DOWNLOAD_MODEL` | `0` | 시작 시 모델 자동 다운로드 |
| `SKIN_METRICS_API_MAX_BYTES` | `20971520` (20MB) | 다운로드 바이트 상한 |
| `SKIN_METRICS_API_MAX_PIXELS` | `40000000` | 디코딩 픽셀 **하드 상한** (압축폭탄 방어, 초과 시 413) |
| `SKIN_METRICS_API_ANALYSIS_MAX_PIXELS` | `16000000` | 분석 픽셀 **예산**. 초과 시 거절 대신 **축소**. 메모리가 MP당 약 63MB로 늘어나므로 이 값이 피크 메모리를 결정 |
| `SKIN_METRICS_API_FETCH_TIMEOUT` | `10.0` | 다운로드 타임아웃(초) |
| `SKIN_METRICS_API_MAX_REDIRECTS` | `3` | 리다이렉트 허용 횟수 |
| `SKIN_METRICS_API_ALLOW_PRIVATE_HOSTS` | `0` | **개발 전용** — SSRF 가드 해제 |
| `SKIN_METRICS_API_MAX_CONCURRENCY` | `2` | 동시 분석 수 (파이프라인은 CPU 바운드) |
| `SKIN_METRICS_REDIS_URL` | (없음) | `redis://user:password@host:port/db`. **`.env`에만** 두고 커밋 금지. 미설정 시 메모리 스토어 폴백 |
| `SKIN_METRICS_RESULT_TTL` | `3600` | 결과가 Redis에 남아 있는 시간(초) |

> `SKIN_METRICS_BIND`(기본 `127.0.0.1`)과 `SKIN_METRICS_PORT`(기본 `8000`)는 API가 아니라
> **compose가 읽는 변수**로, 호스트 쪽 바인딩 주소·포트를 정합니다.

> **응답 시간·메모리**: 분석은 동기·CPU 바운드라 워커 스레드 + 세마포어로 실행됩니다.
> 얼굴 크기 정규화가 들어간 뒤로는 **입력 해상도가 시간에 거의 영향을 주지 않습니다**
> (6.5MP 6.4초 → 40MP 8.9초). 반면 **메모리는 메가픽셀당 약 63MB로 선형 증가**하므로
> 실질적인 제약은 시간이 아니라 메모리입니다 — 자세한 수치와 인스턴스 사이징은
> [AWS EC2 배포](#aws-ec2-배포) 참고. 트래픽이 늘면 큐 + 작업 ID 방식이 필요합니다.

---

## Docker

```bash
docker build -t skin-metrics-api:0.1.0 .        # 기본 = api 타깃 (약 1.7GB)
docker run --rm -p 127.0.0.1:8000:8000 skin-metrics-api:0.1.0
curl localhost:8000/healthz
```

**재배포 스크립트** — 이전 스택 종료 → 빌드 → 재기동 → 헬스체크 통과까지 한 번에:

```bash
./redeploy.sh                          # 코드 고친 뒤 이거 하나면 끝
./redeploy.sh --no-cache               # 의존성까지 처음부터 다시 (약 5분)
./redeploy.sh --logs                   # 뜨고 나서 로그 따라가기
SKIN_METRICS_PORT=8100 ./redeploy.sh   # 8000이 이미 쓰이는 경우
```

포트를 이미 쓰는 프로세스가 있으면 **죽이지 않고 누가 쓰는지 알려주고 멈춥니다**.
이미지 정리도 `skin-metrics-api` 라벨이 붙은 것만 대상으로 해서, 다른 프로젝트의 컨테이너·
이미지·빌드 캐시는 건드리지 않습니다.

compose를 직접 쓸 때:

```bash
docker compose up -d --build      # 빌드 + 백그라운드 실행 (코드 수정 후엔 항상 --build)
docker compose up -d              # 이미지가 이미 있으면 빌드 없이 실행만
docker compose logs -f api
docker compose down

SKIN_METRICS_PORT=8100 docker compose up -d   # 8000 포트가 이미 쓰이는 경우
```

> **`up -d`는 이미지가 없을 때만 빌드합니다.** 이미 `skin-metrics-api:0.1.0`이 있으면
> 소스를 고쳐도 예전 이미지를 그대로 띄웁니다. 코드 변경을 반영하려면 `--build`를 붙이세요.
> `trainer` 서비스는 `full` 프로파일이라 `up`으로는 뜨지 않습니다.

FaceLandmarker 모델(~3.8MB)이 **빌드 시점에 이미지 안에 포함**되므로 컨테이너는 시작할 때
네트워크가 필요 없습니다(빌드에는 필요). 밖으로 나가는 통신은 `/analyze`의 이미지 URL
다운로드뿐입니다.

### 빌드 타깃 2종

| 타깃 | 내용 | 크기 | 용도 |
|---|---|---|---|
| `api` (기본) | Phase 1 + FastAPI. torch 없음 | 1.72GB | 배포용 |
| `full` | 위 + `dl` extra(torch/torchvision/timm/albumentations/pandas) | 2.88GB | 컨테이너에서도 Phase 2 학습 |

```bash
docker build --target full -t skin-metrics-api:0.1.0-full .
docker run --rm skin-metrics-api:0.1.0-full skin-metrics train --dummy --mode ranking --epochs 1
# compose 로도 동일:
docker compose --profile full run --rm trainer train --dummy --mode ranking
```

두 타깃은 소스와 OS 레이어를 공유하고 **가상환경만 다릅니다**. 배포 이미지에 torch가 들어가지
않도록 기본 타깃을 `api`로 두었고, Phase 2가 필요하면 `full`을 쓰면 로컬과 동일하게 동작합니다.

**linux에서는 torch/torchvision을 CPU 전용 인덱스**(`https://download.pytorch.org/whl/cpu`)에서
받도록 `pyproject.toml`에 설정돼 있습니다. PyPI 기본 휠은 nvidia-* CUDA 패키지를 끌고 오는데,
GPU 없는 컨테이너에서 `import torch` 가 **SIGILL로 죽습니다**(`torch._preload_cuda_deps`).
macOS 로컬은 영향 없이 기존 휠을 그대로 씁니다. 이 설정으로 이미지가 9.61GB → 2.88GB가 됐습니다.

> `torchvision`이 `dl` extra에 명시돼 있는 이유: `[tool.uv.sources]`는 **직접 의존성에만**
> 적용됩니다. timm의 전이 의존성으로 두면 torch만 `+cpu`가 되어 ABI가 어긋나고
> `operator torchvision::nms does not exist` 로 학습이 실패합니다.

> 빌드에는 디스크 여유가 넉넉해야 합니다(**15GB 이상 권장**). 부족하면 빌드가 멈추면서 도커
> 데몬까지 응답하지 않을 수 있습니다(그 경우 Docker Desktop 재시작).

### 이미지 올리기 (레지스트리 push)

```bash
# 단일 아키텍처
docker build -t <registry>/<user>/skin-metrics-api:0.1.0 .
docker push <registry>/<user>/skin-metrics-api:0.1.0

# amd64 + arm64 동시 (mediapipe 1.0.0 은 manylinux x86_64·aarch64 휠 모두 제공)
docker buildx build --platform linux/amd64,linux/arm64 \
  -t <registry>/<user>/skin-metrics-api:0.1.0 --push .
```

`ghcr.io/astral-sh/uv` 이미지를 쓰지 않고 **PyPI의 uv를 설치**하도록 되어 있어, 빌드는
Docker Hub만 있으면 됩니다(일부 네트워크에서 ghcr 익명 pull이 막힙니다).

### 이미지에 들어가는 것 / 안 들어가는 것

`.dockerignore`가 **전부 차단 후 필요한 것만 허용**하는 방식이라, 빌드 컨텍스트에는
`pyproject.toml` · `uv.lock` · `README.md` · `skin_metrics/` 만 들어갑니다.
`data/`의 얼굴 사진, `report*.json`, `tests/`, `.venv/`, `.git/`, `*.task`는 **어떤 경로로도
이미지에 포함되지 않습니다**. 나중에 새 파일이 생겨도 기본이 차단이라 안전합니다.

### AWS EC2 배포

⚠️ **아키텍처 주의**: 맥에서 `docker build`로 만든 기본 이미지는 arm64라서 x86_64 EC2에
올리면 `exec format error`로 죽습니다. GitHub Actions가 amd64로 빌드해 주므로
**방법 A(ghcr pull)를 쓰면 이 문제가 없습니다.**

#### 1. 인스턴스 선택

| 타입 | vCPU / RAM | 비고 |
|---|---|---|
| `t3.medium` | 2 / 4GB | **최소**. 버스터블이라 연속 요청 시 CPU 크레딧 소진 주의 |
| **`t3.large`** | 2 / 8GB | **권장** — 메모리 여유가 있어 동시 요청에 안전 |
| `c6i.large` | 2 / 4GB | 비버스터블. CPU 성능이 일정해야 하면 |
| `t4g.large` | 2 / 8GB | **arm64(Graviton)**, 약 20% 저렴. 이 저장소는 arm64에서도 빌드·검증됨 |

- **EBS 루트 볼륨 30GB** (기본 8GB로는 빌드 캐시가 안 들어갑니다)
- **보안 그룹**: 인바운드 TCP `8000`(또는 프록시를 쓸 경우 80/443). 아웃바운드는 기본값
  그대로 두세요 — `/analyze`가 이미지 URL을 받아오려면 외부로 나갈 수 있어야 합니다.

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

> Ubuntu면 `sudo apt install -y docker.io docker-compose-v2 git` 로 대체.

#### 3-A. 배포 — GitHub Packages에서 pull (권장)

이미지는 **GitHub Actions가 빌드해 ghcr.io에 올립니다**
(`.github/workflows/publish-image.yml`). `main`에 푸시하면 자동으로 돌고,
Actions 탭에서 "Publish image" → *Run workflow* 로 수동 실행도 됩니다.
러너가 amd64 네이티브라 맥에서 크로스 빌드하는 것보다 훨씬 빠릅니다.

```
ghcr.io/likelion14-hackathon/skin-metrics-api:latest
ghcr.io/likelion14-hackathon/skin-metrics-api:<commit-sha>   # 롤백용
```

**저장소가 private이면 패키지도 private**이므로 EC2에서 먼저 로그인해야 합니다.
`read:packages` 스코프 PAT을 [github.com/settings/tokens](https://github.com/settings/tokens)에서
만들어(classic, `read:packages`만 체크):

```bash
echo '<PAT>' | docker login ghcr.io -u <github-사용자명> --password-stdin
```

> 패키지를 **public**으로 바꾸면(패키지 페이지 → Package settings → Change visibility)
> EC2에서 로그인 없이 pull할 수 있습니다. 다만 이미지 안에 소스 코드가 들어 있으므로
> 저장소를 private으로 두는 이유가 있다면 위의 PAT 방식을 쓰세요.

EC2에서 실행:

```bash
docker pull ghcr.io/likelion14-hackathon/skin-metrics-api:latest
docker run -d --name skin-metrics --restart unless-stopped \
  -p 0.0.0.0:8000:8000 \
  -e SKIN_METRICS_REDIS_URL='redis://default:<password>@<redis-host>:<port>/0' \
  -e SKIN_METRICS_API_ANALYSIS_MAX_PIXELS=16000000 \
  -e MPLCONFIGDIR=/tmp/mpl \
  --read-only --tmpfs /tmp:rw,size=64m \
  --memory 4g --cpus 2 \
  ghcr.io/likelion14-hackathon/skin-metrics-api:latest
```

> Spring Boot가 **같은 인스턴스**에서 돌아 localhost로만 부른다면
> `-p 127.0.0.1:8000:8000`으로 바꾸세요. 외부에 전혀 노출되지 않아 인증이 없는
> 현재 상태에서 가장 안전합니다.

업데이트는 `docker pull ... && docker rm -f skin-metrics && docker run ...` 입니다.

> **Graviton(t4g/c7g)에 배포한다면** 기본 워크플로는 amd64만 만듭니다.
> Actions에서 *Run workflow* 시 platforms 입력을 `linux/amd64,linux/arm64`로 주세요
> (arm64는 러너에서 에뮬레이션되어 빌드가 몇 배 느립니다).

#### 3-B. 배포 — 저장소 클론 + 인스턴스에서 빌드

```bash
git clone https://github.com/likelion14-hackathon/proof-face.git
cd proof-face
echo 'SKIN_METRICS_REDIS_URL=redis://default:<password>@<redis-host>:<port>/0' > .env

# 0.0.0.0 바인딩이 있어야 인스턴스 밖에서 접근됩니다 (기본은 루프백)
SKIN_METRICS_BIND=0.0.0.0 ./redeploy.sh
```

빌드 약 3~5분, 그 뒤 헬스체크 통과까지 자동으로 기다립니다. 확인:

```bash
curl http://<EC2-퍼블릭-IP>:8000/healthz
curl -X POST http://<EC2-퍼블릭-IP>:8000/analyze/diary \
  -H 'content-type: application/json' \
  -d '{"image_url":"https://example.com/face.jpg"}'
# → {"request_id": "...", "redis_key": "...:diary", ...}
curl http://<EC2-퍼블릭-IP>:8000/results/<request_id>:diary
```

`curl <EC2-IP>:8000/healthz`의 **`result_store`가 `"redis"`인지 꼭 확인**하세요 —
`"memory"`면 Redis URL이 전달되지 않은 것이고, Spring Boot가 결과를 읽을 수 없습니다.

재배포: 방법 A는 새 tar를 올려 `docker load` 후 컨테이너 재생성, 방법 B는
`git pull && SKIN_METRICS_BIND=0.0.0.0 ./redeploy.sh`. 두 방법 모두
`restart: unless-stopped`라 인스턴스를 재부팅해도 컨테이너가 살아납니다.

#### 4. 성능·메모리 (실측)

이 맥에서 컨테이너 CPU 2개 제한으로 잰 값입니다. EC2 x86 vCPU는 더 느리므로
**시간은 2~3배로 잡으세요**(메모리는 아키텍처와 무관하게 동일).

| 입력 | 소요 시간 | 피크 메모리 |
|---|---|---|
| 6.5MP | 6.4s | 647MB |
| 12MP (폰 기본) | 6.8s | 1.0GB |
| 24MP | 8.1s | 1.7GB |
| 40MP | 8.9s | 2.5GB |

메모리는 **메가픽셀당 약 63MB**로 선형 증가합니다(내부 연산이 float64). 그래서
`SKIN_METRICS_API_ANALYSIS_MAX_PIXELS`(기본 **16MP**)를 넘는 이미지는 **거절하지 않고
분석 직전에 축소**합니다 — 요청 1건이 약 1.2GB, 동시 2건이 2.5GB로 묶여 compose의
`memory: 4g` 안에 들어옵니다. 이 예산이 없으면 40MP 사진 2장이 동시에 들어올 때
**5GB를 써서 컨테이너가 OOM으로 죽습니다**.

축소해도 정확도 손실은 없습니다. 파이프라인이 어차피 모든 얼굴을
`normalization.target_eye_span_px`(512px)로 정규화하기 때문입니다. 축소 후 얼굴이 너무
작아지면 기존 `under_resolved` 경고가 그대로 동작해 수분 신뢰도를 낮춥니다.

#### 5. 해커톤 배포 시 반드시 알 것

- **인증·레이트리밋이 없습니다.** 퍼블릭 IP에 그대로 열면 누구나 호출할 수 있고,
  요청 1건이 CPU를 수 초씩 점유합니다. 데모 기간엔 **보안 그룹의 소스 IP를 팀/심사장
  대역으로 제한**하는 게 가장 간단한 방어입니다.
- **HTTPS가 아닙니다.** 프론트엔드가 HTTPS면 브라우저가 mixed content로 차단합니다.
  Caddy/nginx 리버스 프록시 + Let's Encrypt를 앞에 두거나, 프론트도 HTTP로 띄우세요.
- `SKIN_METRICS_API_ALLOW_PRIVATE_HOSTS`는 **절대 켜지 마세요.** EC2에서 켜면
  `169.254.169.254`(인스턴스 메타데이터 = IAM 자격증명)로 요청을 보낼 수 있게 됩니다.
  기본값 `0` 그대로 두면 막힙니다.

### 운영 시 확인할 것

- `docker-compose.yml`은 기본적으로 **127.0.0.1 에만 바인딩**합니다(`SKIN_METRICS_BIND`로
  변경). `/analyze` 앞에 인증·레이트리밋이 없으므로, 외부에 열려면 리버스 프록시에서
  인증·요청 제한을 두세요.
- `SKIN_METRICS_API_ALLOW_PRIVATE_HOSTS=1`은 **개발 전용**입니다. 켜면 컨테이너가
  같은 네트워크의 내부 서비스로 요청을 보낼 수 있게 됩니다(SSRF).
- 분석은 CPU 바운드입니다. `SKIN_METRICS_API_MAX_CONCURRENCY`와 컨테이너 CPU 한도를
  같이 올리세요(compose 기본: 2 CPU / 동시 2건).
- 컨테이너는 비루트(uid 10001)로 실행되며 compose에서 `read_only: true` 로 뜹니다.

---

## 테스트

```bash
uv run pytest -q          # 112 passed
```

- **합성 이미지·합성 랜드마크** 기반이라 Phase 1 테스트는 `detection`/`dl` extra 없이 실행.
- `tests/test_models.py`는 torch 미설치 시, `tests/test_api.py`는 fastapi 미설치 시
  `importorskip`으로 자동 스킵.
- API 테스트는 루프백 HTTP 서버를 띄워 실제 다운로드 경로까지 태우며 외부 네트워크는 쓰지 않음.
- `tests/test_calibrate.py`는 **합성 테이블·합성 코퍼스**로 돌아가므로 43GB 데이터셋 없이
  실행됩니다.
- 커버리지: 색보정 왕복/CCM 복원/D65 화이트/마스크 기반 gray-world, ITA·멜라닌·홍반 공식·가드,
  헤모글로빈 ICA(정상/퇴화), 텍스처·주름 프록시, ROI 기하·마스킹, 정규화·스키마,
  end-to-end 파이프라인, 보정 툴링(코퍼스 인덱싱·릿지 왕복·채택 게이트·분위수 매핑·프로파일 병합),
  Phase 2 forward/GRL/학습 루프(regression·ranking).

---

## 정확도를 더 올리려면 — 데이터 확보 가이드

현재 프로파일은 기존 코호트에서 짜낼 수 있는 것을 거의 다 짜냈습니다(전수 특징 부분집합
탐색까지 완료). 다음 단계는 전부 **새 데이터**가 필요하며, 지표별로 필요한 데이터와
구할 수 있는 곳이 다릅니다.

| 지표 | 병목 | 필요한 데이터 | 어디서 |
|---|---|---|---|
| **홍조** | 실측 라벨이 0장 → 검증 자체가 불가능 | ① 전문의 CEA 등급(0~4) 사진 채점 ② Mexameter/VISIA red 실측 | ①이 최선: **장비 불필요**, 기존 사진 200~300장 + 피부과 전문의 2명. ②는 VISIA 보유 피부과·에스테틱 제휴 |
| **수분** | 표면 광학의 물리적 한계(+0.32) | 서비스 타깃 폰으로 찍은 정면 사진 + **Corneometer 동시 측정** (볼 좌우, 100명~) | 공개 데이터셋 없음. Corneometer CM825 대여/제휴 측정. AI-Hub에 추가 수분 실측 데이터셋 없음(028이 유일) |
| **색소** | 지도학습이 기기 종속으로 탈락 | 타깃 폰 **단일 기종**으로 찍은 사진 + 전문가 등급 또는 장비 스팟 개수 | 공개 데이터셋 없음. 028 재활용 가능: `calibrate fit --device phone`(단, 타 기기 입력에 쓰면 안 됨) |
| **피부 톤** (`/analyze/diary`) | 절대 색상이라 카메라·조명 민감 | 데이터가 아니라 **그레이카드**: 요청에 `reference_bbox` 포함 | 촬영 UI에 그레이카드/흰 종이 가이드 추가 |
| **전 지표 (Phase 2)** | 딥러닝 미학습 | 이미 보유 — 028 라벨 38GB | `calibrate/aihub.py`의 `iter_roi_rows` → `train --data labels.csv --mode regression` |

**공개 데이터셋에 기대지 마세요**: 얼굴 정면 표준 촬영 + 장비 실측이 붙은 공개 데이터는
사실상 없습니다. Fitzpatrick17k(1.6만 장)·SCIN(1만 장+)은 병변 클로즈업/크라우드소싱에
진단명 라벨이라 이 파이프라인의 타깃(정면 얼굴, 중증도·실측값)과 맞지 않고, VISIA+CEA
임상 코호트(1,001명 규모)는 병원 보유라 비공개입니다. AI-Hub에서 얼굴 + 피부 실측이 붙은
데이터셋은 028이 유일합니다.

---

## 알려진 한계 · TODO

- **레퍼런스 코호트가 한국인 성인 위주**: `calibration_profile.yaml`은 AI-Hub 028
  코호트(대부분 타입 3~4)로 피팅됐습니다. 타입 1~2, 5~6 버킷은 표본이 부족해
  `default` 분포로 폴백합니다. 다른 인구집단이 주 대상이면 그 코호트로 재피팅하세요.
- **수분은 실측과 거의 무관**: Corneometer(각질층 전기 용량)는 RGB 표면 광학으로
  예측되지 않습니다(held-out r≈0.18). 지도학습 모델은 채택 게이트에서 탈락했고,
  수분 점수는 **"코호트 대비 텍스처 거칠기 백분위"**입니다. 절대 수분값이 필요하면
  접촉식 장비를 쓰세요.
- **홍조는 실측 검증 안 됨**: 이 코호트에 Mexameter 홍반값이 없습니다. 코호트 백분위로만
  정규화되어 있어 **상대 비교용**입니다.
- **Phase 2 절대값**: 실측 라벨 CSV가 이제 존재하지만(`calibrate/aihub.py`), 아직 실학습은
  돌리지 않았습니다.
- **Tsumura 기준 벡터**(`erythema._HEMOGLOBIN_DIR/_MELANIN_DIR`)는 근사값이고, FastICA가
  수렴 실패하는 이미지가 꽤 됩니다 → 카메라별 측정 흡광 스펙트럼으로 교체 권장(`TODO`).
- **동일 조건 시계열 `compare`** 가 여전히 단일 절대 점수보다 신뢰도가 높습니다.
