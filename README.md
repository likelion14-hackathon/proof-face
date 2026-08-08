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
- [점수 해석 · 신뢰도 · 보정](#점수-해석--신뢰도--보정)
- [CLI 레퍼런스](#cli-레퍼런스)
- [HTTP API](#http-api)
- [테스트](#테스트)
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
skin-metrics analyze data/test2.jpg --download-model --output report.json

# 이후 실행: 모델 캐시 재사용 (플래그 불필요)
skin-metrics analyze data/test2.jpg --output report.json

# 그레이카드/흰 종이가 프레임에 있으면 그 영역 지정 → 보정 신뢰도 상승
skin-metrics analyze data/test2.jpg --reference-bbox 10,10,40,40 --output report.json

# 두 시점 비교 (같은 사람·같은 조건 촬영 권장)
skin-metrics compare data/test-image.jpg data/test2.jpg

# Phase 2 딥러닝 스캐폴드 (라벨 없이 더미로 학습 루프 검증)
skin-metrics train --dummy --mode ranking --epochs 3

# HTTP API 서버 (docs: http://127.0.0.1:8000/docs)
skin-metrics serve --download-model
curl -X POST http://127.0.0.1:8000/analyze \
  -H 'content-type: application/json' \
  -d '{"image_url": "https://example.com/face.jpg"}'
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
│   ├── normalize.py         # Fitzpatrick 타입별 백분위 정규화, compare()
│   └── schema.py            # pydantic SkinReport / MetricScore
├── models/                  # Phase 2: dataset(+dummy) / network / train
├── api/                     # HTTP API (FastAPI)
│   ├── app.py               # /healthz, /analyze 엔드포인트 + lifespan
│   ├── fetch.py             # 이미지 URL 다운로드 (SSRF·크기 가드)
│   ├── schemas.py           # 요청/응답 pydantic 모델
│   └── settings.py          # SKIN_METRICS_API_* 환경변수 설정
├── pipeline.py              # image -> SkinReport 오케스트레이션
├── config.py / config.yaml  # 설정 로더 / 임계값·레퍼런스 분포
└── cli.py                   # typer: analyze / compare / train / serve
tests/                       # 합성 이미지·랜드마크 기반 단위 테스트 (81개)
```

---

## Phase 1: 물리 기반 파이프라인 (기술 상세)

### 1. 색 보정 — `calibration/color.py`

| 함수 | 내용 |
|---|---|
| `linearize_srgb(img)` | sRGB EOTF 역변환(감마 제거). `s≤0.04045 ? s/12.92 : ((s+0.055)/1.055)^2.4` |
| `encode_srgb(lin)` | 역변환(선형→sRGB). 왕복 오차 < 1e-4 |
| `white_balance_grayworld(img)` | 채널 평균을 회색으로 정규화. **약한 fallback → `success=False`** |
| `white_balance_from_reference(img, bbox)` | 프레임 내 중립 패치(그레이카드/흰 종이) 평균으로 채널 게인 산출. 너무 어둡/클리핑 시 거부 |
| `estimate_ccm(detected, reference)` | 24패치 컬러체커 → **최소제곱 3×3 색보정 행렬**. `M = lstsq(detected, reference)`, RMS 잔차 반환 |
| `apply_ccm(img, M)` | `rgb @ M` 적용 |
| `rgb_to_lab(img)` | **colour-science**로 sRGB primaries + **D65** whitepoint 기준 CIELab. `L*∈[0,100]` |
| `calibrate_image(...)` | 오케스트레이션: 선형화 → (CCM) → WB. `status ∈ {reference, grayworld, none}` 와 `success` 반환 |

- **보정 신뢰도 규약**: `reference`(중립 패치 성공) 또는 CCM 적용 시 `success=True`,
  grayworld만이면 `False`. 이 값이 downstream `confidence`를 낮춥니다.
- **배경 영향**: grayworld는 이미지 전체 평균을 쓰므로 강한 색 배경에 영향을 받습니다.
  지표 계산 자체는 ROI 내부만 쓰므로 배경 무관. 정확도의 핵심 레버는 **그레이카드
  `--reference-bbox`** 또는 컬러체커 CCM입니다.

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
- **`composite_raw(metric, subfeatures, config)`**: 서브피처를 `config.yaml`의 anchor
  (mean/std)로 z-score → **가중합**(사용 가능한 가중치로 재정규화).
- **`score_from_raw(raw, metric, fitz, config)`**: 해당 **Fitzpatrick 타입 레퍼런스 분포**의
  mean/std로 z → clip → **정규 CDF**로 0~100 백분위. 타입별 분리로 어두운 피부 편향 완화.
- **`compare(current, baseline, min_delta)`**: 지표별 변화량·방향·유의 여부(시계열).
- **출력 스키마 `schema.py`** (pydantic):

```python
class MetricScore(BaseModel):
    score: float       # 0-100 condition index (높을수록 뚜렷)
    confidence: float  # 0-1
    raw_features: dict
    is_estimate: bool  # hydration은 항상 True

class SkinReport(BaseModel):
    pigmentation / erythema / hydration: MetricScore
    roi_breakdown: dict            # ROI별 valid_ratio + 지표
    calibration_status: Literal["reference","grayworld","none"]
    fitzpatrick_estimate: int      # 1-6
    warnings: list[str]
    disclaimer: str                # 의료기기 아님 고지 (자동 포함)
```

- **confidence 계산**(`pipeline.py`): `calibration(reference 1.0 / grayworld 0.6 / none 0.3)
  × (유효 ROI 수 / 5)`. 홍조는 헤모글로빈 분리 실패 시 ×0.7.

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

## 점수 해석 · 신뢰도 · 보정

각 점수는 0~100 **"condition index"**, **높을수록 해당 상태가 뚜렷**:

| 지표 | 높은 점수 | 비고 |
|---|---|---|
| pigmentation | 색소침착 많음/짙음 | 물리 측정 |
| erythema | 홍조 강함 | 물리 측정 |
| hydration | **더 건조함** | **proxy 추정 (`is_estimate=True`)** |

- **`calibration_status`**: `reference` > `grayworld` > `none` 순으로 신뢰도.
  그레이카드/흰 종이를 프레임에 넣고 `--reference-bbox x,y,w,h`로 지정하면 `reference`로 상승.
- **`fitzpatrick_estimate`**: ITA로 자동 추정. 정규화가 **타입별 레퍼런스**로 수행됨.
- **촬영 팁**: 정면·균일 조명, 맨얼굴, 얼굴이 프레임의 대부분, 그림자·강한 광택 최소화.

---

## CLI 레퍼런스

```bash
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

### `POST /analyze`

```jsonc
// 요청
{
  "image_url": "https://example.com/face.jpg",   // 필수, http(s)
  "reference_bbox": [10, 10, 40, 40]             // 선택, [x, y, w, h] 중립 패치
}
// 응답 200
{
  "report":  { /* SkinReport: CLI analyze 의 JSON 과 동일 (disclaimer 포함) */ },
  "source":  { "url": ..., "final_url": ..., "content_type": "image/jpeg",
               "bytes": 15222620, "width": 4912, "height": 7360 },
  "elapsed_ms": 65987.25,
  "version": "0.1.0"
}
```

### `GET /healthz`

`status` / `version` / `face_model_available`(모델 파일 존재) / `detection_available`
(mediapipe import 가능) — 둘 다 `true` 여야 실제 분석이 가능합니다.

### 오류 응답

모든 4xx·5xx는 `{"error": {"code", "message"}}` 형식입니다.

| status | code | 상황 |
|---|---|---|
| 400 | `invalid_scheme` / `invalid_url` / `dns_error` / `decode_error` / `empty_body` | URL·응답 본문 문제 |
| 403 | `blocked_host` | URL이 사설/루프백/링크로컬 주소로 해석됨 |
| 413 | `image_too_large` | 바이트 또는 픽셀 상한 초과 |
| 422 | `invalid_request` / `analysis_failed` | 요청 검증 실패 / 얼굴 미검출·전 ROI 탈락 |
| 502 | `upstream_error` / `fetch_error` / `too_many_redirects` | 이미지 호스트 실패 |
| 503 | `face_model_missing` / `detection_unavailable` | 서버에 모델·mediapipe 없음 |
| 504 | `fetch_timeout` | 다운로드 타임아웃 |

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
| `SKIN_METRICS_API_MAX_PIXELS` | `40000000` | 디코딩 픽셀 상한 |
| `SKIN_METRICS_API_FETCH_TIMEOUT` | `10.0` | 다운로드 타임아웃(초) |
| `SKIN_METRICS_API_MAX_REDIRECTS` | `3` | 리다이렉트 허용 횟수 |
| `SKIN_METRICS_API_ALLOW_PRIVATE_HOSTS` | `0` | **개발 전용** — SSRF 가드 해제 |
| `SKIN_METRICS_API_MAX_CONCURRENCY` | `2` | 동시 분석 수 (파이프라인은 CPU 바운드) |

> **응답 시간**: 분석은 동기·CPU 바운드라 워커 스레드 + 세마포어로 실행됩니다.
> 36MP(4912×7360) 사진 기준 **1건에 약 66초**가 걸리므로, 상한 픽셀을 낮추거나
> 클라이언트에서 리사이즈해 올리는 것을 권장합니다(단, 리샘플링은 텍스처 기반 수분력
> 프록시 값을 바꿉니다). 트래픽이 있다면 큐 + 작업 ID 방식으로 바꾸는 편이 좋습니다.

---

## 테스트

```bash
uv run pytest -q          # 81 passed
```

- **합성 이미지·합성 랜드마크** 기반이라 Phase 1 테스트는 `detection`/`dl` extra 없이 실행.
- `tests/test_models.py`는 torch 미설치 시, `tests/test_api.py`는 fastapi 미설치 시
  `importorskip`으로 자동 스킵.
- API 테스트는 루프백 HTTP 서버를 띄워 실제 다운로드 경로까지 태우며 외부 네트워크는 쓰지 않음.
- 커버리지: 색보정 왕복/CCM 복원/D65 화이트, ITA·멜라닌·홍반 공식·가드,
  헤모글로빈 ICA(정상/퇴화), 텍스처·주름 프록시, ROI 기하·마스킹, 정규화·스키마,
  end-to-end 파이프라인, Phase 2 forward/GRL/학습 루프(regression·ranking).

---

## 알려진 한계 · TODO

- **레퍼런스 분포는 placeholder**: `config.yaml`의 `reference.*` / `composite.*` anchor는
  문헌에 느슨히 근거한 초기값. **대상 카메라·집단 데이터로 재추정 필요**(파일 내 `TODO`).
  → 절대 점수보다 **동일 조건 시계열 `compare`** 가 더 신뢰 가능.
- **Phase 2 절대값**: 실측 라벨 없이는 무의미. 라벨 CSV 확보 시 `--data`로 전환.
- **Tsumura 기준 벡터**(`erythema._HEMOGLOBIN_DIR/_MELANIN_DIR`)도 근사값 → 카메라별
  측정 흡광 스펙트럼으로 교체 권장(`TODO`).
- **수분력은 원리상 프록시**: RGB로 직접 측정 불가. 절대 수분값이 필요하면 접촉식 장비 + Phase 2.
