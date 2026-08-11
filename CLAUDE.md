# CLAUDE.md

이 파일은 이 저장소에서 작업하는 Claude Code(및 새 세션/기여자)를 위한 안내서입니다.
사용자용 문서는 [README.md](README.md)를, 여기서는 **개발 환경·규약·현재 상태·이어갈 지점**을
다룹니다.

## 프로젝트 한 줄 요약

얼굴 이미지 1장에서 **색소침착 / 홍조 / 수분력(proxy)** 를 0~100으로 산출.
**Phase 1(물리 기반, 학습 불필요)** 이 실제 "이미지→지수" 엔진이고,
**Phase 2(딥러닝)** 는 실측 라벨이 생기면 붙일 수 있는 **완전히 도는 스캐폴드**.

## 현재 상태 (2026-08-11 기준)

### 최신 라운드 (특징 부분집합 탐색 + `/analyze/simple`)

1. **수분 +0.241 → +0.320 (held-out 볼 642행)** — 추출은 되고 있었지만 composite 후보에
   든 적이 없는 특징 3개(`glcm_correlation`/`glcm_energy`/`lbp_uniformity`)를 포함해
   볼 전용 전수 부분집합 탐색을 돌린 결과. 최종 세트
   **`scaling_index` +0.45 / `glcm_contrast` -0.30 / `lbp_uniformity` -0.25**.
   개선 +0.074, 피험자 부트스트랩 95% CI [+0.004, +0.144]; train 기준 선택도 같은 계열이라
   val 쇼핑 아님. `glcm_contrast`는 단독 무상관이지만 **suppressor**(음수 가중치)로
   복귀 — 음수 가중치는 버그가 아닙니다. `wrinkle_density`는 ~-0.02로 수렴해 탈락.
   기기별 +0.292(폰) / +0.389(태블릿) / +0.339(DSLR). **재추출 불필요였음** — face CSV가
   이미 볼 집계로 전 특징을 들고 있음.
2. **`POST /analyze/simple` 추가** — 소비자용 0~10 점수 3개: `skin_tone`(ITA -30°~55° →
   0~10 선형, 절대 색상이라 기기 민감), `dryness`(당김·건조함 통합, `(100-hydration)/10`),
   `redness`(`erythema/10`). 매핑은 `api/schemas.py`의
   `SimpleAnalyzeResponse.from_report`. `app.py`는 두 엔드포인트가 fetch→파이프라인→오류
   매핑을 `_run_report` + `AnalysisError` 핸들러로 공유하도록 리팩토링.
   컨테이너에서 실사진 검증 완료(cohort 정면 → tone 7.8 / dryness 2.9 / redness 5.6).
3. **`/analyze`도 같은 평평한 envelope으로 통일** (사용자 요청) — 중첩 `report`/`source`
   대신 0~100 점수 3개 + `confidence` + `warnings` + `disclaimer`. 두 엔드포인트의 응답
   형식이 점수 축만 다르고 동일함(`tests/test_api.py::
   test_analyze_and_simple_share_the_same_envelope`이 지킴). `SourceInfo`는 삭제됨.
   전체 `SkinReport`가 필요하면 CLI `analyze`.
4. **README에 "데이터 확보 가이드" 섹션 추가** — 지표별 다음 데이터 소스 정리.
   핵심: 얼굴 정면 + 장비 실측이 붙은 공개 데이터셋은 028 외에 사실상 없음. 홍조는
   전문의 CEA 채점(장비 불필요)이 최선, 수분·색소는 타깃 폰 + 장비 동시 측정 필요.

테스트 134개 통과.

### 이전 라운드 (레퍼런스 보정)

AI-Hub 《028. 한국인 피부상태 측정 데이터》(965명 / 정면 2,895장 / ROI 11,580행 /
Corneometer·전문가 등급 실측)로 보정하면서 **정확도 관련 실제 버그 3건**을 찾아 고쳤습니다.

1. 🐛 **gray-world가 색소·홍조 신호를 파괴하고 있었음** (가장 큰 문제).
   얼굴이 프레임을 채우면 장면 평균 = 피부색이라, WB 게인을 적용하는 순간 피부 색도가
   나눠집니다. a*: 15.9 → **0.5**, b*: 26.6 → **-1.2**, ITA는 ±90°로 **포화하고 부호가
   뒤집힘**. 코호트 90명 실측 검증: spearman(-ITA, 전문가 등급)이
   `grayworld` **-0.307**(역상관!) vs `none` **+0.437**.
   → 기본 fallback을 `none`(카메라 AWB 신뢰)으로 변경. `background`(비피부 픽셀만
   gray-world) 옵션도 추가. Fitzpatrick 분포도 [16,3,32,10,28,1] → **[0,0,55,35,0,0]**으로
   정상화(한국인 코호트에서 타입 3~4만 나오는 게 맞음).
2. 🐛 **anchor 스케일이 실제 특징값과 두 자릿수 어긋나 있었음**.
   `glcm_contrast` anchor mean 50.0 vs 실제 0.8, `scaling_index` 0.10 vs 0.0009.
   → 코호트에서 재추정(`calibration_profile.yaml`).
3. 🐛 **텍스처 지표가 얼굴 크기에 종속**. 기기별 eye-span 1140 / 972 / 889px.
   → 특징 추출 전 eye-span 512px로 **다운스케일 전용** 정규화. 부수 효과로
   36MP 분석이 11초 → 3.6초.

그 외: 리포트 `raw_features` 반올림 4→6자리(작은 텍스처 값이 뭉개지던 문제),
`calibrate_image`가 landmark 이후로 이동(비피부 마스크가 필요해서).

4. 🐛 **`hemoglobin` 특징이 구조적으로 항상 0**. FastICA는 입력을 중심화하므로 추출된
   source의 평균은 정의상 0입니다(코호트 전체에서 |값| 최대 1.5e-12). 이게 홍조
   composite에서 **가중치 0.3을 차지하며 신호를 30% 희석**하고 있었습니다.
   → composite weights에서 제거(`config.yaml`에 사유 기재). 재설계는 next steps 참조.

5. 🐛 **색소 composite이 가장 불안정한 특징에 가중치 40%**. `melanin_index`는 기기별
   순위 일치도가 +0.52 / **-0.09** / +0.03으로 부호가 뒤집히는데 가중치 0.40,
   반면 가장 강하고 기기 무관한 `spot_count`(+0.60/+0.73/+0.63)는 composite에 아예
   없었습니다. 절대 색상(멜라닌·ITA)은 카메라를 넘으면 전이되지 않고 형태학 특징은
   전이됩니다. → 형태학 특징만 남기고 가중치를 코호트에서 피팅.
   **실측 일치도 +0.139 → +0.422 (전문가 등급), +0.208 → +0.578 (장비 스팟 개수)**.
   ⚠️ 지표 의미가 "톤 어두움 포함"에서 **"반점 부담"**으로 좁아졌습니다.
6. 🐛 **`composite_raw`가 부호 있는 가중치 합으로 정규화**. 피팅된 가중치 합이 ≤0이면
   조용히 전부 0이 됩니다(수분에서 실제 발생). → `|w|` 합으로 변경. `fit.py`가 들고
   있던 사본에도 같은 버그가 있어서 사본을 없애고 런타임 함수를 직접 쓰게 했습니다.

**검증 (held-out 321명, 동일 이미지 전후 비교)**: 보정 전 점수는 변별력이 거의 없었습니다.
홍조는 321명 중 **215명이 최상위 십분위**(평균 90.3), 수분은 **288명이 두 십분위에 몰림**
(범위 25.4~53.5). 보정 후 세 지표 모두 십분위 균등(각 21~43명), 평균 ≈ 50, 0~100 전 구간 사용.

**지도학습은 둘 다 게이트 탈락**: 색소 r=+0.302/MAE +2.4%, 수분 r=+0.173/MAE +0.9%.
원인은 **기기 종속성** — 기기별로는 잘 되지만(디지털카메라 r=+0.600, MAE +17.4%)
디지털카메라 모델을 폰 사진에 쓰면 MAE가 평균 예측보다 **217% 나빠집니다**.
기기를 모르는 입력에서는 쓸 수 없다는 뜻. 수분은 기기를 고정해도 안 됩니다
(Corneometer는 전기 용량 측정이라 표면 광학에 신호가 거의 없음). 최종 프로파일은
**풀링 + `supervised: {}`**.

### 수분력 라운드 (2026-08-11)

"수분을 무조건 제공해야 한다"는 요구에 맞춰 다시 파고들어 **버그 2건**을 찾았습니다.
이전 라운드에서 "수분은 물리적으로 불가능"이라고 결론 냈던 것은 **부분적으로 틀렸습니다** —
신호는 있었고, 두 버그가 그걸 가리고 있었습니다.

7. 🐛 **`fit_composite_weights`가 `COMPOSITE_TARGET_SIGN`을 무시**. 수분 점수는 *건조함*이
   높을수록 올라가는데 Corneometer는 *수분량*을 잽니다(sign=-1). 원본 타깃에 회귀하니
   **항상 정확히 부호가 뒤집힌 가중치**가 나왔고, 게이트는 그걸 "반상관"이라며 기각했습니다.
   전 기기에서 음수였던 게(pooled -0.208 / 폰 -0.128 / 태블릿 -0.338 / DSLR -0.365) 전부
   이 버그입니다. 색소는 sign=+1이라 우연히 맞아서 드러나지 않았습니다.
   → `sign` 인자 추가. pooled -0.208 → **+0.234**.
8. 🐛 **수분을 전체 ROI로 집계 (코 포함)**. Corneometer는 이마·볼·턱에서 재고 **코는 잰 적이
   없는데** 집계에 들어가 있었습니다. 볼에만 신호가 있습니다(held-out, n=107/셀):
   | ROI | 폰 | 태블릿 | DSLR |
   |---|---|---|---|
   | left_cheek | +0.284 | +0.232 | +0.385 |
   | right_cheek | +0.253 | +0.088 | +0.258 |
   | forehead | +0.051 | +0.187 | +0.165 |
   | chin | +0.072 | -0.019 | +0.005 |
   → `composite.<metric>.rois`로 지표별 집계 ROI 제한 도입(`pipeline._aggregation_rois`).
   `glcm_contrast`/`specular_inv`도 볼에서 무상관~역상관이라 제거.
   특징 4개 → `scaling_index` 0.75 / `wrinkle_density` 0.25.
   (이후 최신 라운드의 부분집합 탐색으로 현재 세트로 교체됨 — 위 참조.)

**수분 실측 일치도 +0.048(얼굴 단위) → +0.241(볼, held-out n=642) → 현재 +0.320**.
여전히 **proxy**이며 `is_estimate=True` 유지.

**시도했지만 채택 안 한 것** (근거 남김):
- **측면 이미지**(`--angles L,R`, 폰 1,930장): 볼이 1.7배 크게 잡혀 실제로 개선되지만
  (+0.053 → +0.128) 정면 재보정(+0.241)에 한참 못 미침. 사용자에게 추가 촬영을 요구할
  가치 없음. 다각도 추출 인프라(`Sample.angle`, `index_dataset(angles=...)`,
  `extract --angles`)는 남겨둠.
- **정규화 해제**(원본 해상도): **더 나빠짐** (+0.053 → +0.003). 폰의 노이즈 리덕션·샤프닝
  아티팩트가 원본에서 피부와 무관한 텍스처로 잡힘. `target_eye_span_px: 512`가 옳다는
  실측 확인. 해상도 가설은 반증됨 — 각도의 이득은 픽셀 수가 아니라 시야 기하 때문.

### 기존 상태

- ✅ **Phase 1 완료** — 전 모듈 구현, 단위 테스트 **134개 통과**(API·보정 툴링 포함).
- ✅ **Phase 2 스캐폴드 완료** — dataset(+더미)/network/train, 더미로 학습 루프 end-to-end 확인.
- ✅ 실제 이미지 경로(MediaPipe **Tasks API**) 동작하도록 `detect_landmarks` 이중 API 지원.
- ✅ **HTTP API 완료** (`skin_metrics/api/`, `api` extra) — `POST /analyze`(이미지 URL) /
  `POST /analyze/simple`(0~10 소비자 점수) / `GET /healthz`. 실제 사진으로 end-to-end 확인.
- ✅ **Docker 완료** — `Dockerfile`(멀티스테이지, `api`/`full`) · `docker-compose.yml` ·
  deny-all `.dockerignore`. 두 타깃 모두 arm64에서 검증:
  `api`(1.72GB) 빌드→기동→`/analyze`, `full`(2.88GB) `train --dummy` 정상 완료.
- ✅ **linux torch는 CPU 전용 인덱스** (`pyproject.toml`의 `[tool.uv.index] pytorch-cpu`).
  macOS 로컬은 기존 PyPI 휠 그대로. 이미지 9.61GB → 2.88GB.
- 🗑 `data/*.jpg`, `report.json`은 사용자 요청으로 **삭제됨**(gitignore 대상이라 복구 불가).
  README/CLAUDE 예시는 `data/face.jpg` 같은 자리표시자 이름으로 바뀜.
- ⏳ 미완/의도적 보류: `config.yaml` 레퍼런스 분포는 **placeholder**(재추정 필요),
  Phase 2는 **실측 라벨 CSV 부재로 절대값 무의미**(더미/ranking만 검증됨).

## 개발 환경 (중요 — 환경 특이사항)

- **Python 3.11**을 uv로 관리 (`.python-version` = 3.11). 시스템 python은 3.9라 직접 쓰지 말 것.
- **uv 위치**: `~/Library/Python/3.9/bin/uv`. 대화형 셸 PATH에 없을 수 있음. 필요 시:
  ```bash
  export PATH="$HOME/Library/Python/3.9/bin:$HOME/.local/bin:$PATH"
  ```
- **venv 활성화 시 uv 없이도** `skin-metrics` / `pytest` 실행 가능 (`source .venv/bin/activate`).
- **의존성 그룹** (`pyproject.toml`):
  - core: numpy/scipy/scikit-image/scikit-learn/opencv-headless/colour-science/pydantic/typer/pyyaml
  - `detection` extra: `mediapipe` (실이미지 얼굴 검출)
  - `dl` extra: `torch`/`timm`/`albumentations`/`pandas`
  - `api` extra: `fastapi`/`uvicorn`/`httpx` (Pillow는 scikit-image 스택에 이미 있음)
  - `dev` extra: `pytest`
  ```bash
  uv sync --extra dev                        # Phase 1 개발/테스트
  uv sync --extra detection --extra dl --extra api --extra dev   # 전체
  ```
  ⚠️ `uv sync`는 **지정한 extra만 남기고 나머지는 제거**합니다. 일부만 sync하면
  `cv2`(mediapipe 스택)가 빠져 테스트가 깨질 수 있으니 전체 sync 줄을 쓰세요.
- **설치된 버전 특이점**:
  - `mediapipe 1.0.0` — 레거시 `mp.solutions.face_mesh` **없음**, **Tasks API만** 존재.
    `detect_landmarks`는 둘 다 지원하지만 이 환경에선 Tasks 경로를 탐. `face_landmarker.task`
    모델(~3.8MB)이 `~/.cache/skin_metrics/`에 필요 → `ensure_face_model()` 또는
    `analyze --download-model`로 받음.
  - `albumentations 2.x` — `ShiftScaleRotate` deprecated → `Affine` 사용 중.

## 자주 쓰는 명령

```bash
uv run pytest -q                                   # 전체 테스트
uv run pytest tests/test_models.py -q              # Phase 2만 (torch 필요)
uv run pytest tests/test_calibrate.py -q           # 보정 툴링 (합성 테이블, 코퍼스 불필요)
uv run skin-metrics analyze data/face.jpg --download-model --output report.json
uv run skin-metrics train --dummy --mode ranking --epochs 1
uv run skin-metrics serve --download-model                # HTTP API (/docs)
uv run pytest tests/test_api.py -q                        # API만 (fastapi 필요)

# 레퍼런스 보정 (코퍼스 필요, 추출 약 27분 / 6워커)
uv run skin-metrics calibrate extract --data-root "028. 한국인 피부상태 측정 데이터" --workers 6
uv run skin-metrics calibrate fit --dry-run               # 검증 수치만
uv run skin-metrics calibrate fit                         # calibration_profile.yaml 기록

./redeploy.sh                                             # down→build→up→헬스 대기
docker build -t skin-metrics-api:0.1.0 .                  # 기본 = api 타깃 (torch 없음)
docker build --target full -t skin-metrics-api:0.1.0-full .   # + Phase 2
docker compose up --build                                 # 127.0.0.1:8000
```

## 아키텍처 지도 (수정 시 진입점)

```
pipeline.analyze(img, ref_bbox, ccm, landmarks, model_path, config)
└─ pipeline.extract_raw(...) → RawExtraction   # ← 이미지→원시특징 (반올림 없음)
   ├─ detection.face.detect_landmarks (레거시 or Tasks API) → (468,2)
   │     ※ 보정보다 먼저: WB가 비피부 마스크를 필요로 함
   ├─ detection.face.face_mask → ~mask = background
   ├─ calibration.color.calibrate_image(fallback=none|background|grayworld)
   │     linearize_srgb / white_balance_* / estimate_ccm+apply_ccm
   ├─ pipeline._normalize_face_scale  # eye-span 512px로 다운스케일 (선형 광에서)
   ├─ calibration.color.rgb_to_lab(D65)
   ├─ detection.face.extract_rois → {name: ROIResult|None}  (5 ROI, 0.6 게이트)
   └─ features.{pigmentation,erythema,hydration_proxy}   (ROI valid_mask 내부만)
└─ pipeline._score_metric (지표별)
   ├─ config["supervised"][metric] 있으면 → scoring.normalize.predict_instrument
   │     (학습된 ROI에만 적용, 유효픽셀 가중평균) → driver = score_sign × 예측값
   └─ 없으면 → scoring.normalize.composite_raw (anchor z-score 가중합)
   → scoring.normalize.score_from_raw (Fitzpatrick별 경험적 분위수 격자 → 0~100)
   → scoring.schema.SkinReport (pydantic, 의료 고지 포함)
```

오프라인 보정 (런타임에서 import되지 않음):
```
calibrate.aihub.index_dataset  → [Sample]  (이미지 + facepart별 실측 라벨/bbox 조인)
calibrate.extract.run_extraction → features_roi.csv / features_face.csv  (멀티프로세스·재개가능)
calibrate.fit.fit_calibration  → anchors / supervised(릿지) / reference(분위수 격자)
   └─ accept_model 게이트 통과 실패 시 supervised에서 제외 (사유는 validation에 기록)
calibrate.fit.write_profile    → skin_metrics/calibration_profile.yaml (생성 파일)
config.load_config             → config.yaml + calibration_profile.yaml 병합
```

HTTP API (`api` extra):
```
api.app.create_app(settings) → FastAPI          # 모듈 최상단 app = create_app() (uvicorn 타깃)
├─ lifespan: load_config / ensure|resolve_face_model / anyio.Semaphore / 공유 httpx.AsyncClient
├─ GET  /healthz  → face_model_available · detection_available
├─ POST /analyze         ┐ 공유 _run_report: api.fetch.fetch_image(URL 검증·스트리밍·디코딩)
└─ POST /analyze/simple  ┘ → anyio.to_thread.run_sync(pipeline.analyze)  # CPU 바운드
   두 응답 모두 평평한 동일 envelope (점수 + confidence + warnings + disclaimer):
   /analyze = 0~100 AnalyzeResponse.from_report, /simple = 0~10 SimpleAnalyzeResponse.from_report
   전체 SkinReport는 CLI 전용
api.settings.ApiSettings.from_env()  # SKIN_METRICS_API_* (한도·타임아웃·동시성)
```

Phase 2: `models.dataset(SkinDataset/DummyLabelGenerator)` → `models.network.SkinNet`
(EfficientNet + PhysicsMLP concat, 3 heads, uncertainty weighting, GRL 도메인 적대)
→ `models.train.run_training` (Huber/ranking, MAE·Pearson·Spearman, 타입별 리포트).

## 코드 규약 (지킬 것)

- **타입 힌트 + numpy 스타일 docstring** 전 함수.
- **수치 방어**: 0 division·음수/0 log는 반드시 epsilon clip (`_EPS` 패턴 참고).
- **색공간 규약**: 내부 연산은 **선형 RGB**. `rgb_to_lab(assume_linear=True)`가 기본.
  이미지는 `(H,W,3)` float, `linear`/`sRGB` 구분을 주석·인자로 명확히.
- **무거운 의존성은 지연 import**: 함수 내부에서 `import torch/mediapipe/timm/...`.
  모듈 최상단에서 import하면 Phase 1 테스트가 깨짐.
- **수분력은 항상 proxy**: 함수명·docstring·`is_estimate=True`로 표기 유지.
- **`hydration` 리포트 방향은 "높을수록 촉촉"**, 내부 계산은 "높을수록 건조".
  `scoring.report_inverted`가 `pipeline._score_metric._finish`에서 마지막에 뒤집습니다.
  더 앞에서 뒤집으면 프로파일의 검증 수치·`COMPOSITE_TARGET_SIGN`과 어긋납니다.
- **의료기기 아님 고지**는 `SkinReport.disclaimer`(자동)·README·CLI에 유지.
- **테스트는 합성 데이터**: 실제 얼굴 사진/네트워크 없이 도는 것을 유지
  (`tests/conftest.py`의 `synthetic_image`/`synthetic_landmarks`,
  Phase 2는 `pretrained=False`·더미).

## 주의할 함정 (Gotchas)

- **인물 사진에 whole-image gray-world 금지**. 장면 평균이 곧 피부색이라 a*/b*가 0으로
  붕괴하고 ITA가 ±90°에서 포화·부호 반전합니다. `calibration.fallback`을 `grayworld`로
  되돌리지 마세요(측정 근거는 README "인물 사진에 gray-world를 쓰면 안 되는 이유").
- **얼굴 크기 정규화는 다운스케일 전용**. 업샘플링하면 텍스처가 매끄러워져 거짓으로
  촉촉하게 읽힙니다. 500×500 합성 테스트 이미지는 eye-span이 140px이라 정규화가
  no-op이 되고, 그래서 기존 테스트 지오메트리가 그대로 유지됩니다.
- **`target_eye_span_px: 512`를 올리지 마세요**. "다운스케일이 정보를 버린다"는 직관은
  실측으로 반증됐습니다. 정규화를 끄고 폰 정면을 원본 해상도로 돌리면 수분 일치도가
  +0.053 → **+0.003**으로 무너집니다(전 특징 악화). 폰의 노이즈 리덕션·샤프닝
  아티팩트가 원본 해상도에서 피부와 무관한 텍스처로 읽히고, 다운스케일이 그걸 뭉갭니다.
- **집계 ROI는 지표마다 다릅니다** (`composite.<metric>.rois`). 수분은 **볼만** 씁니다 —
  Corneometer가 코에서는 측정된 적이 없고 이마·턱은 신호가 거의 없습니다. 전체 ROI로
  되돌리면 일치도가 절반으로 떨어집니다. 지표를 추가하면 그 지표의 실측 부위를 확인하고
  `rois`를 명시하세요.
- **`COMPOSITE_TARGET_SIGN`은 피팅에도 전달돼야 합니다**. `fit_composite_weights(sign=...)`를
  빠뜨리면 sign=-1 지표(수분)에서 **부호가 뒤집힌 가중치**가 나오고, 게이트가 그걸
  "반상관"으로 오해해 기각합니다. 조용히 실패하므로 `tests/test_calibrate.py::
  test_fit_composite_weights_honours_an_inverted_target`가 지킵니다.
- **`calibration_profile.yaml`은 생성 파일**. 손으로 고치지 말고
  `skin-metrics calibrate fit`으로 재생성. 사람이 쓰는 설정·근거는 `config.yaml`에만.
  `config.yaml`의 composite 가중치는 **피팅이 채택 안 됐을 때의 fallback**이고,
  거기 적힌 **특징 목록이 피팅 대상 집합**을 정합니다(anchor도 이 목록으로 만들어짐).
- **composite 가중치는 음수가 될 수 있습니다**(suppressor). 정규화는 반드시 `|w|` 합으로.
  부호 있는 합으로 나누면 합이 0 근처일 때 폭발하거나 조용히 0이 됩니다.
  현재 수분 세트의 `glcm_contrast` -0.30 / `lbp_uniformity` -0.25가 실제 예 — 단독으론
  무상관이지만 suppressor로 +0.075를 벌어줍니다. "음수라서" 지우면 안 됩니다.
- **`/analyze/simple`의 `dryness`는 리포트 방향(높을수록 촉촉) 위에서 한 번 더 뒤집습니다**
  (`(100-hydration)/10`). `scoring.report_inverted`를 건드리면
  `SimpleAnalyzeResponse.from_report`와 `tests/test_api.py`의 simple 테스트도 같이 확인.
  `skin_tone`은 ITA 절대 색상 기반이라 simple 응답에서 유일하게 기기·조명 민감.
- **절대 색상 특징은 기기 간 전이 안 됨**. `melanin_index`/`ita`를 composite에 다시
  넣으려면 그레이카드/컬러체커 보정(`--reference-bbox`)이 전제되어야 합니다.
- **`calibrate fit`은 `load_config(use_profile=False)`로 읽어야** 합니다. 안 그러면
  직전 프로파일의 anchor가 새 anchor 추정에 되먹임됩니다.
- **`extract.FEATURE_COLUMNS`는 하드코딩**(CSV 열 순서 안정성). 파이프라인 특징을
  추가/변경하면 `tests/test_calibrate.py::test_feature_columns_match_the_pipeline`가
  잡아줍니다.
- **`imap_unordered` 결과는 입력 순서와 무관**. `ExtractResult`가 자기 key를 들고
  다니는 이유이며, `zip(pending, results)`로 짝지으면 라벨이 뒤섞입니다.
- **워커 스레드 과구독 주의**. MediaPipe는 워커마다 13개 스레드를 띄웁니다. 12코어에서
  10워커로 돌렸다가 load average 140까지 올라가 멈췄습니다.
  `extract._limit_threads()`가 BLAS/OpenMP를 1스레드로 고정하고, 워커는 6개 권장.
- **`face_mask`의 마진은 형태학 dilate가 아니라 헐 폴리곤 확대**. 얼굴 폭의 35%면
  841×841 구조 요소가 되어 이미지당 수 초가 걸립니다.
- **synthetic_landmarks의 공유 인덱스**: 이마 ROI와 눈썹 exclusion이 66/107을 공유.
  conftest는 exclusion을 먼저, ROI를 나중에 배치해 ROI가 이기게 함. 랜드마크 지오메트리를
  바꾸면 ROI valid_ratio 게이트(0.6)를 다시 확인.
- **grayworld는 이미지 전체 평균** → 강한 색 배경에 영향. 단, 지표는 ROI 내부만 쓰므로
  배경 누끼는 불필요. 정확도 레버는 `--reference-bbox`(그레이카드)/CCM. (자세한 논의는
  README "배경 영향" 참고.)
- **Tasks API는 478점 반환** (468 mesh + iris). `[:468]`만 사용 중.
- **ranking 모드의 MAE는 무의미** (절대 스케일 미보정). Pearson/Spearman로 판단.
- **API의 `from __future__ import annotations` + 지역 import 조합 금지**: FastAPI는 문자열
  애노테이션을 **모듈 전역**에서 해석하므로 `Request`를 함수 안에서 import하면 경로 파라미터로
  오인해 전 요청이 422가 됩니다. `api/app.py`의 fastapi/anyio/httpx는 최상단 import 유지
  (지연 import 규칙의 예외 — 이 모듈 자체가 `api` extra 전용).
- **API 테스트는 실제 루프백 HTTP 서버**를 씁니다(외부 네트워크 없음). mediapipe 없이 돌도록
  `skin_metrics.pipeline.detect_landmarks`를 monkeypatch → 랜드마크 지오메트리를 바꾸면
  `tests/test_api.py`도 같이 확인.
- **SSRF 가드 해제 플래그**(`--allow-private-hosts` / `SKIN_METRICS_API_ALLOW_PRIVATE_HOSTS`)는
  개발·테스트 전용. 배포 설정에 새지 않게 할 것.
- **Docker: mediapipe가 비-headless opencv를 끌고 옴** → 런타임 이미지에 `libgl1`
  `libglib2.0-0` `libxcb1` (+ sounddevice용 `libportaudio2`, `libgomp1`) 필요.
  의존성 바뀌면 `ldd .venv/.../cv2/cv2.abi3.so | grep "not found"`로 재확인.
- **Docker: `ghcr.io` 익명 pull이 이 환경에서 막힘** → uv를 PyPI에서 설치(Docker Hub만 사용).
- **Docker: 모델 다운로드는 최종 스테이지에서** 실행해야 함. `detection/face.py`가 최상단에서
  `cv2`를 import하므로 OS 라이브러리가 없는 builder 스테이지에서는 `ensure_face_model`조차
  실패. 그래서 `api`/`full` 각각에 동일한 RUN이 한 줄씩 있음(의도된 중복).
- **`.dockerignore`는 deny-all + allow-list**. 새 파일이 자동으로 이미지에 들어가지 않으므로,
  런타임에 필요한 파일을 추가했다면 `!` 규칙을 직접 넣어야 함.
- **linux torch는 반드시 CPU 인덱스**: PyPI 기본 휠은 nvidia-* CUDA를 끌고 와
  `import torch`가 SIGILL(exit 132)로 죽음(`_preload_cuda_deps`). `pyproject.toml`의
  `[[tool.uv.index]] pytorch-cpu` + `[tool.uv.sources]`(`sys_platform == 'linux'`)로 해결.
- **`tool.uv.sources`는 직접 의존성에만 적용**. 그래서 `torchvision`을 `dl` extra에 명시함.
  빼면 torch만 `+cpu`가 되어 `operator torchvision::nms does not exist` 발생.
- **`dl` extra 버전을 건드리면** `uv lock` 후 **로컬(macOS)과 컨테이너(linux) 양쪽** 확인:
  두 플랫폼이 서로 다른 인덱스에서 해석됨.

## 이어서 하면 좋은 작업 (Next steps)

1. **홍조 실측 라벨 확보**: 이 코호트에는 Mexameter 홍반값이 없어 홍조만 검증되지 않은
   상태입니다. 현재는 코호트 백분위로만 정규화. Mexameter/Antera 데이터가 생기면
   `calibrate/fit.py`의 `specs` 튜플에 한 줄 추가하면 그대로 붙습니다.
2. **색소 모델 개선**: 현재 릿지(선형)로 held-out r≈0.54. 특징을 늘리거나
   (ROI별 상수항, 나이 대용 특징) 순서형 회귀로 바꾸면 더 오를 여지가 있습니다.
   ※ 수분은 +0.320까지 왔습니다(부분집합 탐색으로 기존 추출 특징은 소진).
   다음 레버는 **새 볼 전용 텍스처 특징**(재추출 필요, 27분) 아니면
   **TEWL/Corneometer 자체 측정**입니다.
3. **Phase 2 실학습**: 라벨 CSV가 이제 실제로 존재합니다.
   `calibrate/aihub.py`의 `iter_roi_rows`로 `SkinDataset.from_csv` 형식을 만들면
   `train --data labels.csv --mode regression` 가능. 다만 38GB 이미지 학습이라
   이 맥에서는 수 시간~하루 단위.
4. **리포트 시각화**: `SkinReport` → HTML/차트(얼굴 오버레이, ROI별 점수, 시계열).
5. **헤모글로빈 특징 재설계** (제거된 상태). ROI별로 FastICA를 돌리면 source가 그 ROI
   안에서 zero-mean이라 절대 수준을 못 냅니다. 두 방향:
   (a) ICA를 **얼굴당 1회** 피팅하고 ROI별 평균을 사후에 취하기,
   (b) ICA를 버리고 광학 밀도를 **측정된 흡광 방향**에 직접 투영하기
   (`erythema._HEMOGLOBIN_DIR/_MELANIN_DIR`를 카메라별 실측 스펙트럼으로 교체).
   FastICA 수렴 실패(`ConvergenceWarning`)도 흔합니다. 바꾸면 특징이 달라지므로
   `calibrate extract --no-resume`으로 재추출 필요.
6. **API 처리량**: 얼굴 크기 정규화로 36MP 기준 66초 → 훨씬 짧아졌습니다. 재측정 후
   작업 큐(`202 → /jobs/{id}`) 필요 여부 재판단. 인증·레이트리밋은 아직 없음.
7. **기기별 프로파일**: `calibrate fit --device phone`으로 기기별 피팅이 가능합니다.
   현재 기본 프로파일은 3기기 pooled. 서비스가 폰 전용이면 폰 전용 프로파일이 나을 수
   있으니 비교해볼 것.

## 결정 로그 (왜 이렇게 했나)

- **Phase 1 우선, Phase 2는 스캐폴드**: 사용자가 실측 라벨 확보 계획이 없다고 확인.
  라벨 없는 딥러닝은 무의미한 절대값을 내므로, 실사용 엔진은 Phase 1로 두고 Phase 2는
  라벨 확보 시 즉시 붙일 수 있는 구조만 완성.
- **의존성 extra 분리 + 지연 import**: 무거운 mediapipe/torch 없이도 물리 로직·테스트가
  빠르게 돌도록.
- **Fitzpatrick 타입별 정규화**: 어두운 피부에서의 지표 편향을 줄이기 위해 레퍼런스 분포를
  타입별로 분리.
- **화이트밸런스 기본값 `none`**: 무보정이 "아무것도 안 함"이라 나빠 보이지만, 인물 사진에서
  gray-world는 측정 대상인 색 신호 자체를 파괴합니다. 코호트 실측으로 세 모드를 비교해
  결정했습니다(README 표). 카메라 AWB는 이미 sRGB JPEG에 반영된 최선의 추정치입니다.
- **설정 파일 2분할**: `config.yaml`(사람·주석) / `calibration_profile.yaml`(기계·생성).
  YAML을 기계가 덮어쓰면 근거 주석이 날아가고, 무엇이 피팅된 값이고 무엇이 손으로 정한
  값인지 구분이 사라집니다.
- **지도학습 모델 채택 게이트**: 평균 예측과 별 차이 없는 모델이 리포트에 구체적인 실측값을
  찍으면, 개인 정보량이 거의 없는데도 있는 것처럼 보입니다. 수분 모델이 실제로 여기서
  탈락했고, 그게 의도된 동작입니다.
- **경험적 분위수 격자 > 정규 CDF**: 스팟 개수·예측 등급 분포가 치우쳐 있어 가우시안 가정이
  꼬리를 잘못 배치합니다. 격자가 있으면 격자를, 없으면 기존 CDF로 폴백.
