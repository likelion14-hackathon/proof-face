# CLAUDE.md

이 파일은 이 저장소에서 작업하는 Claude Code(및 새 세션/기여자)를 위한 안내서입니다.
사용자용 문서는 [README.md](README.md)를, 여기서는 **개발 환경·규약·현재 상태·이어갈 지점**을
다룹니다.

## 프로젝트 한 줄 요약

얼굴 이미지 1장에서 **색소침착 / 홍조 / 수분력(proxy)** 를 0~100으로 산출.
**Phase 1(물리 기반, 학습 불필요)** 이 실제 "이미지→지수" 엔진이고,
**Phase 2(딥러닝)** 는 실측 라벨이 생기면 붙일 수 있는 **완전히 도는 스캐폴드**.

## 현재 상태 (2026-07-31 기준)

- ✅ **Phase 1 완료** — 전 모듈 구현, 단위 테스트 **56개 통과**.
- ✅ **Phase 2 스캐폴드 완료** — dataset(+더미)/network/train, 더미로 학습 루프 end-to-end 확인.
- ✅ 실제 이미지 경로(MediaPipe **Tasks API**) 동작하도록 `detect_landmarks` 이중 API 지원.
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
  - `dev` extra: `pytest`
  ```bash
  uv sync --extra dev                        # Phase 1 개발/테스트
  uv sync --extra detection --extra dl --extra dev   # 전체
  ```
- **설치된 버전 특이점**:
  - `mediapipe 1.0.0` — 레거시 `mp.solutions.face_mesh` **없음**, **Tasks API만** 존재.
    `detect_landmarks`는 둘 다 지원하지만 이 환경에선 Tasks 경로를 탐. `face_landmarker.task`
    모델(~3.8MB)이 `~/.cache/skin_metrics/`에 필요 → `ensure_face_model()` 또는
    `analyze --download-model`로 받음.
  - `albumentations 2.x` — `ShiftScaleRotate` deprecated → `Affine` 사용 중.

## 자주 쓰는 명령

```bash
uv run pytest -q                                   # 전체 테스트 (56)
uv run pytest tests/test_models.py -q              # Phase 2만 (torch 필요)
uv run skin-metrics analyze data/test2.jpg --download-model --output report.json
uv run skin-metrics train --dummy --mode ranking --epochs 1
```

## 아키텍처 지도 (수정 시 진입점)

```
pipeline.analyze(img, ref_bbox, ccm, landmarks, model_path, config)  # ← 전체 오케스트레이션
├─ calibration.color.calibrate_image → CalibrationResult(image, status, success)
│     linearize_srgb / white_balance_* / estimate_ccm+apply_ccm / rgb_to_lab(D65)
├─ detection.face.detect_landmarks (레거시 or Tasks API) → (468,2)
├─ detection.face.extract_rois → {name: ROIResult|None}  (5 ROI, 아티팩트 마스킹, 0.6 게이트)
├─ features.{pigmentation,erythema,hydration_proxy}      (ROI valid_mask 내부만 연산)
└─ scoring.normalize.score_metric (composite_raw → score_from_raw, Fitzpatrick별)
      → scoring.schema.SkinReport (pydantic, 의료 고지 포함)
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
- **의료기기 아님 고지**는 `SkinReport.disclaimer`(자동)·README·CLI에 유지.
- **테스트는 합성 데이터**: 실제 얼굴 사진/네트워크 없이 도는 것을 유지
  (`tests/conftest.py`의 `synthetic_image`/`synthetic_landmarks`,
  Phase 2는 `pretrained=False`·더미).

## 주의할 함정 (Gotchas)

- **synthetic_landmarks의 공유 인덱스**: 이마 ROI와 눈썹 exclusion이 66/107을 공유.
  conftest는 exclusion을 먼저, ROI를 나중에 배치해 ROI가 이기게 함. 랜드마크 지오메트리를
  바꾸면 ROI valid_ratio 게이트(0.6)를 다시 확인.
- **grayworld는 이미지 전체 평균** → 강한 색 배경에 영향. 단, 지표는 ROI 내부만 쓰므로
  배경 누끼는 불필요. 정확도 레버는 `--reference-bbox`(그레이카드)/CCM. (자세한 논의는
  README "배경 영향" 참고.)
- **Tasks API는 478점 반환** (468 mesh + iris). `[:468]`만 사용 중.
- **ranking 모드의 MAE는 무의미** (절대 스케일 미보정). Pearson/Spearman로 판단.

## 이어서 하면 좋은 작업 (Next steps)

1. **레퍼런스 재추정 스크립트**: 실촬영 코호트에서 `config.yaml`의 `reference.*`/`composite.*`
   anchor(mean/std)를 피팅하는 CLI/노트북. (현재 placeholder + `TODO` 주석)
2. **실측 라벨 연동**: Corneometer/Mexameter CSV → `SkinDataset.from_csv`로 Phase 2 실학습.
   `train --data labels.csv --mode regression`.
3. **리포트 시각화**: `SkinReport` → HTML/차트(얼굴 오버레이, ROI별 점수, 시계열).
4. **Tsumura 기준 벡터 교체**: 카메라별 측정 흡광 스펙트럼으로
   `erythema._HEMOGLOBIN_DIR/_MELANIN_DIR` 대체.
5. **실제 얼굴 사진 end-to-end 검증**: `data/`에 사진 두고 `analyze --download-model` 실행,
   ROI 마스킹·Fitzpatrick 추정 품질 육안 확인.

## 결정 로그 (왜 이렇게 했나)

- **Phase 1 우선, Phase 2는 스캐폴드**: 사용자가 실측 라벨 확보 계획이 없다고 확인.
  라벨 없는 딥러닝은 무의미한 절대값을 내므로, 실사용 엔진은 Phase 1로 두고 Phase 2는
  라벨 확보 시 즉시 붙일 수 있는 구조만 완성.
- **의존성 extra 분리 + 지연 import**: 무거운 mediapipe/torch 없이도 물리 로직·테스트가
  빠르게 돌도록.
- **Fitzpatrick 타입별 정규화**: 어두운 피부에서의 지표 편향을 줄이기 위해 레퍼런스 분포를
  타입별로 분리.
