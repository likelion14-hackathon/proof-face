"""Tests for the HTTP API (skin_metrics.api).

Everything runs against a throwaway loopback HTTP server, so no external network
is touched. MediaPipe is never needed: ``skin_metrics.pipeline.detect_landmarks``
is monkeypatched with the synthetic landmarks from ``conftest``.
"""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from skin_metrics.api.app import create_app  # noqa: E402
from skin_metrics.api.fetch import ImageFetchError, decode_image, validate_url  # noqa: E402
from skin_metrics.api.schemas import AnalyzeRequest, DiaryAnalyzeResponse  # noqa: E402
from skin_metrics.api.settings import ApiSettings  # noqa: E402
from skin_metrics.scoring.schema import MetricScore, SkinReport  # noqa: E402


def _png_bytes(image: np.ndarray) -> bytes:
    """Encode an RGB uint8 array as PNG bytes."""
    from PIL import Image

    buf = BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def image_server(synthetic_image):
    """A loopback HTTP server exposing image / error / redirect routes."""
    png = _png_bytes(synthetic_image)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # silence stderr noise
            pass

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            if self.path == "/face.png":
                self._send(200, png, "image/png")
            elif self.path == "/notimage.txt":
                self._send(200, b"definitely not an image", "text/plain")
            elif self.path == "/empty.png":
                self._send(200, b"", "image/png")
            elif self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "/face.png")
                self.send_header("Content-Length", "0")
                self.end_headers()
            elif self.path == "/loop":
                self.send_response(302)
                self.send_header("Location", "/loop")
                self.send_header("Content-Length", "0")
                self.end_headers()
            else:
                self._send(404, b"nope", "text/plain")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def client(monkeypatch, synthetic_landmarks):
    """TestClient with loopback URLs allowed and landmark detection stubbed."""
    monkeypatch.setattr(
        "skin_metrics.pipeline.detect_landmarks",
        lambda img, model_path=None: synthetic_landmarks,
    )
    settings = ApiSettings(allow_private_hosts=True, max_concurrency=1)
    with TestClient(create_app(settings)) as test_client:
        yield test_client


# --------------------------------------------------------------------------
# URL validation (SSRF guard)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/a.png",
        "http://localhost/a.png",
        "http://169.254.169.254/latest/meta-data",  # cloud metadata endpoint
        "http://10.0.0.5/a.png",
        "http://192.168.1.1/a.png",
        "http://[::1]/a.png",
    ],
)
def test_validate_url_blocks_non_public_hosts(url):
    with pytest.raises(ImageFetchError) as exc:
        validate_url(url)
    assert exc.value.code == "blocked_host"
    assert exc.value.status_code == 403


def test_validate_url_blocks_non_http_scheme():
    with pytest.raises(ImageFetchError) as exc:
        validate_url("file:///etc/passwd")
    assert exc.value.code == "invalid_scheme"


def test_validate_url_allows_private_when_opted_in():
    validate_url("http://127.0.0.1:8000/a.png", allow_private_hosts=True)


def test_validate_url_allows_public_literal():
    validate_url("https://93.184.216.34/a.png")


# --------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------


def test_decode_image_roundtrip(synthetic_image):
    arr = decode_image(_png_bytes(synthetic_image), max_pixels=10_000_000)
    assert arr.shape == synthetic_image.shape
    assert arr.dtype == np.uint8
    np.testing.assert_array_equal(arr, synthetic_image)


def test_decode_image_rejects_garbage():
    with pytest.raises(ImageFetchError) as exc:
        decode_image(b"not an image at all", max_pixels=10_000_000)
    assert exc.value.code == "decode_error"


def test_decode_image_downscales_to_the_analysis_budget(synthetic_image):
    """Over-budget images are resized to fit, not rejected."""
    budget = 40_000  # 500x500 = 250k px, so this forces a downscale
    arr = decode_image(
        _png_bytes(synthetic_image), max_pixels=10_000_000, analysis_max_pixels=budget
    )
    assert arr.shape[0] * arr.shape[1] <= budget
    # Aspect ratio preserved (square in, square out).
    assert arr.shape[0] == arr.shape[1]
    assert arr.dtype == np.uint8


def test_decode_image_leaves_under_budget_images_untouched(synthetic_image):
    """No upscaling: a small image must come back at its native size."""
    arr = decode_image(
        _png_bytes(synthetic_image), max_pixels=10_000_000, analysis_max_pixels=10_000_000
    )
    np.testing.assert_array_equal(arr, synthetic_image)


def test_analyze_downscales_instead_of_rejecting_a_big_image(
    monkeypatch, synthetic_landmarks, image_server
):
    """A photo over the analysis budget still scores, rather than returning 413."""
    monkeypatch.setattr(
        "skin_metrics.pipeline.detect_landmarks",
        lambda img, model_path=None: synthetic_landmarks,
    )
    settings = ApiSettings(allow_private_hosts=True, analysis_max_pixels=40_000)
    with TestClient(create_app(settings)) as budgeted:
        result = _analyze_result(
            budgeted, "/analyze", {"image_url": f"{image_server}/face.png"}
        )
    assert 0.0 <= result["pigmentation"] <= 100.0


def test_decode_image_rejects_too_many_pixels(synthetic_image):
    with pytest.raises(ImageFetchError) as exc:
        decode_image(_png_bytes(synthetic_image), max_pixels=100)
    assert exc.value.code == "image_too_large"
    assert exc.value.status_code == 413


# --------------------------------------------------------------------------
# Endpoints (asynchronous flow: 202 -> background analysis -> result document)
# --------------------------------------------------------------------------


def _submit(client, path, body):
    """POST a submission and return the accepted body after basic checks."""
    resp = client.post(path, json=body)
    assert resp.status_code == 202, resp.text
    accepted = resp.json()
    kind = path.rsplit("/", 1)[-1] if path != "/analyze" else "analyze"
    assert accepted["redis_key"] == f"{accepted['request_id']}:{kind}"
    assert accepted["status"] == "processing"
    return accepted


def _wait_result(client, redis_key, timeout=30.0):
    """Poll GET /results/{key} until the document leaves 'processing'."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/results/{redis_key}")
        assert resp.status_code == 200, resp.text
        doc = resp.json()
        if doc["status"] != "processing":
            return doc
        time.sleep(0.05)
    raise AssertionError(f"result {redis_key} still processing after {timeout}s")


def _analyze_result(client, path, body):
    """Submit and wait; assert success; return the stored result payload."""
    accepted = _submit(client, path, body)
    doc = _wait_result(client, accepted["redis_key"])
    assert doc["status"] == "done", doc
    return doc["result"]


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["result_store"] == "memory"  # tests run without SKIN_METRICS_REDIS_URL
    assert "의료기기가 아니" in body["disclaimer"]


def test_openapi_example_is_a_request_that_would_succeed(client):
    """The docs' "Try it out" body must be valid.

    Pydantic would otherwise synthesise ``reference_bbox: [0, 0, 0, 0]`` from the
    type, and that body is rejected by the validator (width/height must be > 0).
    """
    schema = client.get("/openapi.json").json()["components"]["schemas"]["AnalyzeRequest"]
    example = schema["examples"][0]

    assert "reference_bbox" not in example, "bbox is optional; keep it out of the example"
    assert AnalyzeRequest.model_validate(example).reference_bbox is None


def test_analyze_stores_flat_scores(client, image_server):
    result = _analyze_result(client, "/analyze", {"image_url": f"{image_server}/face.png"})
    for metric in ("pigmentation", "erythema", "hydration"):
        assert 0.0 <= result[metric] <= 100.0
        assert 0.0 <= result["confidence"][metric] <= 1.0
    assert result["disclaimer"]
    assert result["elapsed_ms"] >= 0.0


def test_analyze_and_diary_share_the_same_envelope(client, image_server):
    """Both result documents must share the flat shape, only the scores differ."""
    url = {"image_url": f"{image_server}/face.png"}
    full = _analyze_result(client, "/analyze", url)
    diary = _analyze_result(client, "/analyze/diary", url)

    full_scores = {"pigmentation", "erythema", "hydration"}
    diary_scores = {"skin_tone", "dryness", "redness"}
    assert set(full) - full_scores == set(diary) - diary_scores
    assert set(full["confidence"]) == full_scores
    assert set(diary["confidence"]) == diary_scores


def test_result_document_carries_request_metadata(client, image_server):
    accepted = _submit(client, "/analyze", {"image_url": f"{image_server}/face.png"})
    doc = _wait_result(client, accepted["redis_key"])
    assert doc["request_id"] == accepted["request_id"]
    assert doc["kind"] == "analyze"
    assert doc["submitted_at"] and doc["completed_at"]


def test_unknown_result_key_is_404(client):
    resp = client.get("/results/doesnotexist:analyze")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "result_not_found"


def _report_stub(ita=None, hydration=70.0, erythema=35.0) -> SkinReport:
    """Minimal SkinReport for exercising the diary-score mapping."""
    pig_features = {} if ita is None else {"ita": ita}
    return SkinReport(
        pigmentation=MetricScore(score=50.0, confidence=0.8, raw_features=pig_features),
        erythema=MetricScore(score=erythema, confidence=0.7),
        hydration=MetricScore(score=hydration, confidence=0.6, is_estimate=True),
        calibration_status="none",
        fitzpatrick_estimate=3,
    )


def test_diary_mapping_orients_and_scales_each_score():
    diary = DiaryAnalyzeResponse.from_report(
        _report_stub(ita=34.5, hydration=70.0, erythema=35.0), elapsed_ms=1.0
    )
    # ITA 34.5 sits (34.5+30)/85 of the way from dark to very light.
    assert diary.skin_tone == pytest.approx(round(10.0 * 64.5 / 85.0, 1))
    # hydration 70 (moist) -> dryness 3.0; erythema 35 -> redness 3.5.
    assert diary.dryness == pytest.approx(3.0)
    assert diary.redness == pytest.approx(3.5)
    assert diary.confidence == {"skin_tone": 0.8, "dryness": 0.6, "redness": 0.7}


@pytest.mark.parametrize("ita,expected", [(55.0, 10.0), (-30.0, 0.0), (90.0, 10.0), (-80.0, 0.0)])
def test_diary_mapping_clips_skin_tone_to_0_10(ita, expected):
    diary = DiaryAnalyzeResponse.from_report(_report_stub(ita=ita), elapsed_ms=1.0)
    assert diary.skin_tone == pytest.approx(expected)


def test_diary_mapping_survives_a_missing_ita():
    """No `ita` in raw_features -> Fitzpatrick bucket centre, not a crash."""
    diary = DiaryAnalyzeResponse.from_report(_report_stub(ita=None), elapsed_ms=1.0)
    assert 0.0 <= diary.skin_tone <= 10.0


def test_analyze_diary_stores_scores(client, image_server):
    result = _analyze_result(client, "/analyze/diary", {"image_url": f"{image_server}/face.png"})
    for field in ("skin_tone", "dryness", "redness"):
        assert 0.0 <= result[field] <= 10.0
        assert 0.0 <= result["confidence"][field] <= 1.0
    assert result["disclaimer"]


def test_diary_is_consistent_with_the_full_scores(client, image_server):
    url = {"image_url": f"{image_server}/face.png"}
    full = _analyze_result(client, "/analyze", url)
    diary = _analyze_result(client, "/analyze/diary", url)
    assert diary["dryness"] == pytest.approx(
        round((100.0 - full["hydration"]) / 10.0, 1), abs=0.05
    )
    assert diary["redness"] == pytest.approx(round(full["erythema"] / 10.0, 1), abs=0.05)


def test_analyze_with_reference_bbox(client, image_server):
    result = _analyze_result(
        client,
        "/analyze",
        {"image_url": f"{image_server}/face.png", "reference_bbox": [5, 5, 20, 20]},
    )
    assert 0.0 <= result["pigmentation"] <= 100.0


def test_analyze_follows_redirect(client, image_server):
    result = _analyze_result(client, "/analyze", {"image_url": f"{image_server}/redirect"})
    assert 0.0 <= result["pigmentation"] <= 100.0


# --- failures the consumer discovers through the stored document ----------


def _failed_doc(client, path, body):
    accepted = _submit(client, path, body)
    doc = _wait_result(client, accepted["redis_key"])
    assert doc["status"] == "failed", doc
    return doc["error"]


def test_redirect_loop_fails_the_stored_result(client, image_server):
    error = _failed_doc(client, "/analyze", {"image_url": f"{image_server}/loop"})
    assert error["code"] == "too_many_redirects"


def test_non_image_fails_the_stored_result(client, image_server):
    error = _failed_doc(client, "/analyze", {"image_url": f"{image_server}/notimage.txt"})
    assert error["code"] == "decode_error"


def test_empty_body_fails_the_stored_result(client, image_server):
    error = _failed_doc(client, "/analyze", {"image_url": f"{image_server}/empty.png"})
    assert error["code"] == "empty_body"


def test_upstream_404_fails_the_stored_result(client, image_server):
    error = _failed_doc(client, "/analyze", {"image_url": f"{image_server}/missing.png"})
    assert error["code"] == "upstream_error"


def test_byte_limit_fails_the_stored_result(monkeypatch, synthetic_landmarks, image_server):
    monkeypatch.setattr(
        "skin_metrics.pipeline.detect_landmarks",
        lambda img, model_path=None: synthetic_landmarks,
    )
    settings = ApiSettings(allow_private_hosts=True, max_bytes=128)
    with TestClient(create_app(settings)) as small:
        error = _failed_doc(small, "/analyze", {"image_url": f"{image_server}/face.png"})
    assert error["code"] == "image_too_large"


def test_missing_face_fails_the_stored_result(monkeypatch, image_server):
    monkeypatch.setattr(
        "skin_metrics.pipeline.detect_landmarks", lambda img, model_path=None: None
    )
    with TestClient(create_app(ApiSettings(allow_private_hosts=True))) as faceless:
        error = _failed_doc(faceless, "/analyze/diary", {"image_url": f"{image_server}/face.png"})
    assert error["code"] == "analysis_failed"


# --- failures cheap enough to stay synchronous 4xx -------------------------


def test_analyze_blocks_private_host_by_default(image_server):
    with TestClient(create_app(ApiSettings())) as guarded:
        resp = guarded.post("/analyze", json={"image_url": f"{image_server}/face.png"})
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "blocked_host"


def test_analyze_rejects_non_http_url(client):
    resp = client.post("/analyze", json={"image_url": "file:///etc/passwd"})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_request"


def test_analyze_rejects_bad_bbox(client, image_server):
    resp = client.post(
        "/analyze",
        json={"image_url": f"{image_server}/face.png", "reference_bbox": [0, 0, 0, 10]},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_request"
