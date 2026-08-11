"""Tests for the HTTP API (skin_metrics.api).

Everything runs against a throwaway loopback HTTP server, so no external network
is touched. MediaPipe is never needed: ``skin_metrics.pipeline.detect_landmarks``
is monkeypatched with the synthetic landmarks from ``conftest``.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from skin_metrics.api.app import create_app  # noqa: E402
from skin_metrics.api.fetch import ImageFetchError, decode_image, validate_url  # noqa: E402
from skin_metrics.api.schemas import AnalyzeRequest, SimpleAnalyzeResponse  # noqa: E402
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


def test_decode_image_rejects_too_many_pixels(synthetic_image):
    with pytest.raises(ImageFetchError) as exc:
        decode_image(_png_bytes(synthetic_image), max_pixels=100)
    assert exc.value.code == "image_too_large"
    assert exc.value.status_code == 413


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
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


def test_analyze_returns_flat_scores(client, image_server):
    resp = client.post("/analyze", json={"image_url": f"{image_server}/face.png"})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    for metric in ("pigmentation", "erythema", "hydration"):
        assert 0.0 <= body[metric] <= 100.0
        assert 0.0 <= body["confidence"][metric] <= 1.0
    assert body["disclaimer"]
    assert body["elapsed_ms"] >= 0.0


def test_analyze_and_simple_share_the_same_envelope(client, image_server):
    """Both endpoints must answer in the same flat shape, only the scores differ."""
    full = client.post("/analyze", json={"image_url": f"{image_server}/face.png"}).json()
    simple = client.post(
        "/analyze/simple", json={"image_url": f"{image_server}/face.png"}
    ).json()

    full_scores = {"pigmentation", "erythema", "hydration"}
    simple_scores = {"skin_tone", "dryness", "redness"}
    assert set(full) - full_scores == set(simple) - simple_scores
    assert set(full["confidence"]) == full_scores
    assert set(simple["confidence"]) == simple_scores


def _report_stub(ita=None, hydration=70.0, erythema=35.0) -> SkinReport:
    """Minimal SkinReport for exercising the simple-score mapping."""
    pig_features = {} if ita is None else {"ita": ita}
    return SkinReport(
        pigmentation=MetricScore(score=50.0, confidence=0.8, raw_features=pig_features),
        erythema=MetricScore(score=erythema, confidence=0.7),
        hydration=MetricScore(score=hydration, confidence=0.6, is_estimate=True),
        calibration_status="none",
        fitzpatrick_estimate=3,
    )


def test_simple_mapping_orients_and_scales_each_score():
    simple = SimpleAnalyzeResponse.from_report(
        _report_stub(ita=34.5, hydration=70.0, erythema=35.0), elapsed_ms=1.0
    )
    # ITA 34.5 sits (34.5+30)/85 of the way from dark to very light.
    assert simple.skin_tone == pytest.approx(round(10.0 * 64.5 / 85.0, 1))
    # hydration 70 (moist) -> dryness 3.0; erythema 35 -> redness 3.5.
    assert simple.dryness == pytest.approx(3.0)
    assert simple.redness == pytest.approx(3.5)
    assert simple.confidence == {"skin_tone": 0.8, "dryness": 0.6, "redness": 0.7}


@pytest.mark.parametrize("ita,expected", [(55.0, 10.0), (-30.0, 0.0), (90.0, 10.0), (-80.0, 0.0)])
def test_simple_mapping_clips_skin_tone_to_0_10(ita, expected):
    simple = SimpleAnalyzeResponse.from_report(_report_stub(ita=ita), elapsed_ms=1.0)
    assert simple.skin_tone == pytest.approx(expected)


def test_simple_mapping_survives_a_missing_ita():
    """No `ita` in raw_features -> Fitzpatrick bucket centre, not a crash."""
    simple = SimpleAnalyzeResponse.from_report(_report_stub(ita=None), elapsed_ms=1.0)
    assert 0.0 <= simple.skin_tone <= 10.0


def test_analyze_simple_returns_scores(client, image_server):
    resp = client.post("/analyze/simple", json={"image_url": f"{image_server}/face.png"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    for field in ("skin_tone", "dryness", "redness"):
        assert 0.0 <= body[field] <= 10.0
        assert 0.0 <= body["confidence"][field] <= 1.0
    assert body["disclaimer"]
    assert body["elapsed_ms"] >= 0.0


def test_analyze_simple_is_consistent_with_the_full_scores(client, image_server):
    full = client.post("/analyze", json={"image_url": f"{image_server}/face.png"}).json()
    simple = client.post(
        "/analyze/simple", json={"image_url": f"{image_server}/face.png"}
    ).json()
    assert simple["dryness"] == pytest.approx(
        round((100.0 - full["hydration"]) / 10.0, 1), abs=0.05
    )
    assert simple["redness"] == pytest.approx(round(full["erythema"] / 10.0, 1), abs=0.05)


def test_analyze_simple_propagates_fetch_errors(client, image_server):
    resp = client.post("/analyze/simple", json={"image_url": f"{image_server}/missing.png"})
    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "upstream_error"


def test_analyze_simple_reports_missing_face(monkeypatch, image_server):
    monkeypatch.setattr(
        "skin_metrics.pipeline.detect_landmarks", lambda img, model_path=None: None
    )
    with TestClient(create_app(ApiSettings(allow_private_hosts=True))) as faceless:
        resp = faceless.post(
            "/analyze/simple", json={"image_url": f"{image_server}/face.png"}
        )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "analysis_failed"


def test_analyze_with_reference_bbox(client, image_server):
    resp = client.post(
        "/analyze",
        json={"image_url": f"{image_server}/face.png", "reference_bbox": [5, 5, 20, 20]},
    )
    assert resp.status_code == 200, resp.text


def test_analyze_follows_redirect(client, image_server):
    resp = client.post("/analyze", json={"image_url": f"{image_server}/redirect"})
    assert resp.status_code == 200, resp.text
    assert 0.0 <= resp.json()["pigmentation"] <= 100.0


def test_analyze_redirect_loop_is_bounded(client, image_server):
    resp = client.post("/analyze", json={"image_url": f"{image_server}/loop"})
    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "too_many_redirects"


def test_analyze_rejects_non_image(client, image_server):
    resp = client.post("/analyze", json={"image_url": f"{image_server}/notimage.txt"})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "decode_error"


def test_analyze_rejects_empty_body(client, image_server):
    resp = client.post("/analyze", json={"image_url": f"{image_server}/empty.png"})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "empty_body"


def test_analyze_maps_upstream_404(client, image_server):
    resp = client.post("/analyze", json={"image_url": f"{image_server}/missing.png"})
    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "upstream_error"


def test_analyze_enforces_byte_limit(monkeypatch, synthetic_landmarks, image_server):
    monkeypatch.setattr(
        "skin_metrics.pipeline.detect_landmarks",
        lambda img, model_path=None: synthetic_landmarks,
    )
    settings = ApiSettings(allow_private_hosts=True, max_bytes=128)
    with TestClient(create_app(settings)) as small:
        resp = small.post("/analyze", json={"image_url": f"{image_server}/face.png"})
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "image_too_large"


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


def test_analyze_reports_missing_face(monkeypatch, image_server):
    monkeypatch.setattr(
        "skin_metrics.pipeline.detect_landmarks", lambda img, model_path=None: None
    )
    with TestClient(create_app(ApiSettings(allow_private_hosts=True))) as faceless:
        resp = faceless.post("/analyze", json={"image_url": f"{image_server}/face.png"})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "analysis_failed"
