"""Tests for deploy/runtime startup helpers."""

import json
from pathlib import Path


def test_runtime_smoke_check_reports_har_source(monkeypatch, tmp_path: Path):
    from msocr.service.deploy import runtime_smoke_check

    pulled_model = tmp_path / "runtime" / "syr-estrangela-printed" / "v14" / "model.traineddata"

    def fake_resolve_printed_runtime_model_path(*, language, script_variant, model):
        assert language == "syriac"
        assert script_variant == "estrangela"
        assert model is None
        return str(pulled_model)

    monkeypatch.setenv("MSOCR_RUNTIME_HAR_REGISTRY", "msocr-models")
    monkeypatch.setattr(
        "msocr.service.deploy.resolve_printed_runtime_model_path",
        fake_resolve_printed_runtime_model_path,
    )

    payload = runtime_smoke_check(language="syriac", script_variant="estrangela")

    assert payload["ok"] is True
    assert payload["runtime_source"] == "har"
    assert payload["model_path"] == str(pulled_model)


def test_runtime_smoke_check_runs_printed_service_when_image_provided(monkeypatch, tmp_path: Path):
    from msocr.service.deploy import runtime_smoke_check

    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"png")

    def fake_resolve_printed_runtime_model_path(*, language, script_variant, model):
        return "/tmp/runtime/model.traineddata"

    def fake_run_printed_service(**kwargs):
        assert kwargs["image_path"] == image_path
        assert kwargs["variant"] == "estrangela"
        return {"text": "ܫܠܡܐ", "engine": "tesseract", "language": "syriac"}

    monkeypatch.setattr(
        "msocr.service.deploy.resolve_printed_runtime_model_path",
        fake_resolve_printed_runtime_model_path,
    )
    monkeypatch.setattr("msocr.service.deploy.run_printed_service", fake_run_printed_service)

    payload = runtime_smoke_check(
        language="syriac",
        script_variant="estrangela",
        image_path=image_path,
        engine="tesseract",
    )

    assert payload["engine"] == "tesseract"
    assert payload["text_length"] == len("ܫܠܡܐ")


def test_runtime_http_smoke_check_uses_canonical_image(monkeypatch):
    from msocr.service.deploy import runtime_http_smoke_check

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def read(self):
            return json.dumps(self.payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    observed = {"calls": []}

    def fake_urlopen(req, timeout=0):
        observed["calls"].append((req.full_url, req.get_method(), req.data, timeout))
        if req.full_url.endswith("/health"):
            return FakeResponse({"status": "ok", "service": "msocr-api"})
        if req.full_url.endswith("/ocr"):
            assert req.data is not None
            assert b'name="lang"' in req.data
            assert b"syriac" in req.data
            assert b'name="variant"' in req.data
            assert b"estrangela" in req.data
            return FakeResponse(
                {
                    "ok": True,
                    "text": "ܐܒܓ ܕܗܘ",
                    "engine": "tesseract",
                    "language": "syriac",
                    "mode": "ocr",
                }
            )
        raise AssertionError(f"Unexpected URL: {req.full_url}")

    monkeypatch.setattr("msocr.service.deploy.request.urlopen", fake_urlopen)

    payload = runtime_http_smoke_check(
        base_url="http://127.0.0.1:18000",
        language="syriac",
        script_variant="estrangela",
        engine="tesseract",
        timeout_sec=1,
        poll_interval_sec=0,
    )

    assert payload["ok"] is True
    assert payload["health"]["status"] == "ok"
    assert payload["ocr"]["engine"] == "tesseract"
    assert payload["ocr"]["text_length"] == len("ܐܒܓ ܕܗܘ")
    assert payload["ocr"]["image_path"].endswith("assets/runtime/syriac_estrangela_smoke.png")
    assert [call[1] for call in observed["calls"]] == ["GET", "POST"]


def test_runtime_http_smoke_check_rejects_empty_text(monkeypatch, tmp_path: Path):
    from msocr.service.deploy import runtime_http_smoke_check

    image_path = tmp_path / "smoke.png"
    image_path.write_bytes(b"png")

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def read(self):
            return json.dumps(self.payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(req, timeout=0):
        if req.full_url.endswith("/health"):
            return FakeResponse({"status": "ok"})
        if req.full_url.endswith("/ocr"):
            return FakeResponse({"ok": True, "text": "", "engine": "tesseract"})
        raise AssertionError(f"Unexpected URL: {req.full_url}")

    monkeypatch.setattr("msocr.service.deploy.request.urlopen", fake_urlopen)

    try:
        runtime_http_smoke_check(
            base_url="http://127.0.0.1:18000",
            language="syriac",
            script_variant="estrangela",
            image_path=image_path,
            timeout_sec=1,
            poll_interval_sec=0,
        )
    except RuntimeError as exc:
        assert "empty text" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for empty OCR smoke text")
