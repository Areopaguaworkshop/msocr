"""Tests for HTR deployment and smoke-check helpers."""

import json
from pathlib import Path


def test_runtime_htr_smoke_check_reports_default_route(monkeypatch):
    from msocr.service.deploy import runtime_htr_smoke_check

    monkeypatch.delenv("MSOCR_HTR_RUNTIME_MODEL_PATH", raising=False)
    monkeypatch.delenv("MSOCR_HTR_MODEL_PATH", raising=False)
    monkeypatch.delenv("MSOCR_RUNTIME_MODEL_PATH", raising=False)

    payload = runtime_htr_smoke_check(language="old_sogdian", script_variant="standard")

    assert payload["ok"] is True
    assert payload["mode"] == "htr"
    assert payload["language"] == "sogdian"
    assert payload["runtime_source"] == "default_route"
    assert payload["model_path"].endswith("models/kraken/sogdian_manuscript.mlmodel")


def test_runtime_htr_smoke_check_runs_htr_service_when_image_provided(monkeypatch, tmp_path: Path):
    from msocr.service.deploy import runtime_htr_smoke_check

    image_path = tmp_path / "line.png"
    image_path.write_bytes(b"png")

    def fake_resolve_htr_runtime_model_path(*, language, script_variant, model):
        assert language == "sogdian"
        assert script_variant == "standard"
        assert model is None
        return "/tmp/runtime/sogdian.mlmodel"

    def fake_run_htr_service(**kwargs):
        assert kwargs["image_path"] == image_path
        assert kwargs["variant"] == "standard"
        assert kwargs["device"] == "cpu"
        return {"text": "𐼷𐼹𐼻", "engine": "kraken", "language": "sogdian"}

    monkeypatch.setattr(
        "msocr.service.deploy.resolve_htr_runtime_model_path",
        fake_resolve_htr_runtime_model_path,
    )
    monkeypatch.setattr("msocr.service.deploy.run_htr_service", fake_run_htr_service)

    payload = runtime_htr_smoke_check(
        language="sogdian",
        script_variant="standard",
        image_path=image_path,
    )

    assert payload["engine"] == "kraken"
    assert payload["text_length"] == len("𐼷𐼹𐼻")


def test_runtime_http_htr_smoke_check_posts_to_htr_endpoint(monkeypatch, tmp_path: Path):
    from msocr.service.deploy import runtime_http_htr_smoke_check

    image_path = tmp_path / "line.png"
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

    observed = {"calls": []}

    def fake_urlopen(req, timeout=0):
        observed["calls"].append((req.full_url, req.get_method(), req.data))
        if req.full_url.endswith("/health"):
            return FakeResponse({"status": "ok", "service": "msocr-htr-api"})
        if req.full_url.endswith("/htr"):
            assert req.data is not None
            assert b'name="lang"' in req.data
            assert b"sogdian" in req.data
            assert b'name="variant"' in req.data
            return FakeResponse(
                {
                    "ok": True,
                    "text": "𐼷𐼹𐼻",
                    "engine": "kraken",
                    "language": "sogdian",
                    "mode": "htr",
                }
            )
        raise AssertionError(f"Unexpected URL: {req.full_url}")

    monkeypatch.setattr("msocr.service.deploy.request.urlopen", fake_urlopen)

    payload = runtime_http_htr_smoke_check(
        base_url="http://127.0.0.1:18020",
        language="sogdian",
        script_variant="standard",
        image_path=image_path,
        timeout_sec=1,
        poll_interval_sec=0,
    )

    assert payload["ok"] is True
    assert payload["htr"]["engine"] == "kraken"
    assert payload["htr"]["language"] == "sogdian"
    assert payload["htr"]["text_length"] == len("𐼷𐼹𐼻")
    assert [call[1] for call in observed["calls"]] == ["GET", "POST"]


def test_runtime_http_htr_smoke_check_rejects_empty_text(monkeypatch, tmp_path: Path):
    from msocr.service.deploy import runtime_http_htr_smoke_check

    image_path = tmp_path / "line.png"
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
        if req.full_url.endswith("/htr"):
            return FakeResponse({"ok": True, "text": "", "engine": "kraken"})
        raise AssertionError(f"Unexpected URL: {req.full_url}")

    monkeypatch.setattr("msocr.service.deploy.request.urlopen", fake_urlopen)

    try:
        runtime_http_htr_smoke_check(
            base_url="http://127.0.0.1:18020",
            language="sogdian",
            script_variant="standard",
            image_path=image_path,
            timeout_sec=1,
            poll_interval_sec=0,
        )
    except RuntimeError as exc:
        assert "empty text" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for empty HTR smoke text")


def test_gradio_http_smoke_check_waits_for_expected_text(monkeypatch):
    from msocr.service.deploy import gradio_http_smoke_check

    class FakeResponse:
        def read(self):
            return b"<html><body><h1>msocr Sogdian HTR</h1></body></html>"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(req, timeout=0):
        assert req.full_url == "http://127.0.0.1:7860/"
        return FakeResponse()

    monkeypatch.setattr("msocr.service.deploy.request.urlopen", fake_urlopen)

    assert gradio_http_smoke_check(
        base_url="http://127.0.0.1:7860",
        timeout_sec=1,
        poll_interval_sec=0,
    ) == {
        "ok": True,
        "base_url": "http://127.0.0.1:7860",
        "expected_text": "msocr Sogdian HTR",
    }
