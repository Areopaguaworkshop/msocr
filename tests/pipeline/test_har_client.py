"""Tests for Harness Artifact Registry helpers."""

from pathlib import Path

from msocr.pipeline.har_client import HARClient, build_bundle, build_model_artifact_name


def test_build_model_artifact_name_uses_iso_language_token():
    assert build_model_artifact_name("syriac", "estrangela", "printed") == "syr-estrangela-printed"


def test_build_bundle_includes_sidecar_paths(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("HARNESS_API_TOKEN", raising=False)
    monkeypatch.delenv("HARNESS_API_KEY", raising=False)
    monkeypatch.delenv("HARNESS_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("HARNESS_ORG_ID", raising=False)
    monkeypatch.delenv("HARNESS_PROJECT_ID", raising=False)

    model_file = tmp_path / "model.mlmodel"
    metrics_file = tmp_path / "metrics.json"
    model_file.write_text("model", encoding="utf-8")
    metrics_file.write_text("{}", encoding="utf-8")

    bundle = build_bundle(
        registry="msocr-models",
        language="syriac",
        script_variant="estrangela",
        writing_mode="printed",
        version="v14",
        model_file=model_file,
        metrics_file=metrics_file,
        metadata={"stage": "staging"},
    )
    client = HARClient()
    commands = client.plan_commands(bundle)

    assert bundle.artifact_ref == "syr-estrangela-printed:v14"
    assert len(bundle.files) == 2
    assert bundle.files[1].package_path == "sidecars/metrics.json"
    assert commands[0][:4] == ["hc", "artifact", "push", "generic"]
    assert "--metadata" in commands[-1]


def test_plan_commands_include_noninteractive_login_when_token_present(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HARNESS_API_TOKEN", "pat.abc123.xyz789.qwe456")
    monkeypatch.setenv("HARNESS_ACCOUNT_ID", "acc123")
    monkeypatch.setenv("HARNESS_ORG_ID", "msocr")
    monkeypatch.setenv("HARNESS_PROJECT_ID", "ocr")

    model_file = tmp_path / "model.traineddata"
    model_file.write_text("model", encoding="utf-8")
    bundle = build_bundle(
        registry="msocr-models",
        language="syriac",
        script_variant="estrangela",
        writing_mode="printed",
        version="v14",
        model_file=model_file,
    )

    client = HARClient()
    commands = client.plan_commands(bundle)

    assert commands[0][:3] == ["hc", "auth", "login"]
    assert "$HARNESS_API_TOKEN" in commands[0]
    assert "--non-interactive" in commands[0]


def test_build_pull_command_uses_generic_package_path(tmp_path: Path):
    destination = tmp_path / "downloads" / "model.traineddata"
    client = HARClient(pkg_url="https://pkg.harness.io")

    command = client.build_pull_command(
        registry="msocr-models",
        package_name="syr-estrangela-printed",
        version="v14",
        filename="model.traineddata",
        destination=destination,
    )

    assert command[:4] == ["hc", "artifact", "pull", "generic"]
    assert command[4] == "msocr-models"
    assert command[5] == "syr-estrangela-printed/v14/model.traineddata"
    assert command[6] == str(destination)
