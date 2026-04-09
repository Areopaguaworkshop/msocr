"""Tests for Harness Artifact Registry helpers."""

from pathlib import Path

from msocr.pipeline.har_client import HARClient, build_bundle, build_model_artifact_name


def test_build_model_artifact_name_uses_iso_language_token():
    assert build_model_artifact_name("syriac", "estrangela", "printed") == "syr-estrangela-printed"


def test_build_bundle_includes_sidecar_paths(tmp_path: Path):
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
