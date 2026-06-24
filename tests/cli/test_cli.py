"""Smoke tests for the new CLI subcommands (train-remote, evaluate, annotate).

ponytail: CliRunner + --help assertions are the smallest thing that fails if
the Click decorators, names, or options regress. No fixtures, no live calls.
"""
from click.testing import CliRunner

from msocr.cli import main


def _help_flags(cmd_name):
    runner = CliRunner()
    result = runner.invoke(main, [cmd_name, "--help"])
    assert result.exit_code == 0, result.output
    return result.output


def test_train_remote_help_lists_all_flags():
    out = _help_flags("train-remote")
    for flag in [
        "--manifest", "--style-group", "--base-model", "--output-model",
        "--reports-dir", "--pod-gpu", "--pod-image", "--ssh-key",
        "--epochs", "--min-epochs", "--lag", "--freeze-backbone",
        "--augment", "--device", "--workers",
    ]:
        assert flag in out, f"missing {flag!r} in train-remote help"


def test_evaluate_help_lists_all_flags():
    out = _help_flags("evaluate")
    for flag in ["--manifest", "--style-group", "--model", "--reports-dir"]:
        assert flag in out, f"missing {flag!r} in evaluate help"


def test_annotate_help_lists_all_flags():
    out = _help_flags("annotate")
    for flag in ["--host", "--port", "--base-dir"]:
        assert flag in out, f"missing {flag!r} in annotate help"


def test_extract_lines_help_lists_all_flags():
    out = _help_flags("extract-lines")
    for flag in ["--expected-lines", "--output-dir", "--roi", "--row-centers", "--min-component-area"]:
        assert flag in out, f"missing {flag!r} in extract-lines help"


def test_train_remote_requires_runpod_api_key(tmp_path, monkeypatch):
    """No RUNPOD_API_KEY -> ClickException before any pod call is made."""
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    runner = CliRunner()
    result = runner.invoke(main, [
        "train-remote",
        "--manifest", str(tmp_path / "m.json"),
        "--style-group", "g1",
        "--base-model", str(tmp_path / "base.safetensors"),
        "--output-model", str(tmp_path / "out.safetensors"),
    ])
    assert result.exit_code != 0
    assert "RUNPOD_API_KEY" in result.output


def test_train_remote_reports_missing_ssh_key_before_pod_call(tmp_path, monkeypatch):
    """A missing private key should name the exact path before creating a pod."""
    monkeypatch.setenv("RUNPOD_API_KEY", "fake")
    missing_key = tmp_path / "missing_id_ed25519"
    runner = CliRunner()
    result = runner.invoke(main, [
        "train-remote",
        "--manifest", str(tmp_path / "m.json"),
        "--style-group", "g1",
        "--output-model", str(tmp_path / "out.safetensors"),
        "--ssh-key", str(missing_key),
    ])

    assert result.exit_code != 0
    assert "SSH private key not found" in result.output
    assert str(missing_key) in result.output


def test_main_help_lists_new_subcommands():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0, result.output
    out = result.output
    for name in ["train-remote", "evaluate", "annotate", "extract-lines"]:
        assert name in out, f"missing {name!r} in top-level help"
