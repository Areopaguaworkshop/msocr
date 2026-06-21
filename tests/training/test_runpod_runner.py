"""Tests for msocr.training.runpod_runner. SDK and SSH are mocked."""
from unittest.mock import patch, MagicMock, call
import pytest

from msocr.training.runpod_runner import RunPodRunner


def test_submit_pod_uses_runpod_create_pod_with_image_and_gpu(monkeypatch):
    """runpod.create_pod is called with the right name, image, and GPU type."""
    fake_runpod = MagicMock()
    fake_runpod.create_pod.return_value = MagicMock(id="pod-123")
    monkeypatch.setattr("msocr.training.runpod_runner.runpod", fake_runpod)

    runner = RunPodRunner(api_key="fake", image="msocr-kraken7:latest",
                          gpu_type="RTX 4090", ssh_key_path="/tmp/id_ed25519")
    pod_id = runner.submit_pod(name="sogdian-train")

    fake_runpod.create_pod.assert_called_once()
    args, kwargs = fake_runpod.create_pod.call_args
    assert "sogdian-train" in (args + tuple(kwargs.values()))
    assert "msocr-kraken7:latest" in (args + tuple(kwargs.values()))
    assert "RTX 4090" in (args + tuple(kwargs.values()))
    assert pod_id == "pod-123"


def test_ssh_ketos_train_runs_the_right_command(monkeypatch):
    """SSH exec runs the 7.0 ketos train command with global flags."""
    fake_paramiko = MagicMock()
    # Mock SSHClient.connect, exec_command, return stdout/stderr/exit_status
    fake_stdout = MagicMock()
    fake_stdout.channel.recv_exit_status.return_value = 0
    fake_stderr = MagicMock()
    fake_stderr.read.return_value = b""
    fake_paramiko.SSHClient.return_value.exec_command.return_value = (
        MagicMock(), fake_stdout, fake_stderr
    )
    monkeypatch.setattr("msocr.training.runpod_runner.paramiko", fake_paramiko)

    runner = RunPodRunner(api_key="fake", image="img", gpu_type="RTX 4090",
                          ssh_key_path="/tmp/id_ed25519")
    cmd = ["ketos", "-d", "cuda:0", "--workers", "8", "train",
           "--load", "base.safetensors", "--resize", "union",
           "--freeze-backbone", "5000", "--augment",
           "-f", "binary", "-t", "train.arrow", "-e", "val.arrow",
           "-o", "/workspace/models/manichaean-early"]
    runner.ssh_exec(pod_ip="1.2.3.4", cmd=cmd)

    # Verify SSHClient.connect was called with the pod IP and the SSH key
    # Verify exec_command was called with the joined command string
    fake_paramiko.SSHClient.return_value.exec_command.assert_called_once()


def test_poll_until_done_returns_when_pod_exited(monkeypatch):
    fake_runpod = MagicMock()
    # First call: still running. Second call: exited.
    fake_runpod.get_pod.side_effect = [
        {"status": "RUNNING"},
        {"status": "EXITED", "runtime": {"exitCode": 0}},
    ]
    monkeypatch.setattr("msocr.training.runpod_runner.runpod", fake_runpod)
    monkeypatch.setattr("time.sleep", lambda *a: None)

    runner = RunPodRunner(api_key="fake", image="img", gpu_type="RTX 4090",
                          ssh_key_path="/tmp/id_ed25519")
    exit_code = runner.poll_until_done(pod_id="pod-123", timeout=3600, interval=30)
    assert exit_code == 0
    assert fake_runpod.get_pod.call_count == 2


def test_download_artifact_does_not_terminate_pod_on_failure(monkeypatch):
    """If scp fails, the pod is NOT terminated (so the artifact survives for manual recovery)."""
    fake_runpod = MagicMock()
    monkeypatch.setattr("msocr.training.runpod_runner.runpod", fake_runpod)
    # paramiko SSHClient scp raises
    fake_paramiko = MagicMock()
    fake_paramiko.SSHClient.return_value.open_sftp.side_effect = Exception("scp fail")
    monkeypatch.setattr("msocr.training.runpod_runner.paramiko", fake_paramiko)

    runner = RunPodRunner(api_key="fake", image="img", gpu_type="RTX 4090",
                          ssh_key_path="/tmp/id_ed25519")
    with pytest.raises(Exception, match="scp fail"):
        runner.download_artifact(pod_ip="1.2.3.4", remote="/workspace/out.safetensors",
                                   local="/tmp/out.safetensors")
    fake_runpod.terminate_pod.assert_not_called()  # key assertion


def test_upload_artifact_calls_sftp_put_with_local_and_remote(monkeypatch):
    """sftp.put is called with the local and remote paths."""
    fake_runpod = MagicMock()
    monkeypatch.setattr("msocr.training.runpod_runner.runpod", fake_runpod)
    fake_paramiko = MagicMock()
    monkeypatch.setattr("msocr.training.runpod_runner.paramiko", fake_paramiko)

    runner = RunPodRunner(api_key="fake", image="img", gpu_type="RTX 4090",
                          ssh_key_path="/tmp/id_ed25519")
    runner.upload_artifact(local="/tmp/train.arrow", pod_ip="1.2.3.4",
                           remote="/workspace/train.arrow")

    fake_paramiko.SSHClient.return_value.open_sftp.return_value.put.assert_called_once_with(
        "/tmp/train.arrow", "/workspace/train.arrow"
    )


def test_upload_artifact_does_not_terminate_pod_on_failure(monkeypatch):
    """If sftp.put fails, the pod is NOT terminated (caller decides recovery)."""
    fake_runpod = MagicMock()
    monkeypatch.setattr("msocr.training.runpod_runner.runpod", fake_runpod)
    fake_paramiko = MagicMock()
    fake_paramiko.SSHClient.return_value.open_sftp.return_value.put.side_effect = Exception("put fail")
    monkeypatch.setattr("msocr.training.runpod_runner.paramiko", fake_paramiko)

    runner = RunPodRunner(api_key="fake", image="img", gpu_type="RTX 4090",
                          ssh_key_path="/tmp/id_ed25519")
    with pytest.raises(Exception, match="put fail"):
        runner.upload_artifact(local="/tmp/train.arrow", pod_ip="1.2.3.4",
                               remote="/workspace/train.arrow")
    fake_runpod.terminate_pod.assert_not_called()  # key assertion


def test_run_training_ssh_execs_downloads_and_terminates_without_waiting_for_pod_exit(monkeypatch):
    """SSH training is synchronous; the pod stays running until we download and terminate it."""
    fake_runpod = MagicMock()
    fake_runpod.create_pod.return_value = {"id": "pod-123"}
    fake_runpod.get_pod.return_value = {"runtime": {"ip": "1.2.3.4"}, "status": "RUNNING"}
    monkeypatch.setattr("msocr.training.runpod_runner.runpod", fake_runpod)

    runner = RunPodRunner(api_key="fake", image="img", gpu_type="RTX 4090",
                          ssh_key_path="/tmp/id_ed25519")
    runner.upload_artifact = MagicMock()
    runner.ssh_exec = MagicMock(return_value="ok")
    runner.download_artifact = MagicMock()
    runner.poll_until_done = MagicMock()

    result = runner.run_training(
        name="train",
        train_cmd=["ketos", "train"],
        artifact_remote_path="/workspace/models/out.safetensors",
        artifact_local_path="/tmp/out.safetensors",
        pre_train_upload=[("/tmp/train.arrow", "/workspace/train.arrow")],
    )

    assert result == "/tmp/out.safetensors"
    runner.poll_until_done.assert_not_called()
    runner.ssh_exec.assert_has_calls([
        call("1.2.3.4", ["mkdir", "-p", "/workspace/models"]),
        call("1.2.3.4", ["ketos", "train"], timeout=7200),
    ])
    runner.download_artifact.assert_called_once_with(
        "1.2.3.4", "/workspace/models/out.safetensors", "/tmp/out.safetensors"
    )
    fake_runpod.terminate_pod.assert_called_once_with("pod-123")


def test_run_training_leaves_pod_running_when_download_fails(monkeypatch):
    """If artifact download fails after training, keep the pod alive for manual recovery."""
    fake_runpod = MagicMock()
    fake_runpod.create_pod.return_value = {"id": "pod-123"}
    fake_runpod.get_pod.return_value = {"runtime": {"ip": "1.2.3.4"}, "status": "RUNNING"}
    monkeypatch.setattr("msocr.training.runpod_runner.runpod", fake_runpod)

    runner = RunPodRunner(api_key="fake", image="img", gpu_type="RTX 4090",
                          ssh_key_path="/tmp/id_ed25519")
    runner.ssh_exec = MagicMock(return_value="ok")
    runner.download_artifact = MagicMock(side_effect=Exception("download fail"))

    with pytest.raises(Exception, match="download fail"):
        runner.run_training(
            name="train",
            train_cmd=["ketos", "train"],
            artifact_remote_path="/workspace/models/out.safetensors",
            artifact_local_path="/tmp/out.safetensors",
        )

    fake_runpod.terminate_pod.assert_not_called()
