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


def test_ssh_endpoint_polls_until_runtime_ports_has_public_22(monkeypatch):
    """_ssh_endpoint polls get_pod until runtime.ports has a public entry for port 22."""
    fake_runpod = MagicMock()
    # First: runtime None. Second: ports empty. Third: SSH port appears.
    fake_runpod.get_pod.side_effect = [
        {"id": "p", "runtime": None},
        {"id": "p", "runtime": {"ports": []}},
        {"id": "p", "runtime": {"ports": [
            {"privatePort": 22, "isIpPublic": True, "ip": "1.2.3.4", "publicPort": 17445},
        ]}},
    ]
    monkeypatch.setattr("msocr.training.runpod_runner.runpod", fake_runpod)
    monkeypatch.setattr("time.sleep", lambda *a: None)

    runner = RunPodRunner(api_key="fake", image="img", gpu_type="RTX 4090",
                          ssh_key_path="/tmp/id_ed25519")
    host, port = runner._ssh_endpoint("p", deadline_s=60)
    assert host == "1.2.3.4"
    assert port == 17445
    assert fake_runpod.get_pod.call_count == 3


def test_ssh_endpoint_treats_none_ports_as_not_ready(monkeypatch):
    """RunPod can report runtime.ports as None before port metadata is ready."""
    fake_runpod = MagicMock()
    fake_runpod.get_pod.side_effect = [
        {"id": "p", "runtime": {"ports": None}},
        {"id": "p", "runtime": {"ports": [
            {"privatePort": 22, "isIpPublic": True, "ip": "1.2.3.4", "publicPort": 17445},
        ]}},
    ]
    monkeypatch.setattr("msocr.training.runpod_runner.runpod", fake_runpod)
    monkeypatch.setattr("time.sleep", lambda *a: None)

    runner = RunPodRunner(api_key="fake", image="img", gpu_type="RTX 4090",
                          ssh_key_path="/tmp/id_ed25519")
    host, port = runner._ssh_endpoint("p", deadline_s=60)
    assert host == "1.2.3.4"
    assert port == 17445
    assert fake_runpod.get_pod.call_count == 2


def test_ssh_endpoint_raises_when_no_public_port(monkeypatch):
    """If runtime.ports never exposes a public port 22, raise RuntimeError."""
    fake_runpod = MagicMock()
    fake_runpod.get_pod.return_value = {"id": "p", "runtime": None}
    monkeypatch.setattr("msocr.training.runpod_runner.runpod", fake_runpod)
    monkeypatch.setattr("time.sleep", lambda *a: None)

    runner = RunPodRunner(api_key="fake", image="img", gpu_type="RTX 4090",
                          ssh_key_path="/tmp/id_ed25519")
    with pytest.raises(RuntimeError, match="never exposed a public SSH port"):
        runner._ssh_endpoint("p", deadline_s=1)


def test_ssh_exec_connects_with_host_and_port(monkeypatch):
    """ssh_exec passes (host, port) to paramiko connect."""
    fake_paramiko = MagicMock()
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
    runner.ssh_exec(("1.2.3.4", 17445), ["ketos", "train"])

    fake_client = fake_paramiko.SSHClient.return_value
    fake_client.connect.assert_called_once()
    args, kwargs = fake_client.connect.call_args
    assert args[0] == "1.2.3.4"
    assert kwargs["port"] == 17445
    fake_client.exec_command.assert_called_once()


def test_download_artifact_does_not_terminate_pod_on_failure(monkeypatch):
    """If scp fails, the pod is NOT terminated (so the artifact survives for manual recovery)."""
    fake_runpod = MagicMock()
    monkeypatch.setattr("msocr.training.runpod_runner.runpod", fake_runpod)
    fake_paramiko = MagicMock()
    fake_paramiko.SSHClient.return_value.open_sftp.side_effect = Exception("scp fail")
    monkeypatch.setattr("msocr.training.runpod_runner.paramiko", fake_paramiko)

    runner = RunPodRunner(api_key="fake", image="img", gpu_type="RTX 4090",
                          ssh_key_path="/tmp/id_ed25519")
    with pytest.raises(Exception, match="scp fail"):
        runner.download_artifact(("1.2.3.4", 17445), "/workspace/out.safetensors",
                                  "/tmp/out.safetensors")
    fake_runpod.terminate_pod.assert_not_called()


def test_upload_artifact_calls_sftp_put_with_local_and_remote(monkeypatch):
    """sftp.put is called with the local and remote paths."""
    fake_runpod = MagicMock()
    monkeypatch.setattr("msocr.training.runpod_runner.runpod", fake_runpod)
    fake_paramiko = MagicMock()
    monkeypatch.setattr("msocr.training.runpod_runner.paramiko", fake_paramiko)

    runner = RunPodRunner(api_key="fake", image="img", gpu_type="RTX 4090",
                          ssh_key_path="/tmp/id_ed25519")
    runner.upload_artifact("/tmp/train.arrow", ("1.2.3.4", 17445),
                            "/workspace/train.arrow")

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
        runner.upload_artifact("/tmp/train.arrow", ("1.2.3.4", 17445),
                                "/workspace/train.arrow")
    fake_runpod.terminate_pod.assert_not_called()


def test_run_training_full_lifecycle(monkeypatch):
    """submit → upload → mkdir → setup → train → download → terminate."""
    fake_runpod = MagicMock()
    fake_runpod.create_pod.return_value = {"id": "pod-123"}
    monkeypatch.setattr("msocr.training.runpod_runner.runpod", fake_runpod)

    runner = RunPodRunner(api_key="fake", image="img", gpu_type="RTX 4090",
                          ssh_key_path="/tmp/id_ed25519")
    runner._ssh_endpoint = MagicMock(return_value=("1.2.3.4", 17445))
    runner.upload_artifact = MagicMock()
    runner.ssh_exec = MagicMock(return_value="ok")
    runner.download_artifact = MagicMock()

    result = runner.run_training(
        name="train",
        train_cmd=["ketos", "train"],
        artifact_remote_path="/workspace/models/out.safetensors",
        artifact_local_path="/tmp/out.safetensors",
        pre_train_upload=[("/tmp/train.arrow", "/workspace/train.arrow")],
        setup_cmds=["pip install kraken"],
    )

    assert result == "/tmp/out.safetensors"
    runner.upload_artifact.assert_called_once_with(
        "/tmp/train.arrow", ("1.2.3.4", 17445), "/workspace/train.arrow"
    )
    runner.ssh_exec.assert_has_calls([
        call(("1.2.3.4", 17445), ["mkdir", "-p", "/workspace/models"]),
        call(("1.2.3.4", 17445), ["pip", "install", "kraken"], timeout=600),
        call(("1.2.3.4", 17445), ["ketos", "train"], timeout=7200),
    ])
    runner.download_artifact.assert_called_once_with(
        ("1.2.3.4", 17445), "/workspace/models/out.safetensors", "/tmp/out.safetensors"
    )
    fake_runpod.terminate_pod.assert_called_once_with("pod-123")


def test_run_training_leaves_pod_running_when_download_fails(monkeypatch):
    """If artifact download fails after training, keep the pod alive for manual recovery."""
    fake_runpod = MagicMock()
    fake_runpod.create_pod.return_value = {"id": "pod-123"}
    monkeypatch.setattr("msocr.training.runpod_runner.runpod", fake_runpod)

    runner = RunPodRunner(api_key="fake", image="img", gpu_type="RTX 4090",
                          ssh_key_path="/tmp/id_ed25519")
    runner._ssh_endpoint = MagicMock(return_value=("1.2.3.4", 17445))
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
