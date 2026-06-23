"""RunPod GPU Cloud Pod runner for remote ketos training.

Per design D7: GPU Cloud Pods (SSH-style), not Serverless Endpoints.
Submit a pod from a custom Docker image, SSH-exec `ketos train`,
poll pod status, download the .safetensors artifact, terminate the pod.

Ponytail: procedural, one pod at a time. No queue, no DAG. If we need
durable parallelism later, add RQ on top.
"""
from __future__ import annotations

import os
import time
import shlex
from pathlib import Path
from typing import Optional

import runpod
import paramiko


class RunPodRunner:
    """Submit, SSH-train, poll, download, terminate one RunPod GPU Cloud Pod."""

    def __init__(self, api_key: str, image: str, gpu_type: str,
                 ssh_key_path: str, ssh_user: str = "root",
                 pod_disk_gb: int = 50):
        self.api_key = api_key
        self.image = image
        self.gpu_type = gpu_type
        self.ssh_key_path = ssh_key_path
        self.ssh_user = ssh_user
        self.pod_disk_gb = pod_disk_gb
        runpod.api_key = api_key

    def submit_pod(self, name: str) -> str:
        """Create a GPU Cloud Pod. Returns pod_id.

        ponytail: ports="22/tcp" ensures SSH is exposed; account-level SSH key
        (RunPod console settings) is auto-injected — no per-pod env var needed.
        """
        resp = runpod.create_pod(
            name=name,
            image_name=self.image,
            gpu_type_id=self.gpu_type,
            container_disk_in_gb=self.pod_disk_gb,
            ports="22/tcp",
        )
        return resp["id"] if isinstance(resp, dict) else resp.id

    def ssh_exec(self, pod_ip: str, cmd: list[str], timeout: int = 7200) -> str:
        """SSH into the pod and exec a command. Returns stdout. Raises on non-zero exit."""
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        # ponytail: 3 retries with 30s backoff — pod boot can be slow.
        for attempt in range(3):
            try:
                client.connect(pod_ip, username=self.ssh_user,
                                key_filename=self.ssh_key_path, timeout=60)
                break
            except paramiko.SSHException:
                if attempt == 2:
                    raise
                time.sleep(30)
        cmd_str = shlex.join(cmd)
        stdin, stdout, stderr = client.exec_command(cmd_str, timeout=timeout)
        exit_status = stdout.channel.recv_exit_status()
        err_text = stderr.read().decode()
        client.close()
        if exit_status != 0:
            raise RuntimeError(f"pod command failed (exit {exit_status}): {err_text[-2000:]}")
        return stdout.read().decode()

    def poll_until_done(self, pod_id: str, timeout: int = 7200,
                        interval: int = 30) -> int:
        """Poll pod status until EXITED or timeout. Returns exit code."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            pod = runpod.get_pod(pod_id)
            status = pod.get("status") if isinstance(pod, dict) else pod.status
            if status == "EXITED":
                runtime = pod.get("runtime", {}) if isinstance(pod, dict) else pod.runtime
                return int(runtime.get("exitCode", -1)) if isinstance(runtime, dict) else -1
            time.sleep(interval)
        raise TimeoutError(f"pod {pod_id} did not exit within {timeout}s")

    def download_artifact(self, pod_ip: str, remote: str, local: str) -> None:
        """SCP a file from the pod to local. Does NOT terminate the pod on failure
        (so the artifact survives for manual recovery)."""
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(pod_ip, username=self.ssh_user,
                       key_filename=self.ssh_key_path, timeout=60)
        sftp = client.open_sftp()
        try:
            sftp.get(remote, local)
        finally:
            sftp.close()
            client.close()

    def upload_artifact(self, local: str, pod_ip: str, remote: str) -> None:
        """SCP a local file to the pod. Symmetric to download_artifact.
        Does NOT terminate the pod on failure — let the exception propagate so
        the caller can recover or terminate explicitly."""
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(pod_ip, username=self.ssh_user,
                       key_filename=self.ssh_key_path, timeout=60)
        sftp = client.open_sftp()
        try:
            sftp.put(local, remote)
        finally:
            sftp.close()
            client.close()

    def terminate_pod(self, pod_id: str) -> None:
        runpod.terminate_pod(pod_id)

    def run_training(self, name: str, train_cmd: list[str],
                     artifact_remote_path: str, artifact_local_path: str,
                     poll_timeout: int = 7200,
                     pre_train_upload: list[tuple[str, str]] | None = None,
                     setup_cmds: list[str] | None = None) -> str:
        """Full lifecycle: submit → [upload] → [setup] → ssh train → download → terminate.
        If ``pre_train_upload`` is provided, each ``(local, remote)`` pair is
        SFTP-put to the pod after submit + IP resolution and before training.
        If ``setup_cmds`` is provided, each is SSH-exec'd in order after upload
        and before the train command (e.g. ``pip install kraken``).
        Returns the local artifact path."""
        pod_id = self.submit_pod(name)
        keep_pod_for_recovery = False
        try:
            # ponytail: RunPod assigns the pod an IP after boot; poll for it briefly.
            pod_ip = None
            for _ in range(60):
                pod = runpod.get_pod(pod_id)
                pod_ip = pod.get("runtime", {}).get("ip") if isinstance(pod, dict) else None
                if pod_ip:
                    break
                time.sleep(10)
            if not pod_ip:
                raise RuntimeError(f"pod {pod_id} never got an IP")
            if pre_train_upload:
                for local, remote in pre_train_upload:
                    self.upload_artifact(local, pod_ip, remote)
            self.ssh_exec(pod_ip, ["mkdir", "-p", str(Path(artifact_remote_path).parent)])
            if setup_cmds:
                for cmd_str in setup_cmds:
                    self.ssh_exec(pod_ip, shlex.split(cmd_str), timeout=600)
            self.ssh_exec(pod_ip, train_cmd, timeout=poll_timeout)
            keep_pod_for_recovery = True
            self.download_artifact(pod_ip, artifact_remote_path, artifact_local_path)
            keep_pod_for_recovery = False
            return artifact_local_path
        finally:
            if not keep_pod_for_recovery:
                self.terminate_pod(pod_id)
