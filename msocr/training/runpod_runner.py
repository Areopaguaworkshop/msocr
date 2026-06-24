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

    # ponytail: RunPod exposes SSH on a random public port; runtime.ports[].ip
    # and publicPort hold host:port once the container has booted. runtime is
    # None until the container starts; on Community Cloud that can take 60-180s.
    def _ssh_endpoint(self, pod_id: str, deadline_s: int = 600) -> tuple[str, int]:
        """Poll get_pod until runtime.ports has a public entry for port 22.

        Returns (host, port).
        """
        deadline = time.time() + deadline_s
        while time.time() < deadline:
            pod = runpod.get_pod(pod_id)
            rt = pod.get("runtime") if isinstance(pod, dict) else None
            ports = (rt.get("ports") or []) if isinstance(rt, dict) else []
            for p in ports:
                if not isinstance(p, dict):
                    continue
                if p.get("privatePort") == 22 and p.get("isIpPublic"):
                    return (p["ip"], int(p["publicPort"]))
            time.sleep(10)
        raise RuntimeError(f"pod {pod_id} never exposed a public SSH port")

    def ssh_exec(self, pod_hostport: tuple[str, int], cmd: list[str],
                 timeout: int = 7200) -> str:
        """SSH into the pod and exec a command. Returns stdout. Raises on non-zero exit."""
        host, port = pod_hostport
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        # ponytail: 3 retries with 30s backoff — pod boot can be slow.
        for attempt in range(3):
            try:
                client.connect(host, port=port, username=self.ssh_user,
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

    def download_artifact(self, pod_hostport: tuple[str, int], remote: str, local: str) -> None:
        """SCP a file from the pod to local. Does NOT terminate the pod on failure
        (so the artifact survives for manual recovery)."""
        host, port = pod_hostport
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(host, port=port, username=self.ssh_user,
                        key_filename=self.ssh_key_path, timeout=60)
        sftp = client.open_sftp()
        try:
            sftp.get(remote, local)
        finally:
            sftp.close()
            client.close()

    def upload_artifact(self, local: str, pod_hostport: tuple[str, int], remote: str) -> None:
        """SCP a local file to the pod. Symmetric to download_artifact.
        Does NOT terminate the pod on failure — let the exception propagate so
        the caller can recover or terminate explicitly."""
        host, port = pod_hostport
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(host, port=port, username=self.ssh_user,
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
                     artifact_remote_dir: str, artifact_local_path: str,
                     poll_timeout: int = 7200,
                     pre_train_upload: list[tuple[str, str]] | None = None,
                     setup_cmds: list[str] | None = None) -> str:
        """Full lifecycle: submit → [upload] → [setup] → ssh train → glob → download → terminate.
        If ``pre_train_upload`` is provided, each ``(local, remote)`` pair is
        SFTP'd to the pod before ``setup_cmds`` run.
        If ``setup_cmds`` is provided, each is run via SSH before training.
        Ketos 7.0 writes ``best_{score:.4f}.safetensors`` into ``artifact_remote_dir``;
        the score suffix is unknown until training ends, so we glob the dir
        and pick the (alphabetically) last ``best_*.safetensors`` — highest score.
        Returns the local artifact path."""
        stage = "creating RunPod pod"
        pod_id = self.submit_pod(name)
        keep_pod_for_recovery = False
        try:
            stage = "waiting for pod SSH endpoint"
            pod_hostport = self._ssh_endpoint(pod_id)
            if pre_train_upload:
                for local, remote in pre_train_upload:
                    stage = f"uploading {local} to {remote}"
                    self.upload_artifact(local, pod_hostport, remote)
            stage = f"creating remote artifact directory {artifact_remote_dir}"
            self.ssh_exec(pod_hostport, ["mkdir", "-p", artifact_remote_dir])
            if setup_cmds:
                for cmd_str in setup_cmds:
                    stage = f"running setup command: {cmd_str[:120]}"
                    self.ssh_exec(pod_hostport, shlex.split(cmd_str), timeout=600)
            stage = "running remote training command"
            self.ssh_exec(pod_hostport, train_cmd, timeout=poll_timeout)
            keep_pod_for_recovery = True
            # ponytail: glob best_*.safetensors, sort desc, take first.
            # ls sorts ascending so `best_0.9` < `best_0.95`; tail -1 = highest score.
            stage = f"locating best_*.safetensors under {artifact_remote_dir}"
            listing = self.ssh_exec(pod_hostport,
                ["sh", "-c", f"ls -1 {artifact_remote_dir}/best_*.safetensors 2>/dev/null | sort | tail -1"])
            remote_best = listing.strip()
            if not remote_best:
                raise FileNotFoundError(
                    f"no best_*.safetensors found under {artifact_remote_dir}; "
                    f"training may have failed or written to a different path")
            stage = f"downloading {remote_best} to {artifact_local_path}"
            self.download_artifact(pod_hostport, remote_best, artifact_local_path)
            keep_pod_for_recovery = False
            return artifact_local_path
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"{stage}: {exc}") from exc
        finally:
            if not keep_pod_for_recovery:
                self.terminate_pod(pod_id)
