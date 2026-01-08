"""
Utilities for TensorRT-LLM tool including Docker container management
and TensorRT-LLM adapter.
"""

import subprocess
import time
import os
from pathlib import Path
from typing import Optional, Tuple
import lemonade.common.printing as printing


# Default Docker image for TensorRT-LLM
DEFAULT_TRTLLM_IMAGE = "nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev"
DEFAULT_CONTAINER_NAME_PREFIX = "lemonade_trtllm"


class DockerManager:
    """Manages Docker container lifecycle for TensorRT-LLM"""

    def __init__(self, image: str = DEFAULT_TRTLLM_IMAGE, container_name: str = None):
        self.image = image
        self.container_name = (
            container_name or f"{DEFAULT_CONTAINER_NAME_PREFIX}_{int(time.time())}"
        )

    def check_docker_available(self) -> bool:
        """Check if Docker is installed and running"""
        try:
            result = subprocess.run(
                ["docker", "version"], capture_output=True, timeout=5
            )
            return result.returncode == 0
        except Exception as e:
            printing.log_error(f"Docker check failed: {e}")
            return False

    def check_container_exists(self) -> bool:
        """Check if container with the given name already exists"""
        try:
            result = subprocess.run(
                [
                    "docker",
                    "ps",
                    "-a",
                    "--filter",
                    f"name={self.container_name}",
                    "--format",
                    "{{.Names}}",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return self.container_name in result.stdout.strip()
        except Exception as e:
            printing.log_warning(f"Failed to check container existence: {e}")
            return False

    def check_container_running(self) -> bool:
        """Check if container is currently running"""
        try:
            result = subprocess.run(
                [
                    "docker",
                    "ps",
                    "--filter",
                    f"name={self.container_name}",
                    "--format",
                    "{{.Names}}",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return self.container_name in result.stdout.strip()
        except Exception as e:
            printing.log_warning(f"Failed to check container status: {e}")
            return False

    def start_existing_container(self) -> bool:
        """Start an existing stopped container"""
        try:
            printing.log_info(f"Starting existing container: {self.container_name}")
            result = subprocess.run(
                ["docker", "start", self.container_name],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                printing.log_success(
                    f"Container {self.container_name} started successfully"
                )
                return True
            else:
                printing.log_error(f"Failed to start container: {result.stderr}")
                return False
        except Exception as e:
            printing.log_error(f"Error starting container: {e}")
            return False

    def create_and_run_container(self, volume_mapping: str = None) -> bool:
        """Create and run a new Docker container"""
        try:
            cmd = [
                "docker",
                "run",
                "--gpus",
                "all",
                "-d",  # Run in detached mode
                "--name",
                self.container_name,
            ]

            if volume_mapping:
                cmd.extend(["-v", volume_mapping])

            cmd.append(self.image)
            cmd.append("sleep")
            cmd.append("infinity")  # Keep container running

            printing.log_info(f"Creating new container: {self.container_name}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                printing.log_success(
                    f"Container {self.container_name} created successfully"
                )
                return True
            else:
                printing.log_error(f"Failed to create container: {result.stderr}")
                return False

        except Exception as e:
            printing.log_error(f"Error creating container: {e}")
            return False

    def ensure_container_running(self, volume_mapping: str = None) -> bool:
        """
        Ensure container is running. Create if doesn't exist, start if stopped.

        Args:
            volume_mapping: Volume mapping in format "host_path:container_path"

        Returns:
            True if container is running, False otherwise
        """
        if not self.check_docker_available():
            printing.log_error("Docker is not available")
            return False

        if self.check_container_running():
            printing.log_info(f"Container {self.container_name} is already running")
            return True

        if self.check_container_exists():
            return self.start_existing_container()
        else:
            return self.create_and_run_container(volume_mapping)

    def exec_command(
        self, command: list, workdir: str = None
    ) -> subprocess.CompletedProcess:
        """
        Execute a command inside the container

        Args:
            command: Command to execute as a list
            workdir: Working directory inside container

        Returns:
            CompletedProcess object
        """
        cmd = ["docker", "exec"]

        if workdir:
            cmd.extend(["-w", workdir])

        cmd.append(self.container_name)
        cmd.extend(command)

        return subprocess.run(cmd, capture_output=True, text=True)

    def stop_container(self):
        try:
            printing.log_info(f"Stopping container: {self.container_name}")
            subprocess.run(["docker", "stop", self.container_name], timeout=30)
        except Exception as e:
            printing.log_warning(f"Error stopping container: {e}")

    def remove_container(self):
        try:
            printing.log_info(f"Removing container: {self.container_name}")
            subprocess.run(["docker", "rm", self.container_name], timeout=30)
        except Exception as e:
            printing.log_warning(f"Error removing container: {e}")


class TensorRTLLMAdapter:
    """
    Adapter for TensorRT-LLM model that provides a consistent interface
    similar to other lemonade adapters (LlamaCppAdapter, etc.)
    """

    def __init__(
        self,
        model: str,
        device: str = "cuda",
        output_tokens: int = 512,
        max_seq_len: int = 4096,
        max_num_tokens: int = 8192,
        docker_manager: DockerManager = None,
        state=None,
    ):
        self.model = model
        self.device = device
        self.output_tokens = output_tokens
        self.max_seq_len = max_seq_len
        self.max_num_tokens = max_num_tokens
        self.docker_manager = docker_manager
        self.state = state

    def generate(self, prompt: str, max_tokens: int = None):
        """
        Note: This is a placeholder. Actual generation should happen
        through the Docker execution.
        """
        raise NotImplementedError(
            "Direct generation through adapter is not supported. "
            "Use trtllm-bench/Docker tool for benchmarking."
        )


def check_nvidia_gpu() -> bool:
    """Check if NVIDIA GPU is available"""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False


def get_container_path(host_path: str, container_base: str = "/workspace") -> str:
    """
    Convert a host path to a container path

    Args:
        host_path: Path on host system
        container_base: Base path in container where volumes are mounted

    Returns:
        Path to use inside container
    """
    # For simplicity, map to container_base/<basename>
    basename = os.path.basename(host_path.rstrip(os.sep))
    return f"{container_base}/{basename}"


# Copyright (c) 2025 AMD
