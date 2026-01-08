import argparse
import os
import lemonade.common.printing as printing
import lemonade.common.status as status
from lemonade.state import State
from lemonade.tools import FirstTool
from lemonade.cache import Keys
from lemonade.tools.trtllm.utils import (
    TensorRTLLMAdapter,
    DockerManager,
    check_nvidia_gpu,
    DEFAULT_TRTLLM_IMAGE,
)


class LoadTensorRTLLM(FirstTool):
    unique_name = "trtllm-load"

    def __init__(self):
        super().__init__(monitor_message="Loading TensorRT-LLM model")

        self.status_stats = [
            Keys.DEVICE,
            Keys.CHECKPOINT,
        ]

    @staticmethod
    def parser(add_help: bool = True) -> argparse.ArgumentParser:
        parser = __class__.helpful_parser(
            short_description="Load models with TensorRT-LLM in Docker",
            add_help=add_help,
        )

        parser.add_argument("-d", "--device", choices=["cuda"], default="cuda")

        parser.add_argument(
            "--max-seq-len",
            required=False,
            type=int,
            default=4096,
            help="Maximum sequence length (default: 4096)",
        )

        parser.add_argument(
            "--max-num-tokens",
            required=False,
            type=int,
            default=8192,
            help="Maximum total tokens in a batch (default: 8192)",
        )

        parser.add_argument(
            "--output-tokens",
            required=False,
            type=int,
            default=1024,
            help=f"Maximum number of output tokens to generate (default: 1024)",
        )

        parser.add_argument(
            "--docker-image",
            type=str,
            default=DEFAULT_TRTLLM_IMAGE,
            help=f"Docker image to use for TensorRT-LLM (default: {DEFAULT_TRTLLM_IMAGE})",
        )

        parser.add_argument(
            "--container-name",
            type=str,
            default=None,
            help="Name for the Docker container (default: auto-generated)",
        )

        parser.add_argument(
            "--trust-remote-code",
            action="store_true",
            help="Trust remote code when loading model",
        )

        return parser

    def run(
        self,
        state: State,
        input: str = "",
        device: str = "cuda",
        max_seq_len: int = 4096,
        max_num_tokens: int = 8192,
        output_tokens: int = 512,
        docker_image: str = DEFAULT_TRTLLM_IMAGE,
        container_name: str = None,
        trust_remote_code: bool = False,
    ) -> State:

        # Validate NVIDIA GPU is available
        if not check_nvidia_gpu():
            raise RuntimeError(
                "NVIDIA GPU not detected. TensorRT-LLM requires an NVIDIA GPU with nvidia-smi available."
            )

        # Set checkpoint
        checkpoint = input
        state.checkpoint = checkpoint
        state.save_stat(Keys.CHECKPOINT, checkpoint)

        # Determine if input is a directory or HF checkpoint
        if os.path.isdir(input):
            model_path = os.path.abspath(input)
            printing.log_info(f"Using local model directory: {model_path}")
        else:
            # Hugging Face checkpoint
            model_path = input
            printing.log_info(f"Using Hugging Face checkpoint: {model_path}")

        # Initialize Docker manager
        docker_manager = DockerManager(
            image=docker_image, container_name=container_name
        )

        # Prepare volume mapping - map current directory and model directory if local
        cwd = os.getcwd()
        volume_mappings = [f"{cwd}:/workspace/lemonade"]

        # If model is a local directory, also mount it
        if os.path.isdir(input):
            model_dir = os.path.dirname(model_path) or model_path
            volume_mappings.append(f"{model_dir}:/workspace/model")

        primary_volume = volume_mappings[0]

        # Ensure container is running
        printing.log_info("Setting up Docker container for TensorRT-LLM...")
        if not docker_manager.ensure_container_running(volume_mapping=primary_volume):
            raise RuntimeError("Failed to start Docker container for TensorRT-LLM")

        # Store Docker manager in state for use by bench tool
        state.docker_manager = docker_manager

        # Create TensorRT-LLM adapter
        state.model = TensorRTLLMAdapter(
            model=model_path,
            device=device,
            output_tokens=output_tokens,
            max_seq_len=max_seq_len,
            max_num_tokens=max_num_tokens,
            docker_manager=docker_manager,
            state=state,
        )

        state.device = device

        # Save initial stats
        state.save_stat(Keys.DEVICE, device)
        state.save_stat("docker_image", docker_image)
        state.save_stat("container_name", docker_manager.container_name)
        state.save_stat("max_seq_len", max_seq_len)
        state.save_stat("max_num_tokens", max_num_tokens)

        status.add_to_state(
            state=state,
            name=input,
            model=(
                os.path.basename(model_path)
                if os.path.isdir(model_path)
                else model_path
            ),
            extension="",
        )

        printing.log_success(
            f"TensorRT-LLM model loaded in container: {docker_manager.container_name}"
        )

        return state


# Copyright (c) 2025 AMD
