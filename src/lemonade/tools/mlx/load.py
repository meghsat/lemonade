
import argparse
import os
import platform
import lemonade.common.printing as printing
import lemonade.common.status as status
from lemonade.state import State
from lemonade.tools import FirstTool
from lemonade.cache import Keys
from lemonade.tools.mlx.utils import MLXAdapter


class LoadMLX(FirstTool):
    unique_name = "mlx-load"

    def __init__(self):
        super().__init__(monitor_message="Loading MLX model")

        self.status_stats = [
            Keys.DEVICE,
            Keys.CHECKPOINT,
        ]

    @staticmethod
    def parser(add_help: bool = True) -> argparse.ArgumentParser:
        parser = __class__.helpful_parser(
            short_description="Load models with MLX on Apple Silicon",
            add_help=add_help,
        )

        parser.add_argument(
            "-d",
            "--device",
            choices=["gpu", "cpu"],
            default="gpu",
            help="Device to run on (default: gpu)",
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
        device: str = "gpu",
        trust_remote_code: bool = False,
    ) -> State:

        checkpoint = input
        state.checkpoint = checkpoint
        state.save_stat(Keys.CHECKPOINT, checkpoint)

        if os.path.isdir(input):
            model_path = os.path.abspath(input)
            printing.log_info(f"Using local model directory: {model_path}")
        else:
            model_path = input
            printing.log_info(f"Using Hugging Face checkpoint: {model_path}")

        printing.log_info(f"Loading model with MLX on {device}...")
        state.model = MLXAdapter(
            model_path=model_path,
            device=device,
            state=state,
        )

        state.device = device
        state.save_stat(Keys.DEVICE, device)

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

        printing.log_success(f"MLX model loaded successfully: {model_path}")

        return state


# Copyright (c) 2025 AMD
