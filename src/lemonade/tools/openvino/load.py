# openvino_load.py
import argparse
import os

# import time
import json

# import shutil
# import logging
from lemonade.cache import Keys
from lemonade.state import State
from lemonade.tools import FirstTool
import lemonade.common.status as status
import lemonade.common.printing as printing
from lemonade.common.network import get_base_model, is_offline

# from lemonade.tools.adapter import (
#     ModelAdapter,
#     TokenizerAdapter,
#     PassthroughTokenizerResult,
# )
from lemonade.cache import Keys
import openvino_genai as ov_genai


def import_error_heler(e: Exception):
    """
    Print a helpful message in the event of an import error
    """
    raise ImportError(
        f"{e}\n Please install lemonade-sdk with "
        "one of the oga extras, for example:\n"
        "pip install lemonade-sdk[dev,oga-cpu]\n"
        "See https://lemonade-server.ai/install_options.html for details"
    )


class OpenVinoLoad(FirstTool):
    """
    Load class for OpenVINO GenAI models.

    Output state produced:
        - state.model_path: instance of the model.
        - state.device: target device.
    """

    unique_name = "openvino-load"

    def __init__(self):
        super().__init__(monitor_message="openvino-load: DUMMY MESSAGE")

    @staticmethod
    def parser(add_help: bool = True) -> argparse.ArgumentParser:
        parser = __class__.helpful_parser(
            short_description="Load an LLM in PyTorch using huggingface transformers",
            add_help=add_help,
        )
        default_device = "NPU"

        parser.add_argument(
            "--device",
            "-d",
            required=False,
            default=default_device,
            help=f"Device to run model on (e.g., CPU, GPU, NPU).",
        )

        parser.add_argument(
            "--max-prompt-len",
            "-p",
            required=False,
            default=2048,
            help=f"max input prompt length for the llm pipline.",
        )

        parser.add_argument(
            "--min-response-len",
            "-r",
            required=False,
            default=128,
            help="set the minimum response length of the output",
        )

        parser.add_argument(
            "--bench-input-prompt",
            "-bp",
            required=False,
            default=None,
            help="If benchmarking on a specific prompt length, pass in the "
            "input prompt for the pipe to infer the max input token length",
        )
        return parser

    def parse(self, state: State, args, known_only=True) -> argparse.Namespace:
        parsed_args = super().parse(state, args, known_only)
        state.device = parsed_args.device
        state.max_prompt_len = parsed_args.max_prompt_len
        return parsed_args

    @staticmethod
    def _load_model_and_setup_state(
        state,
        full_model_path,
        checkpoint,
        trust_remote_code,
        device,
        max_prompt_len,
        min_response_len,
        bench_input_prompt,
    ):
        """
        Loads the OGA model from local folder and then loads the tokenizer.
        Will auto-detect if we're offline.
        """

        try:
            from lemonade.tools.openvino.utils import (
                OpenVinoModel,
                OpenVinoTokenizer,
            )
            from lemonade.common.network import is_offline
        except ImportError as e:
            import_error_heler(e)

        # Auto-detect offline mode
        offline = is_offline()

        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            import_error_heler(e)

        try:
            # Always try to use local files first
            local_files_only = True

            hf_tokenizer = AutoTokenizer.from_pretrained(
                full_model_path,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
        except ValueError as e:
            if "trust_remote_code" in str(e):
                raise ValueError(
                    "This model requires you to execute code from the repo.  Please review it "
                    "and if you trust it, then use the `--trust-remote-code` flag with oga-load."
                )

            if offline and "Can't load tokenizer for" in str(e):
                raise ValueError(
                    f"Cannot load tokenizer for {checkpoint} in offline mode. "
                    f"The tokenizer files may not be available locally in {full_model_path}."
                )
            raise

        state.tokenizer = OpenVinoTokenizer(
            full_model_path,
            hf_tokenizer,
        )

        input_prompt_len = max_prompt_len
        if bench_input_prompt and os.path.exists(bench_input_prompt):
            try:
                with open(bench_input_prompt, "r", encoding="utf-8") as file:
                    content = file.read()
                tokenizer = state.tokenizer
                encoded = tokenizer(content).input_ids
                prompt_len = encoded.input_ids.data.shape[1]
                input_prompt_len = prompt_len
            except:
                raise FileNotFoundError

        try:
            state.model = OpenVinoModel(
                full_model_path,
                device=device,
                max_prompt_len=input_prompt_len,
                min_response_len=min_response_len,
            )
        except Exception as e:
            raise

        status.add_to_state(state=state, name=checkpoint, model=checkpoint)

    def run(
        self,
        state: State,
        input: str,
        input_path: str = "",
        device: str = "NPU",
        dtype: str = "int4",
        max_prompt_len: int = 2048,
        min_response_len: int = 128,
        bench_input_prompt: str = None,
        int4_block_size: int = None,
        force: bool = False,
        download_only: bool = False,
        trust_remote_code=False,
        subfolder: str = None,
    ) -> State:
        # Auto-detect offline status
        offline = is_offline()
        if offline:
            printing.log_warning(
                "Network connectivity to huggingface.co not detected. Running in offline mode."
            )

        state.device = device
        state.dtype = dtype
        state.model = input

        # Log initial stats
        state.save_stat(Keys.DTYPE, dtype)
        state.save_stat(Keys.DEVICE, device)

        # Check if input is a local folder
        if os.path.isdir(input):
            # input is a local folder
            full_model_path = os.path.abspath(input)
            checkpoint = "local_model"
            state.checkpoint = checkpoint

            state.save_stat(Keys.CHECKPOINT, checkpoint)
            state.save_stat(Keys.LOCAL_MODEL_FOLDER, full_model_path)
        else:
            # input is a model checkpoint
            raise NotImplementedError

        if not download_only:
            self._load_model_and_setup_state(
                state,
                full_model_path,
                checkpoint,
                trust_remote_code,
                device,
                max_prompt_len,
                min_response_len,
                bench_input_prompt,
            )

        return state
