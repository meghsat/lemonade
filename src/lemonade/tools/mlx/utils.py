
import os
import re
import sys
from io import StringIO

# Disable tokenizers parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
from mlx_lm import load, generate


class MLXAdapter:

    def __init__(self, model_path: str, device: str, state):
        self.model_path = model_path
        self.device = device
        self.state = state

        self.model, self.tokenizer = load(
            model_path,
            tokenizer_config={"trust_remote_code": True}
        )

        self.prompt_tokens = 0
        self.response_tokens = 0
        self.time_to_first_token = 0
        self.tokens_per_second = 0
        self.prompt_tokens_per_second = 0
        self.peak_memory_gb = 0

    def generate(self, prompt: str, max_new_tokens: int = 100, verbose: bool = True):
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_new_tokens,
                verbose=verbose,
            )

            # Restore stdout
            sys.stdout = old_stdout
            output = captured_output.getvalue()

            # Parse metrics from MLX verbose output
            self._parse_metrics(output)

            return response

        except Exception as e:
            sys.stdout = old_stdout
            raise e

    def _parse_metrics(self, output: str):
        """
        Parse metrics from MLX verbose output

        Args:
            output: Captured verbose output from MLX generate
        """
        # Parse prompt tokens per second
        prompt_tps_match = re.search(r'Prompt:.*?([0-9.]+) tokens-per-sec', output)
        if prompt_tps_match:
            self.prompt_tokens_per_second = float(prompt_tps_match.group(1))

        # Parse time to first token
        ttft_match = re.search(r'Time to first token:\s*([0-9.]+)\s*sec', output)
        if ttft_match:
            self.time_to_first_token = float(ttft_match.group(1))

        # Parse generation tokens per second
        gen_tps_match = re.search(r'Generation:.*?([0-9.]+) tokens-per-sec', output)
        if gen_tps_match:
            self.tokens_per_second = float(gen_tps_match.group(1))

        # Parse peak memory
        peak_mem_match = re.search(r'Peak memory:\s*([0-9.]+)\s*GB', output)
        if peak_mem_match:
            self.peak_memory_gb = float(peak_mem_match.group(1))

        # Estimate token counts from the metrics
        # If we have TTFT and prompt TPS, we can estimate prompt tokens
        if self.time_to_first_token > 0 and self.prompt_tokens_per_second > 0:
            self.prompt_tokens = int(self.time_to_first_token * self.prompt_tokens_per_second)


# Copyright (c) 2025 AMD
