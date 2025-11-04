import argparse
import statistics
from statistics import StatisticsError
from lemonade.state import State
from lemonade.tools.adapter import ModelAdapter, TokenizerAdapter
from lemonade.tools.bench import Bench
from lemonade.cache import Keys


class OgaBench(Bench):
    """
    Benchmark any model that adheres to the ModelAdapter interface.

    Required input state:
        - MODEL: model instance to benchmark.
        - TOKENIZER: tokenizer instance used to generate inputs for the model.

    Output state produced: None
    """

    unique_name = "oga-bench"

    @staticmethod
    def parser(add_help: bool = True) -> argparse.ArgumentParser:
        parser = __class__.helpful_parser(
            short_description="Benchmark an LLM in onnxruntime-genai (OGA)",
            add_help=add_help,
        )

        parser = Bench.parser(parser)

        return parser

    def get_prompt_str(self, state, token_length):
        """
        Returns a string with the prescribed token length.
        """
        tokenizer: TokenizerAdapter = state.tokenizer
        test_prompt = "word " * (token_length - 1)
        input_ids = tokenizer(test_prompt, return_tensors="pt").input_ids
        test_token_length = len(input_ids)
        delta = test_token_length - token_length
        if delta == 0:
            return test_prompt
        return "word " * max(token_length - 1 - delta, 0)

    def run_prompt(
        self,
        state: State,
        report_progress_fn,
        prompt: str,
        iterations: int,
        warmup_iterations: int,
        output_tokens: int,
    ) -> State:

        model: ModelAdapter = state.model
        tokenizer: TokenizerAdapter = state.tokenizer

        # EXPERIMENT: Tokenize outside the loop once for all iterations
        # (Current behavior - tokenization NOT included in TTFT)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        self.input_ids_len_list.append(len(input_ids))
        per_iteration_time_to_first_token = []
        per_iteration_tokens_per_second = []
        per_iteration_tokenization_overhead = []
        per_iteration_generator_overhead = []
        per_iteration_ttft_without_overhead = []

        # Don't capture time for warmup
        for count in range(warmup_iterations):
            _ = model.generate(input_ids, max_new_tokens=output_tokens)
            self.tokens_out_len_list.append(model.response_tokens)
            report_progress_fn((count + 1) / (warmup_iterations + iterations))

        for count in range(iterations):
            # EXPERIMENT: Include tokenization overhead in TTFT measurement
            # to match external code behavior
            import time
            tokenization_start = time.perf_counter()
            input_ids_retokenized = tokenizer(prompt, return_tensors="pt").input_ids
            tokenization_overhead = time.perf_counter() - tokenization_start

            _ = model.generate(
                input_ids_retokenized,
                max_new_tokens=output_tokens,
                min_new_tokens=output_tokens,
            )
            report_progress_fn(
                (warmup_iterations + count + 1) / (warmup_iterations + iterations)
            )

            self.tokens_out_len_list.append(model.response_tokens)

            # Only count an iteration if it produced enough tokens
            if model.response_tokens >= output_tokens:
                # EXPERIMENT: Report multiple TTFT measurements to understand overhead
                original_ttft = model.time_to_first_token

                # Check if the model has the experimental extended measurements
                generator_overhead = getattr(model, 'generator_creation_overhead', 0.0)

                # Calculate total overhead like external code would measure
                total_ttft_with_overhead = original_ttft + tokenization_overhead + generator_overhead

                # For now, use the version with all overhead to match external code
                per_iteration_time_to_first_token.append(total_ttft_with_overhead)
                per_iteration_tokens_per_second.append(model.tokens_per_second)

                # Collect breakdown components for saving to cache
                per_iteration_ttft_without_overhead.append(original_ttft)
                per_iteration_tokenization_overhead.append(tokenization_overhead)
                per_iteration_generator_overhead.append(generator_overhead)

                # Print breakdown for first iteration to help debugging
                if count == 0:
                    print(f"\n=== TTFT BREAKDOWN (First iteration) ===")
                    print(f"Original TTFT (lemonade default):    {original_ttft*1000:.2f} ms")
                    print(f"Tokenization overhead:                {tokenization_overhead*1000:.2f} ms")
                    print(f"Generator creation overhead:          {generator_overhead*1000:.2f} ms")
                    print(f"TOTAL TTFT (with all overhead):      {total_ttft_with_overhead*1000:.2f} ms")
                    print(f"=====================================\n")

        if not per_iteration_time_to_first_token or not per_iteration_tokens_per_second:
            raise Bench.not_enough_tokens(output_tokens)

        mean_time_to_first_token = statistics.mean(per_iteration_time_to_first_token)
        self.mean_time_to_first_token_list.append(mean_time_to_first_token)
        self.prefill_tokens_per_second_list.append(
            len(input_ids) / mean_time_to_first_token
        )
        self.token_generation_tokens_per_second_list.append(
            statistics.mean(per_iteration_tokens_per_second)
        )

        # Calculate and store breakdown metrics
        if not hasattr(self, 'ttft_without_overhead_list'):
            self.ttft_without_overhead_list = []
        if not hasattr(self, 'std_dev_ttft_without_overhead_list'):
            self.std_dev_ttft_without_overhead_list = []
        if not hasattr(self, 'tokenization_overhead_list'):
            self.tokenization_overhead_list = []
        if not hasattr(self, 'std_dev_tokenization_overhead_list'):
            self.std_dev_tokenization_overhead_list = []
        if not hasattr(self, 'generator_overhead_list'):
            self.generator_overhead_list = []
        if not hasattr(self, 'std_dev_generator_overhead_list'):
            self.std_dev_generator_overhead_list = []

        self.ttft_without_overhead_list.append(
            statistics.mean(per_iteration_ttft_without_overhead)
        )
        self.tokenization_overhead_list.append(
            statistics.mean(per_iteration_tokenization_overhead)
        )
        self.generator_overhead_list.append(
            statistics.mean(per_iteration_generator_overhead)
        )

        # Calculate standard deviations for overhead metrics
        try:
            self.std_dev_ttft_without_overhead_list.append(
                statistics.stdev(per_iteration_ttft_without_overhead)
            )
        except StatisticsError:
            # Less than 2 measurements
            self.std_dev_ttft_without_overhead_list.append(None)
        try:
            self.std_dev_tokenization_overhead_list.append(
                statistics.stdev(per_iteration_tokenization_overhead)
            )
        except StatisticsError:
            # Less than 2 measurements
            self.std_dev_tokenization_overhead_list.append(None)
        try:
            self.std_dev_generator_overhead_list.append(
                statistics.stdev(per_iteration_generator_overhead)
            )
        except StatisticsError:
            # Less than 2 measurements
            self.std_dev_generator_overhead_list.append(None)
        try:
            self.std_dev_time_to_first_token_list.append(
                statistics.stdev(per_iteration_time_to_first_token)
            )
        except StatisticsError:
            # Less than 2 measurements
            self.std_dev_time_to_first_token_list.append(None)
        try:
            self.std_dev_token_generation_tokens_per_second_list.append(
                statistics.stdev(per_iteration_tokens_per_second)
            )
        except StatisticsError:
            # Less than 2 measurements
            self.std_dev_token_generation_tokens_per_second_list.append(None)

    def save_stats(self, state):
        # Call parent class to save standard stats
        super().save_stats(state)

        # Save our additional breakdown metrics
        if hasattr(self, 'ttft_without_overhead_list') and self.ttft_without_overhead_list:
            state.save_stat(
                Keys.TTFT_WITHOUT_OVERHEAD,
                self.get_item_or_list(self.ttft_without_overhead_list),
            )
        if hasattr(self, 'std_dev_ttft_without_overhead_list') and self.std_dev_ttft_without_overhead_list:
            if not all(element is None for element in self.std_dev_ttft_without_overhead_list):
                state.save_stat(
                    Keys.STD_DEV_TTFT_WITHOUT_OVERHEAD,
                    self.get_item_or_list(self.std_dev_ttft_without_overhead_list),
                )
        if hasattr(self, 'tokenization_overhead_list') and self.tokenization_overhead_list:
            state.save_stat(
                Keys.TOKENIZATION_OVERHEAD,
                self.get_item_or_list(self.tokenization_overhead_list),
            )
        if hasattr(self, 'std_dev_tokenization_overhead_list') and self.std_dev_tokenization_overhead_list:
            if not all(element is None for element in self.std_dev_tokenization_overhead_list):
                state.save_stat(
                    Keys.STD_DEV_TOKENIZATION_OVERHEAD,
                    self.get_item_or_list(self.std_dev_tokenization_overhead_list),
                )
        if hasattr(self, 'generator_overhead_list') and self.generator_overhead_list:
            state.save_stat(
                Keys.GENERATOR_CREATION_OVERHEAD,
                self.get_item_or_list(self.generator_overhead_list),
            )
        if hasattr(self, 'std_dev_generator_overhead_list') and self.std_dev_generator_overhead_list:
            if not all(element is None for element in self.std_dev_generator_overhead_list):
                state.save_stat(
                    Keys.STD_DEV_GENERATOR_CREATION_OVERHEAD,
                    self.get_item_or_list(self.std_dev_generator_overhead_list),
                )


# This file was originally licensed under Apache 2.0. It has been modified.
# Modifications Copyright (c) 2025 AMD
