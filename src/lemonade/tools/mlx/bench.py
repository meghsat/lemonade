
import argparse
import statistics
from statistics import StatisticsError
from typing import List

from lemonade.state import State
from lemonade.tools.tool import Tool
from lemonade.tools.bench import (
    Bench,
    default_iterations,
    default_output_tokens,
    default_warmup_runs,
)
from lemonade.tools.mlx.utils import MLXAdapter
import lemonade.common.printing as printing


class MLXBench(Bench):
    unique_name = "mlx-bench"

    def __init__(self, monitor_message="Benchmarking MLX"):
        super().__init__(monitor_message)

        self.peak_gpu_power_list = []
        self.avg_gpu_power_list = []
        self.peak_cpu_power_list = []
        self.avg_cpu_power_list = []
        self.peak_ane_power_list = []
        self.avg_ane_power_list = []
        self.peak_combined_power_list = []
        self.avg_combined_power_list = []
        self.power_plot_list = []

        self.prompt_labels = []

    @staticmethod
    def parser(add_help: bool = True) -> argparse.ArgumentParser:
        parser = __class__.helpful_parser(
            short_description="Benchmark an LLM with MLX on Apple Silicon",
            add_help=add_help,
        )

        parser = Bench.parser(parser)

        parser.add_argument(
            "--prompt-label",
            type=str,
            action="append",
            default=None,
            help="Optional label for each prompt (e.g., filename) to display in results and plots. "
            "Can be specified multiple times, one per prompt.",
        )

        return parser

    def parse(self, state: State, args, known_only=True) -> argparse.Namespace:
        """
        Helper function to parse CLI arguments into the args expected by run()
        """

        # Call Tool parse method, NOT the Bench parse method
        parsed_args = Tool.parse(self, state, args, known_only)

        prompt_labels = getattr(parsed_args, 'prompt_label', None)
        parsed_args = super().parse(state, args, known_only)

        parsed_args.prompt_label = prompt_labels if prompt_labels else None

        return parsed_args

    def run(
        self,
        state: State,
        prompts: List[str] = None,
        iterations: int = default_iterations,
        warmup_iterations: int = default_warmup_runs,
        output_tokens: int = default_output_tokens,
        gpu_cooldown: int = 5,
        prompt_label: List[str] = None,
        **kwargs,
    ) -> State:
        """
        Run MLX benchmark

        This method validates setup and stores parameters, then calls parent run()
        which will call run_prompt() for each prompt.

        Args:
            state: Lemonade state object
            prompts: List of input prompts (file paths or text)
            iterations: Number of benchmark iterations per prompt
            warmup_iterations: Number of warmup iterations
            output_tokens: Number of tokens to generate
            gpu_cooldown: Cooldown time between prompts (seconds)
            prompt_label: Optional labels for each prompt
            kwargs: Additional parameters
        """

        if not isinstance(state.model, MLXAdapter):
            raise ValueError(
                "MLX model not loaded. Please run mlx-load first."
            )

        # Store benchmark-specific parameters in instance for use by run_prompt()
        self.prompt_labels = prompt_label if prompt_label else []

        printing.log_info(f"Running MLX benchmark with {len(prompts)} prompts")
        printing.log_info(f"Iterations: {iterations}, Warmup: {warmup_iterations}")

        # Call parent run() which will iterate through prompts and call run_prompt()
        return super().run(
            state=state,
            prompts=prompts,
            iterations=iterations,
            warmup_iterations=warmup_iterations,
            output_tokens=output_tokens,
            gpu_cooldown=gpu_cooldown,
        )

    def run_prompt(
        self,
        state: State,
        report_progress_fn,
        prompt: str,
        iterations: int,
        warmup_iterations: int,
        output_tokens: int,
        **kwargs,
    ):
        """
        Run benchmark for a single prompt.

        This method is called by the parent Bench.run() for each prompt.

        Args:
            state: Lemonade state object
            report_progress_fn: Progress callback function
            prompt: Input prompt text or file path
            iterations: Number of iterations
            warmup_iterations: Number of warmup iterations
            output_tokens: Number of tokens to generate
        """

        if self.first_run_prompt:
            if not hasattr(state, "model") or not isinstance(
                state.model, MLXAdapter
            ):
                raise Exception(
                    f"{self.__class__.unique_name} requires an MLXAdapter model to be "
                    "loaded first. Please run mlx-load before this tool."
                )

        model: MLXAdapter = state.model

        if prompt.endswith('.txt') and '/' in prompt:
            try:
                with open(prompt, 'r') as f:
                    prompt_text = f.read().strip()
            except Exception as e:
                raise Exception(f"Failed to read prompt file {prompt}: {e}")
        else:
            prompt_text = prompt

        # Run warmup iterations before actual benchmarking
        if warmup_iterations > 0:
            model.run_warmup(
                prompt=prompt_text,
                max_new_tokens=output_tokens,
                num_warmup=warmup_iterations
            )

        per_iteration_tokens_per_second = []
        per_iteration_time_to_first_token = []
        per_iteration_prompt_tokens_per_second = []
        per_iteration_peak_memory = []

        for iteration in range(iterations):
            try:
                _ = model.generate(
                    prompt=prompt_text,
                    max_new_tokens=output_tokens,
                    verbose=True,
                )

                if model.time_to_first_token is None or model.tokens_per_second is None:
                    error_msg = (
                        "Could not find timing information in MLX output.\n"
                        "This may indicate MLX is not installed or not working correctly."
                    )
                    raise Exception(error_msg)

                per_iteration_tokens_per_second.append(model.tokens_per_second)
                per_iteration_time_to_first_token.append(model.time_to_first_token)
                per_iteration_prompt_tokens_per_second.append(model.prompt_tokens_per_second)
                per_iteration_peak_memory.append(model.peak_memory_gb)

                report_progress_fn((iteration + 1) / iterations)

                # Clear MLX Metal cache after each iteration to ensure fresh state
                import gc
                import mlx.core as mx
                mx.clear_cache()
                gc.collect()

            except Exception as e:
                error_msg = f"Failed to run benchmark: {str(e)}"
                raise Exception(error_msg)

        self.input_ids_len_list.append(model.prompt_tokens)
        self.tokens_out_len_list.append(output_tokens)

        mean_time_to_first_token = statistics.mean(per_iteration_time_to_first_token)
        self.mean_time_to_first_token_list.append(mean_time_to_first_token)

        if mean_time_to_first_token == 0:
            error_msg = (
                f"Cannot calculate prefill tokens per second: mean_time_to_first_token is zero.\n"
                f"This indicates the model failed to generate output properly.\n"
                f"Prompt tokens: {model.prompt_tokens}\n"
                f"Time to first token measurements: {per_iteration_time_to_first_token}\n"
                f"Iterations completed: {len(per_iteration_time_to_first_token)}"
            )
            raise Exception(error_msg)

        # Store prefill and generation metrics
        self.prefill_tokens_per_second_list.append(
            statistics.mean(per_iteration_prompt_tokens_per_second)
        )
        self.token_generation_tokens_per_second_list.append(
            statistics.mean(per_iteration_tokens_per_second)
        )

        # Store standard deviations
        try:
            self.std_dev_time_to_first_token_list.append(
                statistics.stdev(per_iteration_time_to_first_token)
            )
        except StatisticsError:
            self.std_dev_time_to_first_token_list.append(None)

        try:
            self.std_dev_token_generation_tokens_per_second_list.append(
                statistics.stdev(per_iteration_tokens_per_second)
            )
        except StatisticsError:
            self.std_dev_token_generation_tokens_per_second_list.append(None)

        try:
            self.std_dev_prefill_tokens_per_second_list.append(
                statistics.stdev(per_iteration_prompt_tokens_per_second)
            )
        except StatisticsError:
            self.std_dev_prefill_tokens_per_second_list.append(None)

        # Store memory usage
        if self.save_max_memory_used and per_iteration_peak_memory:
            mean_memory = statistics.mean(per_iteration_peak_memory)
            self.max_memory_used_gb_list.append(mean_memory)

    def display_prompt_results(self, state, prompt, prompt_index):
        """Display results for a single prompt immediately after benchmarking"""
        import sys
        import csv
        import os

        # Get the real terminal stdout (not redirected by Logger)
        output = sys.stdout
        if hasattr(sys.stdout, 'terminal'):
            output = sys.stdout.terminal

        # Get prompt label
        if len(self.prompt_labels) > prompt_index:
            prompt_label = self.prompt_labels[prompt_index]
        else:
            # If prompt is a file path, use the filename
            if '/' in prompt:
                prompt_label = os.path.basename(prompt)
            else:
                prompt_label = f"prompt_{prompt_index + 1}"

        # Get actual token count from benchmark results
        actual_token_count = self.input_ids_len_list[prompt_index] if len(self.input_ids_len_list) > prompt_index else 0

        # Print newline and separator directly to terminal
        output.write(f"\n{'='*80}\n")
        output.write(f"Results for Prompt: {prompt_label} ({actual_token_count} tokens) (Prompt {prompt_index + 1})\n")
        output.write(f"{'='*80}\n")
        output.flush()

        # Get the current prompt's results
        idx = prompt_index

        results = {
            "Prompt Tokens": self.input_ids_len_list[idx],
            "Response Tokens": self.tokens_out_len_list[idx],
            "Seconds To First Token": f"{self.mean_time_to_first_token_list[idx]:.3f}",
            "Token Generation Tokens Per Second": f"{self.token_generation_tokens_per_second_list[idx]:.3f}",
            "Prefill Tokens Per Second": f"{self.prefill_tokens_per_second_list[idx]:.3f}",
        }

        # Add std dev metrics if available
        if len(self.std_dev_time_to_first_token_list) > idx and self.std_dev_time_to_first_token_list[idx] is not None:
            results["Std Dev Seconds To First Token"] = f"{self.std_dev_time_to_first_token_list[idx]:.3f}"

        if len(self.std_dev_token_generation_tokens_per_second_list) > idx and self.std_dev_token_generation_tokens_per_second_list[idx] is not None:
            results["Std Dev Tokens Per Second"] = f"{self.std_dev_token_generation_tokens_per_second_list[idx]:.3f}"

        if len(self.std_dev_prefill_tokens_per_second_list) > idx and self.std_dev_prefill_tokens_per_second_list[idx] is not None:
            results["Std Dev Prefill Tokens Per Second"] = f"{self.std_dev_prefill_tokens_per_second_list[idx]:.3f}"

        if self.save_max_memory_used and len(self.max_memory_used_gb_list) > idx:
            if self.max_memory_used_gb_list[idx] is not None:
                results["Memory Usage (GB)"] = f"{self.max_memory_used_gb_list[idx]:.3f}"

        # Display Apple power metrics from per-prompt lists
        if len(self.peak_gpu_power_list) > idx and self.peak_gpu_power_list[idx]:
            results["Peak GPU Power"] = self.peak_gpu_power_list[idx]
            results["Avg GPU Power"] = self.avg_gpu_power_list[idx]

        if len(self.peak_cpu_power_list) > idx and self.peak_cpu_power_list[idx]:
            results["Peak CPU Power"] = self.peak_cpu_power_list[idx]
            results["Avg CPU Power"] = self.avg_cpu_power_list[idx]

        if len(self.peak_ane_power_list) > idx and self.peak_ane_power_list[idx] is not None:
            results["Peak ANE Power"] = self.peak_ane_power_list[idx]
            results["Avg ANE Power"] = self.avg_ane_power_list[idx]

        if len(self.peak_combined_power_list) > idx and self.peak_combined_power_list[idx] is not None:
            results["Peak Combined Power"] = self.peak_combined_power_list[idx]
            results["Avg Combined Power"] = self.avg_combined_power_list[idx]

        # Add plot path if available
        if len(self.power_plot_list) > idx and self.power_plot_list[idx]:
            results["Power Usage Plot"] = self.power_plot_list[idx]

        for key, value in results.items():
            output.write(f"  {key}: {value}\n")

        output.write(f"{'='*80}\n\n")
        output.flush()

        # CSV export: Append results to CSV file
        self._append_to_csv(state, prompt_label, results)

    def _append_to_csv(self, state, prompt_label, results):
        """Append benchmark results to a CSV file in the cache directory"""
        import csv
        import os

        csv_filename = "benchmark_results.csv"
        csv_path = os.path.join(state.cache_dir, csv_filename)

        csv_row = {"Prompt File": prompt_label}

        for key, value in results.items():
            if isinstance(value, str) and key != "Power Usage Plot":
                try:
                    csv_row[key] = float(value)
                except ValueError:
                    csv_row[key] = value
            else:
                csv_row[key] = value

        file_exists = os.path.exists(csv_path)

        fieldnames = ["Prompt File"] + [k for k in results.keys()]

        try:
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                if not file_exists:
                    writer.writeheader()

                writer.writerow(csv_row)

            if not file_exists:
                printing.log_info(f"Created CSV file: {csv_path}")
            else:
                printing.log_info(f"Appended results to: {csv_path}")

        except Exception as e:
            printing.log_warning(f"Failed to write to CSV: {e}")

    def _store_power_metrics(self, state):
        """Store per-prompt power metrics from filesystem into lists"""
        from lemonade.profilers.apple_power import Keys as AppleKeys
        import lemonade.common.filesystem as fs

        # Load stats from the YAML file (not from state.stats attribute)
        stats_obj = fs.Stats(state.cache_dir, state.build_name)
        stats = stats_obj.stats  # This loads from lemonade_stats.yaml

        if stats:
            # Check for Apple power metrics
            if AppleKeys.PEAK_GPU_POWER in stats:
                self.peak_gpu_power_list.append(stats[AppleKeys.PEAK_GPU_POWER])
                self.avg_gpu_power_list.append(stats[AppleKeys.AVG_GPU_POWER])
                self.peak_cpu_power_list.append(stats[AppleKeys.PEAK_CPU_POWER])
                self.avg_cpu_power_list.append(stats[AppleKeys.AVG_CPU_POWER])

                # ANE power may not always be available
                if 'peak_ane_power_apple' in stats:
                    self.peak_ane_power_list.append(stats.get('peak_ane_power_apple', 0))
                    self.avg_ane_power_list.append(stats.get('avg_ane_power_apple', 0))
                else:
                    self.peak_ane_power_list.append(None)
                    self.avg_ane_power_list.append(None)

                # Combined power
                if 'peak_combined_power_apple' in stats:
                    self.peak_combined_power_list.append(stats.get('peak_combined_power_apple', 0))
                    self.avg_combined_power_list.append(stats.get('avg_combined_power_apple', 0))
                else:
                    self.peak_combined_power_list.append(None)
                    self.avg_combined_power_list.append(None)

                if AppleKeys.POWER_USAGE_PLOT in stats:
                    self.power_plot_list.append(stats[AppleKeys.POWER_USAGE_PLOT])


# Copyright (c) 2025 AMD
