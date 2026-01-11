"""
TensorRT-LLM benchmarking tool

This tool runs benchmarks on TensorRT-LLM models inside a Docker container.
It integrates with lemonade's profiler system for power monitoring.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from statistics import StatisticsError

from lemonade.state import State
from lemonade.tools.tool import Tool
from lemonade.tools.bench import (
    Bench,
    default_iterations,
    default_output_tokens,
    default_warmup_runs,
)
from lemonade.cache import Keys
from lemonade.tools.trtllm.utils import TensorRTLLMAdapter, DockerManager
import lemonade.common.printing as printing


class TensorRTLLMBench(Bench):
    """
    Benchmark a TensorRT-LLM model running in a Docker container
    """

    unique_name = "trtllm-bench"

    def __init__(self, monitor_message="Benchmarking TensorRT-LLM"):
        super().__init__(monitor_message)

        # Per-prompt power metrics lists (these will be populated by nvidia_power profiler)
        self.peak_gpu_power_list = []
        self.avg_gpu_power_list = []
        self.peak_gpu_temp_list = []
        self.avg_gpu_temp_list = []
        self.power_plot_list = []

        # Per-prompt label tracking
        self.prompt_labels = []

        # Benchmark-specific metrics
        self.query_latency_list = []

    @staticmethod
    def parser(add_help: bool = True) -> argparse.ArgumentParser:
        parser = __class__.helpful_parser(
            short_description="Benchmark an LLM in TensorRT-LLM",
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

        parser.add_argument(
            "--max-seq-len",
            type=int,
            default=4096,
            help="Maximum sequence length (default: 4096)",
        )

        parser.add_argument(
            "--max-num-tokens",
            type=int,
            default=8192,
            help="Maximum total tokens in a batch (default: 8192)",
        )

        parser.add_argument(
            "--trust-remote-code",
            action="store_true",
            help="Trust remote code when loading model",
        )

        return parser

    def parse(self, state: State, args, known_only=True) -> argparse.Namespace:
        """
        Helper function to parse CLI arguments into the args expected by run()
        """

        # Call Tool parse method, NOT the Bench parse method
        parsed_args = Tool.parse(self, state, args, known_only)

        # Save prompt_label from our parser before calling parent
        prompt_labels = getattr(parsed_args, 'prompt_label', None)
        parsed_args = super().parse(state, args, known_only)

        # Restore prompt_label (keep singular for consistency with llamacpp)
        parsed_args.prompt_label = prompt_labels if prompt_labels else None
        parsed_args.max_seq_len = getattr(parsed_args, 'max_seq_len', 4096)
        parsed_args.max_num_tokens = getattr(parsed_args, 'max_num_tokens', 8192)
        parsed_args.trust_remote_code = getattr(parsed_args, 'trust_remote_code', False)

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
        max_seq_len: int = 4096,
        max_num_tokens: int = 8192,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> State:
        """
        Run TensorRT-LLM benchmark inside Docker container

        This method validates setup and stores parameters, then calls parent run()
        which will call run_prompt() for each prompt.
        """

        # Validate we have a TensorRT-LLM model loaded
        if not isinstance(state.model, TensorRTLLMAdapter):
            raise ValueError(
                "TensorRT-LLM model not loaded. Please run trtllm-load first."
            )

        # Get Docker manager from state
        docker_manager: DockerManager = state.docker_manager
        if not docker_manager:
            raise ValueError("Docker manager not found in state")

        # Store benchmark-specific parameters in instance for use by run_prompt()
        self.prompt_labels = prompt_label
        self.max_seq_len = max_seq_len
        self.max_num_tokens = max_num_tokens
        self.trust_remote_code = trust_remote_code
        self.docker_manager = docker_manager

        # Create output directory for results in the cache
        self.results_dir = os.path.join(os.path.expanduser(state.cache_dir), "trtllm_results")
        os.makedirs(self.results_dir, exist_ok=True)

        # Copy benchmark_core.py to container once at the start
        benchmark_core_path = Path(__file__).parent / "benchmark_core.py"
        container_benchmark_path = "/tmp/trtllm_benchmark_core.py"

        self._copy_file_to_container(
            docker_manager,
            str(benchmark_core_path),
            container_benchmark_path
        )

        printing.log_info(f"Running TensorRT-LLM benchmark with {len(prompts)} prompts")
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
        Run benchmark for a single prompt inside Docker container.

        This method is called by the parent Bench.run() for each prompt.
        Each execution reloads the model to avoid cache effects.
        """

        # Determine prompt label for this iteration
        prompt_index = len(self.input_ids_len_list)
        prompt_label = (
            self.prompt_labels[prompt_index]
            if self.prompt_labels and prompt_index < len(self.prompt_labels)
            else f"prompt_{prompt_index + 1}"
        )

        # Prepare prompt data for this single prompt
        prompts_data = [{
            'prompt': prompt,
            'max_tokens': output_tokens,
            'filename': prompt_label
        }]

        # Create output file for this prompt
        output_file = os.path.join(self.results_dir, f"benchmark_results_{prompt_index}.json")

        # Compute container path relative to the mounted workspace
        # The cwd is mounted at /workspace/lemonade, so we need to get the relative path
        # from cwd to the results file
        cwd = os.path.abspath(os.getcwd())
        rel_path = os.path.relpath(output_file, cwd)
        container_output_file = os.path.join("/workspace/lemonade", rel_path).replace("\\", "/")

        # Create a Python script to run the benchmark with this prompt
        benchmark_script = self._create_benchmark_script(
            prompts_data,
            state.model.model,
            iterations,
            warmup_iterations,
            output_tokens,
            self.max_seq_len,
            self.max_num_tokens,
            self.trust_remote_code,
            container_output_file
        )

        # Write script to a temp file and copy to container
        script_path = os.path.join(self.results_dir, f"run_benchmark_{prompt_index}.py")
        with open(script_path, 'w') as f:
            f.write(benchmark_script)

        container_script_path = f"/tmp/run_trtllm_benchmark_{prompt_index}.py"
        self._copy_file_to_container(
            self.docker_manager,
            script_path,
            container_script_path
        )

        # Execute benchmark inside container with live output streaming
        printing.log_info(f"Executing benchmark for prompt {prompt_index + 1} in container...")
        printing.log_info("=" * 80)
        printing.log_info("Live output from Docker container:")
        printing.log_info("=" * 80)
        report_progress_fn(0.1)

        result = self.docker_manager.exec_command(
            ["python", container_script_path],
            workdir="/workspace/lemonade",
            stream_output=True  # Enable real-time output streaming
        )

        printing.log_info("=" * 80)
        if result.returncode != 0:
            printing.log_error(f"Benchmark failed with exit code: {result.returncode}")
            raise RuntimeError(f"TensorRT-LLM benchmark failed with exit code: {result.returncode}")

        report_progress_fn(0.9)
        printing.log_success(f"Benchmark for prompt {prompt_index + 1} completed successfully")

        # Load results from JSON file
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                results = json.load(f)

            # Process results for this single prompt
            per_query_results = results.get('per_query_results', [])
            if per_query_results:
                # Get the first (and only) query result
                query_result = per_query_results[0]
                self._process_single_query_result(state, query_result, prompt_label)
            else:
                printing.log_warning(f"No query results found for prompt: {prompt_label}")
        else:
            printing.log_warning(f"Results file not found: {output_file}")

        report_progress_fn(1.0)

    def _copy_file_to_container(
        self,
        docker_manager: DockerManager,
        host_path: str,
        container_path: str
    ):
        """Copy a file from host to container"""
        import subprocess

        cmd = [
            "docker", "cp",
            host_path,
            f"{docker_manager.container_name}:{container_path}"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to copy file to container: {result.stderr}")

    def _create_benchmark_script(
        self,
        prompts_data: List[Dict],
        model_path: str,
        num_iterations: int,
        num_warmup: int,
        max_tokens: int,
        max_seq_len: int,
        max_num_tokens: int,
        trust_remote_code: bool,
        output_file: str
    ) -> str:
        """Create a Python script that will run the benchmark"""

        # Escape the prompts data for embedding in Python script
        prompts_json = json.dumps(prompts_data, indent=4)

        script = f'''
import sys
import asyncio
import json

# Add the benchmark_core module to path
sys.path.insert(0, '/tmp')

from trtllm_benchmark_core import run_benchmark

# Prompts data
prompts = {prompts_json}

if __name__ == '__main__':
    # Run the benchmark
    results = asyncio.run(run_benchmark(
        model_path="{model_path}",
        prompts=prompts,
        num_iterations={num_iterations},
        num_warmup={num_warmup},
        max_tokens={max_tokens},
        max_seq_len={max_seq_len},
        max_num_tokens={max_num_tokens},
        trust_remote_code={str(trust_remote_code)},
        output_file="{output_file}"
    ))

    print("Benchmark completed successfully!")
    print(f"Results saved to: {output_file}")
'''
        return script

    def _process_single_query_result(
        self,
        state: State,
        query_result: Dict[str, Any],
        prompt_label: str
    ):
        """
        Process benchmark results for a single query and append to measurement lists.

        This method extracts metrics from the benchmark results and appends them to the
        per-prompt measurement lists that are inherited from the Bench base class.

        Args:
            state: Lemonade state object
            query_result: Single query result dictionary
            prompt_label: Label for this prompt
        """

        # Time to first token
        ttft = query_result.get('averaged_ttft', 0)
        self.mean_time_to_first_token_list.append(ttft)

        # Calculate std dev from individual iterations if available
        individual_iterations = query_result.get('individual_iterations', [])
        if len(individual_iterations) > 1:
            import statistics
            ttft_values = [it['ttft'] for it in individual_iterations]
            std_ttft = statistics.stdev(ttft_values)
        else:
            std_ttft = 0
        self.std_dev_time_to_first_token_list.append(std_ttft)

        # Tokens per second (decode phase)
        tps = query_result.get('averaged_tokens_per_sec', 0)
        self.token_generation_tokens_per_second_list.append(tps)

        if len(individual_iterations) > 1:
            tps_values = [it['tokens_per_sec'] for it in individual_iterations]
            std_tps = statistics.stdev(tps_values)
        else:
            std_tps = 0
        self.std_dev_token_generation_tokens_per_second_list.append(std_tps)

        # Prefill tokens per second (estimated from input tokens and TTFT)
        input_tokens = query_result.get('input_tokens', 0)
        if ttft > 0:
            prefill_tps = input_tokens / ttft
        else:
            prefill_tps = 0
        self.prefill_tokens_per_second_list.append(prefill_tps)
        self.std_dev_prefill_tokens_per_second_list.append(0)  # No std dev for prefill

        # Token counts
        self.input_ids_len_list.append(input_tokens)
        self.tokens_out_len_list.append(query_result.get('output_tokens', 0))

        # Query latency (store for potential future use)
        latency = query_result.get('averaged_query_latency', 0)
        self.query_latency_list.append(latency)

        # Power metrics (these will be populated by nvidia_power profiler on host)
        # For now, initialize with 0 - profiler will update these
        self.avg_gpu_power_list.append(0)
        self.peak_gpu_power_list.append(0)
        self.avg_gpu_temp_list.append(0)
        self.peak_gpu_temp_list.append(0)

        # Memory (not available in current TensorRT-LLM results)
        self.max_memory_used_gb_list.append(0)

        printing.log_info(
            f"Prompt '{prompt_label}': TTFT={ttft*1000:.2f}ms, "
            f"TPS={tps:.2f}, Input={input_tokens}, Output={query_result.get('output_tokens', 0)}"
        )

    def display_prompt_results(self, state, prompt, prompt_index):
        """Display results for a single prompt immediately after benchmarking"""
        import sys

        # Get the real terminal stdout (not redirected by Logger)
        # The Logger redirects sys.stdout, but saves the original as self.terminal
        output = sys.stdout
        if hasattr(sys.stdout, 'terminal'):
            output = sys.stdout.terminal

        # Get prompt label
        if len(self.prompt_labels) > prompt_index:
            prompt_label = self.prompt_labels[prompt_index]
        else:
            prompt_label = f"prompt_{prompt_index + 1}"

        # Get actual token count from benchmark results
        actual_token_count = self.input_ids_len_list[prompt_index] if len(self.input_ids_len_list) > prompt_index else 0

        # Print newline and separator directly to terminal
        output.write(f"\n{'='*80}\n")
        output.write(f"Results for Prompt: {prompt_label} ({actual_token_count} tokens) (Prompt {prompt_index + 1})\n")
        output.write(f"{'='*80}\n")
        output.flush()

        # Get the current prompt's results (last item in each list)
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

        # Display power metrics from per-prompt lists
        if len(self.peak_gpu_power_list) > idx and self.peak_gpu_power_list[idx] != 0:
            results["Peak GPU Power"] = self.peak_gpu_power_list[idx]
            results["Avg GPU Power"] = self.avg_gpu_power_list[idx]

            # Add temperature if available
            if len(self.peak_gpu_temp_list) > idx and self.peak_gpu_temp_list[idx] != 0:
                results["Peak GPU Temp"] = self.peak_gpu_temp_list[idx]
                results["Avg GPU Temp"] = self.avg_gpu_temp_list[idx]

            # Add plot path if available
            if len(self.power_plot_list) > idx:
                results["Power Usage Plot"] = self.power_plot_list[idx]

        # Display query latency
        if len(self.query_latency_list) > idx:
            results["Query Latency (s)"] = f"{self.query_latency_list[idx]:.3f}"

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
        from lemonade.profilers.nvidia_power import Keys as NvidiaKeys
        import lemonade.common.filesystem as fs

        # Load stats from the YAML file (not from state.stats attribute)
        stats_obj = fs.Stats(state.cache_dir, state.build_name)
        stats = stats_obj.stats  # This loads from lemonade_stats.yaml

        if stats:
            # Check for Nvidia power metrics
            if NvidiaKeys.PEAK_GPU_POWER in stats:
                self.peak_gpu_power_list[-1] = stats[NvidiaKeys.PEAK_GPU_POWER]
                self.avg_gpu_power_list[-1] = stats[NvidiaKeys.AVG_GPU_POWER]
                self.peak_gpu_temp_list[-1] = stats[NvidiaKeys.PEAK_GPU_TEMP]
                self.avg_gpu_temp_list[-1] = stats[NvidiaKeys.AVG_GPU_TEMP]
                if NvidiaKeys.POWER_USAGE_PLOT in stats:
                    self.power_plot_list.append(stats[NvidiaKeys.POWER_USAGE_PLOT])


# Copyright (c) 2025 AMD
