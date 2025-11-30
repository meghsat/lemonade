#!/usr/bin/env python3

import os
import re
import csv
import time
import argparse
import threading
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple

# Disable tokenizers parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
from mlx_lm import load, generate


class ApplePowerTracker:

    def __init__(self, interval_s: float = 0.05):
        self.interval_s = interval_s
        self.tracking_active = False
        self.powermetrics_data = []
        self.powermetrics_thread = None

    def _monitor_powermetrics(self):
        start_time = time.time()

        while self.tracking_active:
            try:
                current_time = time.time() - start_time

                cmd = [
                    "sudo",
                    "-n",  # non-interactive
                    "powermetrics",
                    "-n", "1",  # single sample
                    "-i", str(int(self.interval_s * 1000)),
                    "--samplers", "cpu_power,gpu_power,ane_power",
                    "--show-usage-summary",
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if result.returncode == 0:
                    output = result.stdout

                    gpu_power = self._extract_value(output, "GPU Power")
                    cpu_power = self._extract_value(output, "CPU Power")
                    ane_power = self._extract_value(output, "ANE Power")
                    combined_power = self._extract_value(output, "Combined Power (CPU + GPU + ANE)")

                    if gpu_power is not None or cpu_power is not None:
                        sample_data = {
                            'time': current_time,
                            'gpu_power': gpu_power if gpu_power is not None else 0,
                            'cpu_power': cpu_power if cpu_power is not None else 0,
                            'ane_power': ane_power if ane_power is not None else 0,
                            'combined_power': combined_power if combined_power is not None else 0,
                        }
                        self.powermetrics_data.append(sample_data)
                else:
                    if len(self.powermetrics_data) == 0:
                        error_msg = (
                            f"\npowermetrics failed (code {result.returncode}). "
                            "Ensure you have run 'sudo powermetrics -n 1' to authenticate first"
                        )
                        self.tracking_active = False
                        raise RuntimeError(error_msg)

                time.sleep(self.interval_s)

            except Exception as e:
                print(f"Error in powermetrics monitoring: {e}")
                break

    def _extract_value(self, text: str, key: str) -> float:
        try:
            pattern_text = rf"{re.escape(key)}:\s*([0-9.]+)\s*mW"
            match = re.search(pattern_text, text)
            if match:
                value = float(match.group(1))
                # powermetrics reports in milliwatts, convert to watts
                return value / 1000.0
            return None
        except Exception:
            return None

    def start(self):
        if self.tracking_active:
            raise RuntimeError("Power tracking already active")

        self.tracking_active = True
        self.powermetrics_data = []
        self.powermetrics_thread = threading.Thread(
            target=self._monitor_powermetrics,
            daemon=True
        )
        self.powermetrics_thread.start()

    def stop(self) -> Dict[str, float]:
        if not self.tracking_active:
            return {}

        self.tracking_active = False

        if self.powermetrics_thread:
            self.powermetrics_thread.join(timeout=5)

        if not self.powermetrics_data:
            print(" [No power data collected]", end='')
            return {}

        gpu_powers = [d['gpu_power'] for d in self.powermetrics_data]
        cpu_powers = [d['cpu_power'] for d in self.powermetrics_data]
        ane_powers = [d['ane_power'] for d in self.powermetrics_data]
        combined_powers = [d['combined_power'] for d in self.powermetrics_data]

        stats = {
            'avg_gpu_power': sum(gpu_powers) / len(gpu_powers) if gpu_powers else 0,
            'peak_gpu_power': max(gpu_powers) if gpu_powers else 0,
            'avg_cpu_power': sum(cpu_powers) / len(cpu_powers) if cpu_powers else 0,
            'peak_cpu_power': max(cpu_powers) if cpu_powers else 0,
            'avg_ane_power': sum(ane_powers) / len(ane_powers) if ane_powers else 0,
            'peak_ane_power': max(ane_powers) if ane_powers else 0,
            'avg_combined_power': sum(combined_powers) / len(combined_powers) if combined_powers else 0,
            'peak_combined_power': max(combined_powers) if combined_powers else 0,
        }

        print(f" [GPU: {stats['avg_gpu_power']:.1f}W, CPU: {stats['avg_cpu_power']:.1f}W, ANE: {stats['avg_ane_power']:.1f}W]", end='')

        return stats


def parse_filename(filename: str) -> Tuple[str, int, int, int]:
    # Pattern: {model}_p{number}_in{number}_out{number}.txt
    pattern = r'^(.+?)_p(\d+)_in(\d+)_out(\d+)\.txt$'
    match = re.match(pattern, filename)

    if not match:
        raise ValueError(f"Filename '{filename}' does not match expected format")

    model_name = match.group(1)
    prompt_num = int(match.group(2))
    input_tokens = int(match.group(3))
    output_tokens = int(match.group(4))

    return model_name, prompt_num, input_tokens, output_tokens


def run_mlx_benchmark_with_metrics(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int,
    power_tracker: ApplePowerTracker = None
) -> Dict[str, float]:
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    print(power_tracker)
    if power_tracker:
        power_tracker.start()

    start_time = time.time()
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=True,
    )
    end_time = time.time()

    power_stats = {}
    if power_tracker:
        power_stats = power_tracker.stop()

    sys.stdout = old_stdout
    output = captured_output.getvalue()

    metrics = {
        'total_time': end_time - start_time,
    }

    prompt_tps_match = re.search(r'Prompt:.*?([0-9.]+) tokens-per-sec', output)
    ttft_match = re.search(r'Time to first token:\s*([0-9.]+)\s*sec', output)
    gen_tps_match = re.search(r'Generation:.*?([0-9.]+) tokens-per-sec', output)
    peak_mem_match = re.search(r'Peak memory:\s*([0-9.]+)\s*GB', output)

    if prompt_tps_match:
        metrics['prompt_tps'] = float(prompt_tps_match.group(1))
    if ttft_match:
        metrics['ttft'] = float(ttft_match.group(1))
    if gen_tps_match:
        metrics['generation_tps'] = float(gen_tps_match.group(1))
    if peak_mem_match:
        metrics['peak_memory_gb'] = float(peak_mem_match.group(1))

    metrics.update(power_stats)

    return metrics


def run_benchmark_for_file(
    prompt_file: Path,
    model,
    tokenizer,
    num_runs: int = 3
) -> Dict[str, any]:
    """
    Returns:
        Dictionary with averaged benchmark results
    """
    model_name, prompt_num, input_tokens, output_tokens = parse_filename(prompt_file.name)

    print(f"\nBenchmarking: {prompt_file.name}")
    print(f"  Model: {model_name}, Prompt: {prompt_num}, Input: {input_tokens}, Output: {output_tokens}")

    with open(prompt_file, 'r') as f:
        prompt_text = f.read().strip()

    all_metrics = []

    for run_idx in range(num_runs):
        print(f"  Run {run_idx + 1}/{num_runs}...", end='', flush=True)

        power_tracker = ApplePowerTracker()

        try:
            metrics = run_mlx_benchmark_with_metrics(
                model,
                tokenizer,
                prompt=prompt_text,
                max_tokens=output_tokens,
                power_tracker=power_tracker
            )
            all_metrics.append(metrics)
            print(" done")
        except Exception as e:
            print(f" ERROR: {e}")
            continue

    if not all_metrics:
        print(f"  WARNING: No successful runs for {prompt_file.name}")
        return None

    # Average the metrics
    avg_metrics = {
        'model': model_name,
        'prompt_num': prompt_num,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'filename': prompt_file.name,
        'num_runs': len(all_metrics),
    }

    metric_keys = ['ttft', 'generation_tps', 'prompt_tps', 'peak_memory_gb',
                   'avg_gpu_power', 'peak_gpu_power', 'avg_cpu_power', 'peak_cpu_power',
                   'avg_ane_power', 'peak_ane_power',
                   'avg_combined_power', 'peak_combined_power', 'total_time']

    for key in metric_keys:
        values = [m[key] for m in all_metrics if key in m]
        if values:
            avg_metrics[f'{key}_avg'] = sum(values) / len(values)
            avg_metrics[f'{key}_min'] = min(values)
            avg_metrics[f'{key}_max'] = max(values)

    return avg_metrics


def check_powermetrics_access():
    try:
        result = subprocess.run(
            ["sudo", "-n", "powermetrics", "-n", "1", "-i", "1000",
             "--samplers", "cpu_power", "--show-usage-summary"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'prompt_dir',
        type=str
        )
    parser.add_argument(
        '--model',
        type=str,
        default='openai/gpt-oss-20b'
    )
    parser.add_argument(
        '--num-runs',
        type=int,
        default=3
    )
    parser.add_argument(
        '--output',
        type=str,
        default='mlx_benchmark_results.csv'
    )

    args = parser.parse_args()

    print("Checking powermetrics access...")
    if not check_powermetrics_access():
        print("\n" + "="*60)
        print("WARNING: Cannot access powermetrics with sudo")
        print("="*60)
        print("Power tracking requires sudo access to powermetrics.")
        print("Please re-run the command with sudo and enter your password:")
        return 1
    print("powermetrics access confirmed\n")

    prompt_dir = Path(args.prompt_dir)
    if not prompt_dir.exists():
        print(f"Error: Directory '{args.prompt_dir}' does not exist")
        return 1

    prompt_files = sorted(prompt_dir.glob('*.txt'))
    if not prompt_files:
        print(f"Error: No .txt files found in '{args.prompt_dir}'")
        return 1

    print(f"Found {len(prompt_files)} prompt files")

    print(f"\nLoading model: {args.model}")
    model, tokenizer = load(
        args.model,
        tokenizer_config={"trust_remote_code": True}
    )
    print("Model loaded successfully")

    results = []

    for prompt_file in prompt_files:
        try:
            result = run_benchmark_for_file(
                prompt_file,
                model,
                tokenizer,
                num_runs=args.num_runs
            )
            if result:
                results.append(result)
        except Exception as e:
            print(f"Error processing {prompt_file.name}: {e}")
            continue

    # Save results to CSV
    if results:
        print(f"\nSaving results to {args.output}")

        all_keys = set()
        for r in results:
            all_keys.update(r.keys())

        fieldnames = ['filename', 'model', 'prompt_num', 'input_tokens', 'output_tokens', 'num_runs']
        metric_columns = [k for k in sorted(all_keys) if k not in fieldnames]
        fieldnames.extend(metric_columns)

        with open(args.output, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result)

        print(f"Results saved: {len(results)} benchmarks completed")

        # Print summary
        print(" BENCHMARK SUMMARY")
        for result in results:
            print(f"\n{result['filename']}:")
            if 'ttft_avg' in result:
                print(f"  Time to First Token: {result['ttft_avg']:.3f}s")
            if 'generation_tps_avg' in result:
                print(f"  Generation TPS:      {result['generation_tps_avg']:.2f} tokens/sec")
            if 'prompt_tps_avg' in result:
                print(f"  Prompt TPS:          {result['prompt_tps_avg']:.2f} tokens/sec")
            if 'peak_memory_gb_avg' in result:
                print(f"  Peak Memory:         {result['peak_memory_gb_avg']:.2f} GB")

            # Power metrics
            has_power = False
            if 'avg_gpu_power_avg' in result and result['avg_gpu_power_avg'] > 0:
                print(f"  Avg GPU Power:       {result['avg_gpu_power_avg']:.2f}W")
                has_power = True
            if 'peak_gpu_power_avg' in result and result['peak_gpu_power_avg'] > 0:
                print(f"  Peak GPU Power:      {result['peak_gpu_power_avg']:.2f}W")
                has_power = True
            if 'avg_cpu_power_avg' in result and result['avg_cpu_power_avg'] > 0:
                print(f"  Avg CPU Power:       {result['avg_cpu_power_avg']:.2f}W")
                has_power = True
            if 'avg_ane_power_avg' in result and result['avg_ane_power_avg'] > 0:
                print(f"  Avg ANE Power:       {result['avg_ane_power_avg']:.2f}W")
                has_power = True
            if 'peak_ane_power_avg' in result and result['peak_ane_power_avg'] > 0:
                print(f"  Peak ANE Power:      {result['peak_ane_power_avg']:.2f}W")
                has_power = True
            if 'avg_combined_power_avg' in result and result['avg_combined_power_avg'] > 0:
                print(f"  Avg Combined Power:  {result['avg_combined_power_avg']:.2f}W")
                has_power = True

            if not has_power:
                print(f"  Power metrics:       Not available")

        print("\n" + "="*70)
    else:
        print("\nNo results to save")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
