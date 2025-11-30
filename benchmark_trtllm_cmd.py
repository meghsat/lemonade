#!/usr/bin/env python3
import os
import sys
import subprocess
import re
import json
import threading
import time
import pandas as pd
from pathlib import Path


class PowerMonitor:
    def __init__(self, gpu_index=0, interval=0.05):
        self.gpu_index = gpu_index
        self.interval = interval
        self.tracking_active = False
        self.data = []
        self.thread = None

    def _monitor(self):
        start_time = time.time()
        while self.tracking_active:
            try:
                current_time = time.time() - start_time
                cmd = [
                    "nvidia-smi",
                    f"--id={self.gpu_index}",
                    "--query-gpu=power.draw,temperature.gpu,utilization.gpu,utilization.memory",
                    "--format=csv,noheader,nounits"
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    values = result.stdout.strip().split(',')
                    if len(values) == 4:
                        self.data.append({
                            'time': current_time,
                            'power_draw': float(values[0].strip()),
                            'temperature': float(values[1].strip()),
                            'gpu_utilization': float(values[2].strip()),
                            'memory_utilization': float(values[3].strip()),
                        })
                time.sleep(self.interval)
            except Exception as e:
                print(f"Error monitoring GPU: {e}")
                break

    def start(self):
        self.tracking_active = True
        self.data = []
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self):
        self.tracking_active = False
        if self.thread:
            self.thread.join(timeout=5)

    def get_stats(self):
        if not self.data:
            return {}
        df = pd.DataFrame(self.data)
        return {
            'peak_power': df['power_draw'].max(),
            'avg_power': df['power_draw'].mean(),
            'peak_temp': df['temperature'].max(),
            'avg_temp': df['temperature'].mean(),
        }


def extract_metrics(output):
    """Extract performance metrics from trtllm-bench output."""
    metrics = {}
    patterns = {
        'request_throughput': r'Request Throughput \(req/sec\):\s+([\d.]+)',
        'total_output_throughput': r'Total Output Throughput \(tokens/sec\):\s+([\d.]+)',
        'total_token_throughput': r'Total Token Throughput \(tokens/sec\):\s+([\d.]+)',
        'total_latency': r'Total Latency \(ms\):\s+([\d.]+)',
        'avg_request_latency': r'Average request latency \(ms\):\s+([\d.]+)',
        'avg_ttft': r'Average time-to-first-token \[TTFT\] \(ms\):\s+([\d.]+)',
        'avg_tpot': r'Average time-per-output-token \[TPOT\] \(ms\):\s+([\d.]+)',
        'tpot_min': r'\[TPOT\] MINIMUM:\s+([\d.]+)',
        'tpot_max': r'\[TPOT\] MAXIMUM:\s+([\d.]+)',
        'tpot_p50': r'\[TPOT\] P50\s+:\s+([\d.]+)',
        'tpot_p90': r'\[TPOT\] P90\s+:\s+([\d.]+)',
        'tpot_p95': r'\[TPOT\] P95\s+:\s+([\d.]+)',
        'tpot_p99': r'\[TPOT\] P99\s+:\s+([\d.]+)',
        'ttft_min': r'\[TTFT\] MINIMUM:\s+([\d.]+)',
        'ttft_max': r'\[TTFT\] MAXIMUM:\s+([\d.]+)',
        'ttft_p50': r'\[TTFT\] P50\s+:\s+([\d.]+)',
        'ttft_p90': r'\[TTFT\] P90\s+:\s+([\d.]+)',
        'ttft_p95': r'\[TTFT\] P95\s+:\s+([\d.]+)',
        'ttft_p99': r'\[TTFT\] P99\s+:\s+([\d.]+)',
        'latency_min': r'\[Latency\] MINIMUM:\s+([\d.]+)',
        'latency_max': r'\[Latency\] MAXIMUM:\s+([\d.]+)',
        'latency_avg': r'\[Latency\] AVERAGE:\s+([\d.]+)',
        'latency_p50': r'\[Latency\] P50\s+:\s+([\d.]+)',
        'latency_p90': r'\[Latency\] P90\s+:\s+([\d.]+)',
        'latency_p95': r'\[Latency\] P95\s+:\s+([\d.]+)',
        'latency_p99': r'\[Latency\] P99\s+:\s+([\d.]+)',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            metrics[key] = float(match.group(1))

    return metrics


def clear_gpu_cache():
    """Clear GPU cache."""
    try:
        subprocess.run(['nvidia-smi', '--gpu-reset'], check=False, capture_output=True)
    except:
        pass


def run_benchmark(model, jsonl_file, gpu_index=0):
    print(f"\n{'='*60}")
    print(f"Processing: {jsonl_file}")
    print(f"{'='*60}")

    power_monitor = PowerMonitor(gpu_index=gpu_index)
    power_monitor.start()

    cmd = [
        'trtllm-bench',
        '--model', model,
        'throughput',
        '--dataset', jsonl_file,
        '--backend', 'pytorch',
        '--streaming'
    ]

    output = ""
    try:
        print(f"Running command: {' '.join(cmd)}\n")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=None)
        output = result.stdout + result.stderr

        if result.returncode != 0:
            print(f"ERROR: Command failed with return code {result.returncode}")
            print(f"STDERR:\n{result.stderr}")
        else:
            print("Command completed successfully")

    except Exception as e:
        print(f"Error running benchmark: {e}")
    finally:
        power_monitor.stop()

    metrics = extract_metrics(output)
    power_stats = power_monitor.get_stats()

    results = {
        'file': str(jsonl_file),
        **metrics,
        **power_stats
    }

    # Save detailed output to log file
    log_file = Path(jsonl_file).stem + "_benchmark.log"
    with open(log_file, 'w') as f:
        f.write(output)
    print(f"Full output saved to {log_file}")

    clear_gpu_cache()
    time.sleep(2)

    return results, output


def main():
    if len(sys.argv) < 3:
        print("Usage: python benchmark_trtllm_runner.py <model> <jsonl_folder> [gpu_index]")
        print("Example: python benchmark_trtllm_runner.py openai/gpt-oss-20b ./prompts 0")
        sys.exit(1)

    model = sys.argv[1]
    jsonl_folder = Path(sys.argv[2])
    gpu_index = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    if not jsonl_folder.exists():
        print(f"Error: Folder {jsonl_folder} does not exist")
        sys.exit(1)

    # Find all JSONL files
    jsonl_files = sorted(jsonl_folder.glob('*.jsonl'))
    if not jsonl_files:
        print(f"Error: No .jsonl files found in {jsonl_folder}")
        sys.exit(1)

    print(f"Found {len(jsonl_files)} JSONL files")
    print(f"Model: {model}")
    print(f"GPU Index: {gpu_index}")

    # Run benchmarks
    all_results = []
    for jsonl_file in jsonl_files:
        results, output = run_benchmark(model, str(jsonl_file), gpu_index)
        all_results.append(results)

        # Print summary
        print(f"\nResults for {jsonl_file.name}:")
        if 'request_throughput' in results:
            print(f"  Request Throughput: {results['request_throughput']:.2f} req/sec")
        if 'total_output_throughput' in results:
            print(f"  Output Throughput: {results['total_output_throughput']:.2f} tokens/sec")
        if 'avg_power' in results:
            print(f"  Avg Power: {results['avg_power']:.1f} W")
        if 'peak_power' in results:
            print(f"  Peak Power: {results['peak_power']:.1f} W")

    # Save results to CSV
    output_file = 'benchmark_results.csv'
    df = pd.DataFrame(all_results)
    df.to_csv(output_file, index=False)
    print(f"\n{'='*60}")
    print(f"Results saved to {output_file}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
