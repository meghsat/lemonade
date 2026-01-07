# python benchmark_estimator_vllm_files.py --model_dir openai/gpt-oss-20b --prompts_folder /workspace/prompt_files --num_warmup 2 --num_iterations 3 --output_file vllm_gptoss20b_phi.json
import argparse
import json
import time
import gc
from pathlib import Path
from typing import List, Dict, Any
import threading
import subprocess
import pandas as pd

from vllm import LLM
from vllm.sampling_params import SamplingParams


DEFAULT_POWER_INTERVAL_S = 0.05  # Sample every 50ms


class PowerMonitor:

    def __init__(self, gpu_index=0, interval=DEFAULT_POWER_INTERVAL_S):
        self.gpu_index = gpu_index
        self.interval = interval
        self.tracking_active = False
        self.data = []
        self.thread = None

    def _monitor(self):
        """Background thread that monitors GPU power and temperature using nvidia-smi."""
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
                print(f"Warning: Error monitoring GPU: {e}")
                break

    def start(self):
        """Start power monitoring in background thread."""
        self.tracking_active = True
        self.data = []
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop power monitoring."""
        self.tracking_active = False
        if self.thread:
            self.thread.join(timeout=5)

    def get_stats(self):
        if not self.data:
            return {}
        df = pd.DataFrame(self.data)
        return {
            'peak_power': float(df['power_draw'].max()),
            'avg_power': float(df['power_draw'].mean()),
            'peak_temp': float(df['temperature'].max()),
            'avg_temp': float(df['temperature'].mean()),
            'avg_gpu_utilization': float(df['gpu_utilization'].mean()),
            'avg_memory_utilization': float(df['memory_utilization'].mean()),
        }


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir', type=str, required=True,
                        help="Model name")
    parser.add_argument('--prompts_folder', type=str,
                        help="Path to folder containing individual prompt text files (phi_p*_in{isl}_out{osl}.txt)")

    parser.add_argument('--max_tokens', type=int, default=2048,
                        help="Maximum number of tokens to generate")

    parser.add_argument('--gpu_memory_utilization', type=float, default=0.90,
                        help="GPU memory utilization fraction (default: 0.90)")

    parser.add_argument('--num_warmup', type=int, default=2)
    parser.add_argument('--num_iterations', type=int, default=3)
    parser.add_argument('--output_file', type=str, default='benchmark_results.json',
                        help="Output JSON file for results")
    parser.add_argument('--trust_remote_code', action='store_true',
                        help="Trust remote code")

    return parser.parse_args()


def load_prompts(args) -> List[Dict[str, Any]]:
    import re

    if args.prompts_folder:
        folder_path = Path(args.prompts_folder)
        if not folder_path.exists():
            raise FileNotFoundError(f"Prompts folder not found: {folder_path}")

        # Pattern: phi_p*_in{isl}_out{osl}.txt
        pattern = re.compile(r'phi_p\d+_in(\d+)_out(\d+)\.txt')

        prompt_data = []
        for file_path in sorted(folder_path.glob('*.txt')):
            match = pattern.match(file_path.name)
            if match:
                isl = int(match.group(1))  # ISL
                osl = int(match.group(2))  # OSL

                with open(file_path, 'r', encoding='utf-8') as f:
                    prompt_text = f.read().strip()

                prompt_data.append({
                    'prompt': prompt_text,
                    'max_tokens': osl,
                    'isl': isl,
                    'osl': osl,
                    'filename': file_path.name
                })

        if not prompt_data:
            raise ValueError(f"No valid prompt files found in {folder_path} matching pattern phi_p*_in*_out*.txt")

        return prompt_data

    else:
        return [
            {'prompt': "Explain the concept of neural networks in simple terms."},
            {'prompt': "What are the main differences between supervised and unsupervised learning?"},
            {'prompt': "Describe how a transformer model works."},
        ]


def setup_llm(args) -> LLM:
    """Initialize vLLM model."""
    llm = LLM(
        model=args.model_dir,
        trust_remote_code=args.trust_remote_code,
        gpu_memory_utilization=args.gpu_memory_utilization,
        disable_log_stats=False,
    )

    return llm


def benchmark_single_query(llm: LLM, prompt: str, sampling_params: SamplingParams, power_monitor: PowerMonitor = None) -> Dict[str, Any]:

    if power_monitor:
        power_monitor.start()

    # Manual timing
    query_start = time.perf_counter()
    outputs = llm.generate([prompt], sampling_params)
    query_end = time.perf_counter()
    manual_query_latency = query_end - query_start

    power_stats = {}
    if power_monitor:
        power_monitor.stop()
        power_stats = power_monitor.get_stats()

    # Output contains: RequestOutput(request_id=0, prompt=' ', prompt_token_ids=[], encoder_prompt=None, encoder_prompt_token_ids=None, prompt_logprobs=None, outputs=[CompletionOutput(index=0, text="", token_ids=[], cumulative_logprob=None, logprobs=None, finish_reason=length, stop_reason=None)], finished=True, metrics=RequestStateStats(num_generation_tokens=50, arrival_time=, queued_ts=, scheduled_ts=, first_token_ts=, last_token_ts=, first_token_latency=, is_corrupted=False), lora_request=None, num_cached_tokens=0, multi_modal_placeholders={})
    output = outputs[0]
    generated_text = output.outputs[0].text
    output_token_count = len(output.outputs[0].token_ids)

    # Get input token count
    input_tokens = len(output.prompt_token_ids)

    # Extract timing metrics from vLLM's RequestMetrics
    if not hasattr(output, 'metrics') or not output.metrics:
        print(output)
        raise RuntimeError("Error: No metrics found in vLLM output. Metrics are required for accurate TTFT measurement.")

    metrics_obj = output.metrics

    # Use first_token_latency[TTFT] directly from vLLM metrics
    if hasattr(metrics_obj, 'first_token_latency') and metrics_obj.first_token_latency is not None:
        ttft = metrics_obj.first_token_latency
    else:
        raise RuntimeError("Error: first_token_latency not found in vLLM metrics. Cannot calculate TTFT.")

    # Calculate query latency: TTFT + decode time
    # decode_time = last_token_ts - first_token_ts
    if hasattr(metrics_obj, 'last_token_ts') and hasattr(metrics_obj, 'first_token_ts'):
        if metrics_obj.last_token_ts and metrics_obj.first_token_ts:
            decode_time = metrics_obj.last_token_ts - metrics_obj.first_token_ts
            query_latency = ttft + decode_time
        else:
            raise RuntimeError("Error: last_token_ts or first_token_ts is None in vLLM metrics.")
    else:
        raise RuntimeError("Error: last_token_ts or first_token_ts not found in vLLM metrics. Cannot calculate query latency.")

    if output_token_count > 1 and query_latency > ttft:
        tokens_per_sec = (output_token_count - 1) / (query_latency - ttft)
    else:
        tokens_per_sec = output_token_count / query_latency if query_latency > 0 else 0

    metrics = {
        'ttft': ttft,
        'query_latency': query_latency,
        'query_latency_manual': manual_query_latency,
        'decode_time': decode_time,
        'input_tokens': input_tokens,
        'output_tokens': output_token_count,
        'tokens_per_sec': tokens_per_sec,
        'generated_text': generated_text,
        'prompt': prompt,
        **power_stats  # Add power metrics
    }

    print(f"\n{'='*80}")
    print(f"Prompt: {prompt[:100]}...")
    print(f"Generated: {generated_text[:100]}...")
    print(f"TTFT: {ttft:.4f} s")
    print(f"Decode Time: {decode_time:.4f} s")
    print(f"Query Latency (vLLM metrics): {query_latency:.4f} s")
    print(f"Query Latency (manual timing): {manual_query_latency:.4f} s")
    print(f"Latency Difference: {abs(query_latency - manual_query_latency):.4f} s")
    print(f"Input Tokens: {input_tokens}")
    print(f"Output Tokens: {output_token_count}")
    print(f"Tokens/sec (decode): {tokens_per_sec:.2f}")
    if power_stats:
        print(f"Avg Power: {power_stats.get('avg_power', 0):.1f} W, Peak Power: {power_stats.get('peak_power', 0):.1f} W")
        print(f"Avg Temp: {power_stats.get('avg_temp', 0):.1f} °C, Peak Temp: {power_stats.get('peak_temp', 0):.1f} °C")
    print(f"{'='*80}\n")

    return metrics


def run_warmup(llm: LLM, prompt: str, sampling_params: SamplingParams, num_warmup: int) -> List[Dict[str, Any]]:
    """Run warmup iterations."""
    warmup_metrics = []
    print(f"\nRunning {num_warmup} warmup iterations...")

    for i in range(num_warmup):
        print(f"Warmup {i+1}/{num_warmup}: ", end='', flush=True)

        # Manual timing
        t_start = time.perf_counter()
        outputs = llm.generate([prompt], sampling_params)
        t_end = time.perf_counter()
        manual_total_time = t_end - t_start

        output = outputs[0]
        output_token_count = len(output.outputs[0].token_ids)

        # Extract TTFT from metrics if available
        if not hasattr(output, 'metrics') or not output.metrics:
            print(output)
            raise RuntimeError("Error: No metrics found in vLLM output during warmup. Metrics are required for accurate TTFT measurement.")

        metrics_obj = output.metrics
        if hasattr(metrics_obj, 'first_token_latency') and metrics_obj.first_token_latency is not None:
            ttft = metrics_obj.first_token_latency
        else:
            raise RuntimeError("Error: first_token_latency not found in vLLM metrics during warmup. Cannot calculate TTFT.")

        # Calculate total time: TTFT + decode time
        # decode_time = last_token_ts - first_token_ts
        if hasattr(metrics_obj, 'last_token_ts') and hasattr(metrics_obj, 'first_token_ts'):
            if metrics_obj.last_token_ts and metrics_obj.first_token_ts:
                decode_time = metrics_obj.last_token_ts - metrics_obj.first_token_ts
                total_time = ttft + decode_time
            else:
                raise RuntimeError("Error: last_token_ts or first_token_ts is None in vLLM metrics during warmup.")
        else:
            raise RuntimeError("Error: last_token_ts or first_token_ts not found in vLLM metrics during warmup.")

        warmup_metrics.append({
            'warmup_iteration': i + 1,
            'ttft': ttft,
            'output_tokens': output_token_count,
            'total_time': total_time,
            'total_time_manual': manual_total_time,
            'decode_time': decode_time
        })

        print(f"TTFT: {ttft:.4f} s - Total (vLLM): {total_time:.4f} s - Total (manual): {manual_total_time:.4f} s - {output_token_count} tokens")

    print("Warmup complete.\n")
    return warmup_metrics


def calculate_aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {}

    ttfts = [r['ttft'] for r in results]
    latencies = [r['query_latency'] for r in results]
    tps_values = [r['tokens_per_sec'] for r in results]
    input_tokens = [r['input_tokens'] for r in results]
    output_tokens = [r['output_tokens'] for r in results]

    total_output_tokens = sum(output_tokens)
    total_time = sum(latencies)

    aggregate = {
        'num_queries': len(results),
        'average_ttft': sum(ttfts) / len(ttfts),
        'min_ttft': min(ttfts),
        'max_ttft': max(ttfts),
        'average_tokens_per_sec': sum(tps_values) / len(tps_values),
        'min_tokens_per_sec': min(tps_values),
        'max_tokens_per_sec': max(tps_values),
        'average_latency': sum(latencies) / len(latencies),
        'total_latency': total_time,
        'average_input_tokens': sum(input_tokens) / len(input_tokens),
        'average_output_tokens': sum(output_tokens) / len(output_tokens),
        'total_output_tokens': total_output_tokens,
        'overall_throughput': total_output_tokens / total_time if total_time > 0 else 0,
    }

    return aggregate


def run_prompt_iterations(args, prompt_data, idx, total_prompts, base_sampling_params):
    """Run multiple iterations for a single prompt with model reload between iterations."""
    prompt_text = prompt_data['prompt']

    # Use per-prompt max_tokens if available, otherwise use base params
    if 'max_tokens' in prompt_data:
        sampling_params = SamplingParams(
            max_tokens=prompt_data['max_tokens'],
            temperature=base_sampling_params.temperature,
        )
        print(f"\n{'='*80}")
        print(f"Processing query {idx}/{total_prompts} (max_tokens={prompt_data['max_tokens']})")
        if 'filename' in prompt_data:
            print(f"  File: {prompt_data['filename']}")
        print(f"  Running {args.num_iterations} iterations with model refresh between each")
        print(f"{'='*80}")
    else:
        sampling_params = base_sampling_params
        print(f"\n{'='*80}")
        print(f"Processing query {idx}/{total_prompts}")
        print(f"  Running {args.num_iterations} iterations with model refresh between each")
        print(f"{'='*80}")

    iteration_results = []

    for iteration in range(args.num_iterations):
        print(f"\n--- Iteration {iteration + 1}/{args.num_iterations} ---")

        print(f"Reloading model to clear cache...")
        llm = setup_llm(args)

        warmup_metrics = run_warmup(llm, prompt_text, sampling_params, args.num_warmup)

        # Run the benchmark with power monitoring
        power_monitor = PowerMonitor(gpu_index=0)
        metrics = benchmark_single_query(llm, prompt_text, sampling_params, power_monitor)

        # Add warmup metrics to the iteration results
        metrics['warmup_metrics'] = warmup_metrics
        iteration_results.append(metrics)

        # Clean up model
        del llm
        gc.collect()
        time.sleep(0.5)

    # Calculate averages across iterations
    avg_ttft = sum(r['ttft'] for r in iteration_results) / len(iteration_results)
    avg_tokens_per_sec = sum(r['tokens_per_sec'] for r in iteration_results) / len(iteration_results)
    avg_query_latency = sum(r['query_latency'] for r in iteration_results) / len(iteration_results)

    # Calculate average power metrics if available
    power_keys = ['avg_power', 'peak_power', 'avg_temp', 'peak_temp', 'avg_gpu_utilization', 'avg_memory_utilization']
    avg_power_metrics = {}
    for key in power_keys:
        values = [r.get(key) for r in iteration_results if key in r]
        if values:
            avg_power_metrics[key] = sum(values) / len(values)

    # Use the first iteration's metadata and generated text
    averaged_metrics = {
        'ttft': avg_ttft,
        'tokens_per_sec': avg_tokens_per_sec,
        'query_latency': avg_query_latency,
        'input_tokens': iteration_results[0]['input_tokens'],
        'output_tokens': iteration_results[0]['output_tokens'],
        'generated_text': iteration_results[0]['generated_text'],
        'prompt': prompt_text,
        'iterations': iteration_results,  # Store all iteration data
        'num_iterations': args.num_iterations,
        **avg_power_metrics  # Add averaged power metrics
    }

    # Add prompt-specific metadata
    averaged_metrics.update({k: v for k, v in prompt_data.items() if k != 'prompt'})

    print(f"\n{'='*80}")
    print(f"AVERAGED RESULTS FOR QUERY {idx}")
    print(f"{'='*80}")
    print(f"Average TTFT: {avg_ttft:.4f} s")
    print(f"Average Tokens/sec: {avg_tokens_per_sec:.2f}")
    print(f"Average Query Latency: {avg_query_latency:.4f} s")
    print(f"Input Tokens: {averaged_metrics['input_tokens']}")
    print(f"Output Tokens: {averaged_metrics['output_tokens']}")
    if avg_power_metrics:
        print(f"Average Power: {avg_power_metrics.get('avg_power', 0):.1f} W, Peak Power: {avg_power_metrics.get('peak_power', 0):.1f} W")
        print(f"Average Temp: {avg_power_metrics.get('avg_temp', 0):.1f} °C, Peak Temp: {avg_power_metrics.get('peak_temp', 0):.1f} °C")
    print(f"{'='*80}\n")

    return averaged_metrics


def main():
    args = parse_arguments()

    print(f"{'='*80}")
    print(f"Model: {args.model_dir}")
    print(f"Backend: vLLM")
    print(f"GPU Memory Utilization: {args.gpu_memory_utilization}")
    print(f"Iterations per prompt: {args.num_iterations}")
    print(f"{'='*80}\n")

    prompts = load_prompts(args)

    print("Loading model (initial)...")
    model_load_start = time.perf_counter()
    llm = setup_llm(args)
    model_load_time = time.perf_counter() - model_load_start
    print(f"Model loaded in {model_load_time:.2f} seconds\n")

    # Clean up initial model - it will be reloaded for each iteration
    del llm
    gc.collect()

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.7
    )

    print("Starting benchmark...")
    results = []

    for idx, prompt_data in enumerate(prompts, 1):
        metrics = run_prompt_iterations(args, prompt_data, idx, len(prompts), sampling_params)
        results.append(metrics)

    aggregate_metrics = calculate_aggregate_metrics(results)

    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"Number of queries: {aggregate_metrics['num_queries']}")
    print(f"Average TTFT: {aggregate_metrics['average_ttft']:.4f} s")
    print(f"Average Tokens/sec: {aggregate_metrics['average_tokens_per_sec']:.2f}")
    print(f"Average Latency: {aggregate_metrics['average_latency']:.4f} s")
    print(f"Total Output Tokens: {aggregate_metrics['total_output_tokens']}")
    print(f"Overall Throughput: {aggregate_metrics['overall_throughput']:.2f} tokens/sec")
    print(f"{'='*80}\n")

    output_data = {
        'model': args.model_dir,
        'backend': 'vllm',
        'model_load_time': model_load_time,
        'configuration': {
            'max_tokens': args.max_tokens,
            'temperature': 0.7,
            'gpu_memory_utilization': args.gpu_memory_utilization,
            'num_iterations': args.num_iterations,
            'gpu_index': 0,
        },
        'aggregate_metrics': aggregate_metrics,
        'per_query_results': [
            {
                'query_index': idx + 1,
                'filename': r.get('filename', None),
                'num_iterations': r.get('num_iterations', 1),
                'averaged_ttft': r['ttft'],
                'averaged_ttft_ms': r['ttft'] * 1000,
                'averaged_query_latency': r['query_latency'],
                'averaged_query_latency_ms': r['query_latency'] * 1000,
                'averaged_tokens_per_sec': r['tokens_per_sec'],
                'input_tokens': r['input_tokens'],
                'output_tokens': r['output_tokens'],
                'prompt': r['prompt'][:100] + '...' if len(r['prompt']) > 100 else r['prompt'],
                'generated_text': r['generated_text'][:100] + '...' if len(r['generated_text']) > 100 else r['generated_text'],
                # Power metrics (averaged across iterations)
                'avg_power': r.get('avg_power'),
                'peak_power': r.get('peak_power'),
                'avg_temp': r.get('avg_temp'),
                'peak_temp': r.get('peak_temp'),
                'avg_gpu_utilization': r.get('avg_gpu_utilization'),
                'avg_memory_utilization': r.get('avg_memory_utilization'),
                'individual_iterations': [
                    {
                        'iteration': i + 1,
                        'ttft': it['ttft'],
                        'ttft_ms': it['ttft'] * 1000,
                        'query_latency': it['query_latency'],
                        'query_latency_ms': it['query_latency'] * 1000,
                        'query_latency_manual': it.get('query_latency_manual'),
                        'query_latency_manual_ms': it.get('query_latency_manual', 0) * 1000,
                        'decode_time': it.get('decode_time'),
                        'decode_time_ms': it.get('decode_time', 0) * 1000,
                        'tokens_per_sec': it['tokens_per_sec'],
                        'input_tokens': it['input_tokens'],
                        'output_tokens': it['output_tokens'],
                        # Power metrics for this iteration
                        'avg_power': it.get('avg_power'),
                        'peak_power': it.get('peak_power'),
                        'avg_temp': it.get('avg_temp'),
                        'peak_temp': it.get('peak_temp'),
                        'avg_gpu_utilization': it.get('avg_gpu_utilization'),
                        'avg_memory_utilization': it.get('avg_memory_utilization'),
                        'warmup_metrics': it.get('warmup_metrics', [])
                    }
                    for i, it in enumerate(r.get('iterations', []))
                ] if 'iterations' in r else []
            }
            for idx, r in enumerate(results)
        ]
    }

    output_path = Path(args.output_file)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {output_path.absolute()}")


if __name__ == '__main__':
    main()
