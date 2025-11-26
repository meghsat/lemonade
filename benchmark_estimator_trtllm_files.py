## docker run --gpus all -it -v ~/phi35_quantization:/workspace/phi35   --name tensorrt_phi35_v4   nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev bash
## /app/tensorrt_llm/examples/llm-api# python benchmark_estimator_trtllm_files.py   --model_dir openai/gpt-oss-20b   --prompts_folder /workspace/phi35/prompt_files   --output_file mlperf_results.json
## trtllm-bench --model openai/gpt-oss-20b throughput --dataset /workspace/phi35/prompts.jsonl --backend pytorch --streaming
import argparse
import json
import time
import gc
from pathlib import Path
from typing import List, Dict, Any
import asyncio
import threading
import subprocess
import pandas as pd

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig


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
        """Calculate power statistics from collected data."""
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
                        help="Model checkpoint directory")
    parser.add_argument('--prompts_folder', type=str,
                        help="Path to folder containing individual prompt text files (mlperf_p1_in{isl}_out{osl}.txt)")

    parser.add_argument('--max_tokens', type=int, default=2048,
                        help="Maximum number of tokens to generate")

    parser.add_argument('--max_seq_len', type=int, default=4096,
                        help="Maximum sequence length")
    parser.add_argument('--max_num_tokens', type=int, default=8192,
                        help="Maximum total tokens in a batch")

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

        # Pattern: mlperf_p1_in{isl}_out{osl}.txt
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
            raise ValueError(f"No valid prompt files found in {folder_path} matching pattern mlperf_p*_in*_out*.txt")

        return prompt_data

    else:
        return [
            {'prompt': "Explain the concept of neural networks in simple terms."},
            {'prompt': "What are the main differences between supervised and unsupervised learning?"},
            {'prompt': "Describe how a transformer model works."},
        ]


def setup_llm(args) -> LLM:
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=True,
        free_gpu_memory_fraction=None,
        dtype="auto",
    )

    llm = LLM(
        model=args.model_dir,
        backend='pytorch',
        kv_cache_config=kv_cache_config,
        max_seq_len=args.max_seq_len,
        max_batch_size=1,
        max_num_tokens=args.max_num_tokens,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        trust_remote_code=args.trust_remote_code,
    )

    return llm


async def benchmark_single_query_async(llm: LLM, prompt: str, sampling_params: SamplingParams, power_monitor: PowerMonitor = None) -> Dict[str, Any]:

    # Count input tokens and tokenize
    input_ids = llm.tokenizer.encode(prompt)
    if hasattr(input_ids, 'input_ids'):
        input_tokens = len(input_ids.input_ids)
        token_ids = input_ids.input_ids.tolist() if hasattr(input_ids.input_ids, 'tolist') else list(input_ids.input_ids)
    else:
        input_tokens = len(input_ids)
        token_ids = input_ids.tolist() if hasattr(input_ids, 'tolist') else list(input_ids)

    # Start power monitoring if provided
    if power_monitor:
        power_monitor.start()

    # Timing tracking
    query_start = time.perf_counter()
    token_times = []
    generated_text = ""
    output_token_count = 0

    t1 = time.perf_counter()
    first_token_received = False
    ttft = 0.0

    print(f"\n[Generating] ", end='', flush=True)

    async for output in llm.generate_async(token_ids, sampling_params=sampling_params, streaming=True):
        t2 = time.perf_counter()

        if not first_token_received:
            ttft = t2 - t1
            first_token_received = True
            print(f"TTFT: {ttft*1000:.2f} ms")

        token_latency = t2 - t1
        token_times.append(token_latency)

        if output.outputs:
            new_text = output.outputs[0].text
            new_tokens = new_text[len(generated_text):]
            print(new_tokens, end='', flush=True)
            generated_text = new_text
            output_token_count = len(output.outputs[0].token_ids)

        t1 = t2

    query_end = time.perf_counter()
    query_latency = query_end - query_start

    # Stop power monitoring and get stats
    power_stats = {}
    if power_monitor:
        power_monitor.stop()
        power_stats = power_monitor.get_stats()

    if output_token_count > 1 and query_latency > ttft:
        tokens_per_sec = (output_token_count - 1) / (query_latency - ttft)
    else:
        tokens_per_sec = output_token_count / query_latency if query_latency > 0 else 0

    metrics = {
        'ttft': ttft,
        'query_latency': query_latency,
        'input_tokens': input_tokens,
        'output_tokens': output_token_count,
        'tokens_per_sec': tokens_per_sec,
        'per_token_latencies': token_times,
        'generated_text': generated_text,
        'prompt': prompt,
        **power_stats  # Add power metrics
    }

    print(f"\n{'='*80}")
    print(f"Prompt: {prompt[:100]}...")
    print(f"Generated: {generated_text[:100]}...")
    print(f"TTFT: {ttft*1000:.2f} ms")
    print(f"All the tokens: {token_times}")
    print(f"Query Latency: {query_latency*1000:.2f} ms")
    print(f"Output Tokens: {output_token_count}")
    print(f"Tokens/sec (decode): {tokens_per_sec:.2f}")
    if power_stats:
        print(f"Avg Power: {power_stats.get('avg_power', 0):.1f} W, Peak Power: {power_stats.get('peak_power', 0):.1f} W")
        print(f"Avg Temp: {power_stats.get('avg_temp', 0):.1f} °C, Peak Temp: {power_stats.get('peak_temp', 0):.1f} °C")
    print(f"{'='*80}\n")

    return metrics


async def run_warmup(llm: LLM, prompt: str, sampling_params: SamplingParams, num_warmup: int) -> List[Dict[str, Any]]:
    prompt = "This is a warmup prompt to initialize the model."
    input_ids = llm.tokenizer.encode(prompt)
    if hasattr(input_ids, 'input_ids'):
        token_ids = input_ids.input_ids.tolist() if hasattr(input_ids.input_ids, 'tolist') else list(input_ids.input_ids)
    else:
        token_ids = input_ids.tolist() if hasattr(input_ids, 'tolist') else list(input_ids)

    warmup_metrics = []
    print(f"\nRunning {num_warmup} warmup iterations with TTFT tracking...")

    for i in range(num_warmup):
        print(f"Warmup {i+1}/{num_warmup}: ", end='', flush=True)

        t1 = time.perf_counter()
        first_token_received = False
        ttft = 0.0
        output_token_count = 0

        # Use async generator for streaming to capture TTFT
        async for output in llm.generate_async(token_ids, sampling_params=sampling_params, streaming=True):
            t2 = time.perf_counter()

            if not first_token_received:
                ttft = t2 - t1
                first_token_received = True
                print(f"TTFT: {ttft*1000:.2f} ms", end='', flush=True)

            if output.outputs:
                output_token_count = len(output.outputs[0].token_ids)

            t1 = t2

        warmup_end = time.perf_counter()
        total_time = warmup_end - (time.perf_counter() - (warmup_end - (t1 - time.perf_counter())))

        warmup_metrics.append({
            'warmup_iteration': i + 1,
            'ttft': ttft,
            'ttft_ms': ttft * 1000,
            'output_tokens': output_token_count
        })
        # del llm
        # gc.collect()
        # await asyncio.sleep(0.5)

        print(f" - Total: {output_token_count} tokens")

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


async def run_prompt_iterations(args, prompt_data, idx, total_prompts, base_sampling_params):
    prompt_text = prompt_data['prompt']

    # Use per-prompt max_tokens if available, otherwise use base params
    if 'max_tokens' in prompt_data:
        sampling_params = SamplingParams(
            max_tokens=prompt_data['max_tokens'],
            temperature=base_sampling_params.temperature,
            top_k=base_sampling_params.top_k,
            top_p=base_sampling_params.top_p,
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

        # Reload model to clear cache
        print(f"Reloading model to clear cache...")
        llm = setup_llm(args)

        #Run warmup for this iteration and capture warmup metrics (using the actual prompt)
        warmup_metrics = await run_warmup(llm, prompt_text, sampling_params, 2)
        # del llm
        # gc.collect()
        # await asyncio.sleep(0.5)

        #llm = setup_llm(args)
        # Run the benchmark with power monitoring
        power_monitor = PowerMonitor(gpu_index=0)
        metrics = await benchmark_single_query_async(llm, prompt_text, sampling_params, power_monitor)

        # Add warmup metrics to the iteration results
        metrics['warmup_metrics'] = warmup_metrics
        iteration_results.append(metrics)

        # Clean up model
        del llm
        gc.collect()
        await asyncio.sleep(0.5)

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
        'per_token_latencies': iteration_results[0]['per_token_latencies'],
        'iterations': iteration_results,  # Store all iteration data
        'num_iterations': args.num_iterations,
        **avg_power_metrics  # Add averaged power metrics
    }

    # Add prompt-specific metadata
    averaged_metrics.update({k: v for k, v in prompt_data.items() if k != 'prompt'})

    print(f"\n{'='*80}")
    print(f"AVERAGED RESULTS FOR QUERY {idx}")
    print(f"{'='*80}")
    print(f"Average TTFT: {avg_ttft*1000:.2f} ms")
    print(f"Average Tokens/sec: {avg_tokens_per_sec:.2f}")
    print(f"Average Query Latency: {avg_query_latency*1000:.2f} ms")
    print(f"Input Tokens: {averaged_metrics['input_tokens']}")
    print(f"Output Tokens: {averaged_metrics['output_tokens']}")
    if avg_power_metrics:
        print(f"Average Power: {avg_power_metrics.get('avg_power', 0):.1f} W, Peak Power: {avg_power_metrics.get('peak_power', 0):.1f} W")
        print(f"Average Temp: {avg_power_metrics.get('avg_temp', 0):.1f} °C, Peak Temp: {avg_power_metrics.get('peak_temp', 0):.1f} °C")
    print(f"{'='*80}\n")

    return averaged_metrics


async def main_async(args, prompts, base_sampling_params):
    results = []

    for idx, prompt_data in enumerate(prompts, 1):
        metrics = await run_prompt_iterations(args, prompt_data, idx, len(prompts), base_sampling_params)
        results.append(metrics)

    return results


def main():
    args = parse_arguments()

    print(f"{'='*80}")
    print(f"Model: {args.model_dir}")
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
        temperature=1.0,
        top_k=1,
        top_p=1.0,
    )

    print("Starting benchmark...")
    try:
        results = asyncio.run(main_async(args, prompts, sampling_params))
        streaming_mode = True
    except (AttributeError, TypeError) as e:
        print(f"Async streaming failed: {e}")

        streaming_mode = False

    aggregate_metrics = calculate_aggregate_metrics(results)

    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"Number of queries: {aggregate_metrics['num_queries']}")
    print(f"Average TTFT: {aggregate_metrics['average_ttft']*1000:.2f} ms ")
    print(f"Average Tokens/sec: {aggregate_metrics['average_tokens_per_sec']:.2f}")
    print(f"Average Latency: {aggregate_metrics['average_latency']*1000:.2f} ms")
    print(f"Total Output Tokens: {aggregate_metrics['total_output_tokens']}")
    print(f"Overall Throughput: {aggregate_metrics['overall_throughput']:.2f} tokens/sec")
    print(f"{'='*80}\n")

    output_data = {
        'model': args.model_dir,
        'model_load_time': model_load_time,
        'streaming_mode': streaming_mode,
        'configuration': {
            'max_tokens': args.max_tokens,
            'temperature': 1.0,
            'top_k': 1,
            'top_p': 1.0,
            'max_seq_len': args.max_seq_len,
            'max_batch_size': 1,
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
