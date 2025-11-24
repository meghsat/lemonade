## docker run --gpus all -it -v ~/phi35_quantization:/workspace/phi35   --name tensorrt_phi35_v3   nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev bash
## /app/tensorrt_llm/examples/llm-api# python benchmark_metrics_prompt_files_v2.py   --model_dir openai/gpt-oss-120b   --prompts_folder /app/tensorrt_llm/examples/llm-api/prompt_files   --output_file mlperf_results.json
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import asyncio

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Benchmark TensorRT-LLM models with detailed metrics")

    # Model configuration
    parser.add_argument('--model_dir', type=str, required=True,
                        help="Model checkpoint directory")
    parser.add_argument('--prompts_folder', type=str,
                        help="Path to folder containing individual prompt text files (mlperf_p1_in{isl}_out{osl}.txt)")

    # Generation parameters
    parser.add_argument('--max_tokens', type=int, default=2048,
                        help="Maximum number of tokens to generate")
    parser.add_argument('--temperature', type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument('--top_k', type=int, default=1,
                        help="Top-k sampling parameter")
    parser.add_argument('--top_p', type=float, default=1.0,
                        help="Top-p (nucleus) sampling parameter")

    # Build configuration
    parser.add_argument('--max_seq_len', type=int, default=4096,
                        help="Maximum sequence length")
    parser.add_argument('--max_batch_size', type=int, default=1,
                        help="Maximum batch size")
    parser.add_argument('--max_num_tokens', type=int, default=8192,
                        help="Maximum total tokens in a batch")

    # Parallelism
    parser.add_argument('--tp_size', type=int, default=1,
                        help="Tensor parallelism size")
    parser.add_argument('--pp_size', type=int, default=1,
                        help="Pipeline parallelism size")

    # KV cache
    parser.add_argument('--kv_cache_dtype', type=str, default='auto',
                        help="KV cache data type")
    parser.add_argument('--kv_cache_fraction', type=float, default=None,
                        help="Fraction of GPU memory for KV cache")

    # Benchmark settings
    parser.add_argument('--num_warmup', type=int, default=2,
                        help="Number of warmup iterations")
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
        pattern = re.compile(r'mlperf_p\d+_in(\d+)_out(\d+)\.txt')

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

        print(f"Loaded {len(prompt_data)} prompts from folder: {folder_path}")
        return prompt_data

    else:
        # Default prompts
        return [
            {'prompt': "Explain the concept of neural networks in simple terms."},
            {'prompt': "What are the main differences between supervised and unsupervised learning?"},
            {'prompt': "Describe how a transformer model works."},
        ]


def setup_llm(args) -> LLM:
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=True,
        free_gpu_memory_fraction=args.kv_cache_fraction,
        dtype=args.kv_cache_dtype,
    )

    llm = LLM(
        model=args.model_dir,
        backend='pytorch',
        kv_cache_config=kv_cache_config,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
        max_num_tokens=args.max_num_tokens,
        tensor_parallel_size=args.tp_size,
        pipeline_parallel_size=args.pp_size,
        trust_remote_code=args.trust_remote_code,
    )

    return llm


async def benchmark_single_query_async(llm: LLM, prompt: str, sampling_params: SamplingParams) -> Dict[str, Any]:

    # Count input tokens and tokenize
    input_ids = llm.tokenizer.encode(prompt)
    if hasattr(input_ids, 'input_ids'):
        input_tokens = len(input_ids.input_ids)
        token_ids = input_ids.input_ids.tolist() if hasattr(input_ids.input_ids, 'tolist') else list(input_ids.input_ids)
    else:
        input_tokens = len(input_ids)
        token_ids = input_ids.tolist() if hasattr(input_ids, 'tolist') else list(input_ids)

    # Timing tracking
    query_start = time.perf_counter()
    token_times = []
    generated_text = ""
    output_token_count = 0

    t1 = time.perf_counter()
    first_token_received = False
    ttft = 0.0

    print(f"\n[Generating] ", end='', flush=True)

    # Use async generator for streaming - pass tokenized input
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

    # Calculate metrics
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
        'prompt': prompt
    }

    print(f"\n{'='*80}")
    print(f"Prompt: {prompt[:100]}...")
    print(f"Generated: {generated_text[:100]}...")
    print(f"TTFT: {ttft*1000:.2f} ms")
    print(f"Query Latency: {query_latency*1000:.2f} ms")
    print(f"Input Tokens: {input_tokens}")
    print(f"Output Tokens: {output_token_count}")
    print(f"Tokens/sec (decode): {tokens_per_sec:.2f}")
    print(f"{'='*80}\n")

    return metrics


def run_warmup(llm: LLM, sampling_params: SamplingParams, num_warmup: int):
    warmup_prompt = "This is a warmup prompt to initialize the model."

    print(f"\nRunning {num_warmup} warmup iterations...")
    for i in range(num_warmup):
        print(f"Warmup {i+1}/{num_warmup}")
        _ = llm.generate([warmup_prompt], sampling_params=sampling_params)
    print("Warmup complete.\n")


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


async def main_async(args, llm, prompts, base_sampling_params):
    results = []

    for idx, prompt_data in enumerate(prompts, 1):
        prompt_text = prompt_data['prompt']

        # Use per-prompt max_tokens if available, otherwise use base params
        if 'max_tokens' in prompt_data:
            sampling_params = SamplingParams(
                max_tokens=prompt_data['max_tokens'],
                temperature=base_sampling_params.temperature,
                top_k=base_sampling_params.top_k,
                top_p=base_sampling_params.top_p,
            )
            print(f"\nProcessing query {idx}/{len(prompts)} (max_tokens={prompt_data['max_tokens']})...")
            if 'filename' in prompt_data:
                print(f"  File: {prompt_data['filename']}")
        else:
            sampling_params = base_sampling_params
            print(f"\nProcessing query {idx}/{len(prompts)}...")

        metrics = await benchmark_single_query_async(llm, prompt_text, sampling_params)

        metrics.update({k: v for k, v in prompt_data.items() if k != 'prompt'})

        results.append(metrics)

    return results


def main():
    args = parse_arguments()

    print(f"{'='*80}")
    print(f"Model: {args.model_dir}")
    print(f"{'='*80}\n")

    # Load prompts
    prompts = load_prompts(args)

    # Initialize model
    print("Loading model...")
    model_load_start = time.perf_counter()
    llm = setup_llm(args)
    model_load_time = time.perf_counter() - model_load_start
    print(f"Model loaded in {model_load_time:.2f} seconds\n")

    new_prompts = []
    for prompt_data in prompts:
        messages = [{"role": "user", "content": prompt_data['prompt']}]
        formatted = llm.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        new_prompt_data = prompt_data.copy()
        new_prompt_data['prompt'] = formatted
        new_prompts.append(new_prompt_data)
    prompts = new_prompts

    # Setup sampling parameters
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    # Run warmup
    if args.num_warmup > 0:
        run_warmup(llm, sampling_params, args.num_warmup)

    print("Starting benchmark...")
    try:
        results = asyncio.run(main_async(args, llm, prompts, sampling_params))
        streaming_mode = True
    except (AttributeError, TypeError) as e:
        print(f"Async streaming failed")
       
        streaming_mode = False

    # Calculate aggregate metrics
    aggregate_metrics = calculate_aggregate_metrics(results)

    # Print summary
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

    # Save results
    output_data = {
        'model': args.model_dir,
        'model_load_time': model_load_time,
        'streaming_mode': streaming_mode,
        'configuration': {
            'max_tokens': args.max_tokens,
            'temperature': args.temperature,
            'top_k': args.top_k,
            'top_p': args.top_p,
            'max_seq_len': args.max_seq_len,
            'max_batch_size': args.max_batch_size,
        },
        'aggregate_metrics': aggregate_metrics,
        'per_query_results': [
            {
                'query_index': idx + 1,
                'filename': r.get('filename', None),
                'expected_isl': r.get('isl', None),
                'expected_osl': r.get('osl', None),
                'ttft': r['ttft'],
                'ttft_ms': r['ttft'] * 1000,
                'query_latency': r['query_latency'],
                'query_latency_ms': r['query_latency'] * 1000,
                'input_tokens': r['input_tokens'],
                'output_tokens': r['output_tokens'],
                'tokens_per_sec': r['tokens_per_sec'],
                'prompt': r['prompt'][:100] + '...' if len(r['prompt']) > 100 else r['prompt'],
                'generated_text': r['generated_text'][:100] + '...' if len(r['generated_text']) > 100 else r['generated_text'],
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
