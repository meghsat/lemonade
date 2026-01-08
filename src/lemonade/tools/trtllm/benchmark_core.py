"""
This module contains the actual benchmark execution code that runs inside the Docker container.
It can be executed as a standalone script or imported as a module.
"""

import argparse
import asyncio
import time
import json
import gc
from pathlib import Path
from typing import List, Dict, Any


async def run_warmup(
    llm, prompt: str, sampling_params, num_warmup: int
) -> List[Dict[str, Any]]:
    input_ids = llm.tokenizer.encode(prompt)
    if hasattr(input_ids, "input_ids"):
        token_ids = (
            input_ids.input_ids.tolist()
            if hasattr(input_ids.input_ids, "tolist")
            else list(input_ids.input_ids)
        )
    else:
        token_ids = (
            input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
        )

    warmup_metrics = []
    print(f"\nRunning {num_warmup} warmup iterations with TTFT tracking...")

    for i in range(num_warmup):
        print(f"Warmup {i+1}/{num_warmup}: ", end="", flush=True)

        t1 = time.perf_counter()
        first_token_received = False
        ttft = 0.0
        output_token_count = 0

        async for output in llm.generate_async(
            token_ids, sampling_params=sampling_params, streaming=True
        ):
            t2 = time.perf_counter()

            if not first_token_received:
                ttft = t2 - t1
                first_token_received = True
                print(f"TTFT: {ttft*1000:.2f} ms", end="", flush=True)

            if output.outputs:
                output_token_count = len(output.outputs[0].token_ids)

            t1 = t2

        warmup_metrics.append(
            {
                "warmup_iteration": i + 1,
                "ttft": ttft,
                "ttft_ms": ttft * 1000,
                "output_tokens": output_token_count,
            }
        )

        print(f" - Total: {output_token_count} tokens")

    print("Warmup complete.\n")
    return warmup_metrics


async def benchmark_single_query_async(
    llm, prompt: str, sampling_params
) -> Dict[str, Any]:
    # Count input tokens and tokenize
    input_ids = llm.tokenizer.encode(prompt)
    if hasattr(input_ids, "input_ids"):
        input_tokens = len(input_ids.input_ids)
        token_ids = (
            input_ids.input_ids.tolist()
            if hasattr(input_ids.input_ids, "tolist")
            else list(input_ids.input_ids)
        )
    else:
        input_tokens = len(input_ids)
        token_ids = (
            input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
        )

    # Timing tracking
    query_start = time.perf_counter()
    token_times = []
    generated_text = ""
    output_token_count = 0

    t1 = time.perf_counter()
    first_token_received = False
    ttft = 0.0

    print(f"\n[Generating] ", end="", flush=True)

    async for output in llm.generate_async(
        token_ids, sampling_params=sampling_params, streaming=True
    ):
        t2 = time.perf_counter()

        if not first_token_received:
            ttft = t2 - t1
            first_token_received = True
            print(f"TTFT: {ttft*1000:.2f} ms")

        token_latency = t2 - t1
        token_times.append(token_latency)

        if output.outputs:
            new_text = output.outputs[0].text
            new_tokens = new_text[len(generated_text) :]
            print(new_tokens, end="", flush=True)
            generated_text = new_text
            output_token_count = len(output.outputs[0].token_ids)

        t1 = t2

    query_end = time.perf_counter()
    query_latency = query_end - query_start

    if output_token_count > 1 and query_latency > ttft:
        tokens_per_sec = (output_token_count - 1) / (query_latency - ttft)
    else:
        tokens_per_sec = output_token_count / query_latency if query_latency > 0 else 0

    metrics = {
        "ttft": ttft,
        "query_latency": query_latency,
        "input_tokens": input_tokens,
        "output_tokens": output_token_count,
        "tokens_per_sec": tokens_per_sec,
        "per_token_latencies": token_times,
        "generated_text": generated_text,
        "prompt": prompt,
    }

    print(f"\n{'='*80}")
    print(f"Prompt: {prompt[:100]}...")
    print(f"Generated: {generated_text[:100]}...")
    print(f"TTFT: {ttft*1000:.2f} ms")
    print(f"Query Latency: {query_latency*1000:.2f} ms")
    print(f"Output Tokens: {output_token_count}")
    print(f"Tokens/sec (decode): {tokens_per_sec:.2f}")
    print(f"{'='*80}\n")

    return metrics


async def run_prompt_iterations(
    llm,
    prompt_text: str,
    sampling_params,
    num_iterations: int,
    num_warmup: int,
    prompt_metadata: dict = None,
):
    """Run multiple iterations of a prompt with model reloading between iterations"""

    iteration_results = []

    for iteration in range(num_iterations):
        print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")

        # Run warmup for this iteration
        warmup_metrics = await run_warmup(llm, prompt_text, sampling_params, num_warmup)

        # Run the benchmark
        metrics = await benchmark_single_query_async(llm, prompt_text, sampling_params)

        # Add warmup metrics to the iteration results
        metrics["warmup_metrics"] = warmup_metrics
        iteration_results.append(metrics)

        # Small delay between iterations
        await asyncio.sleep(0.5)

    # Calculate averages across iterations
    avg_ttft = sum(r["ttft"] for r in iteration_results) / len(iteration_results)
    avg_tokens_per_sec = sum(r["tokens_per_sec"] for r in iteration_results) / len(
        iteration_results
    )
    avg_query_latency = sum(r["query_latency"] for r in iteration_results) / len(
        iteration_results
    )

    # Use the first iteration's metadata and generated text
    averaged_metrics = {
        "ttft": avg_ttft,
        "tokens_per_sec": avg_tokens_per_sec,
        "query_latency": avg_query_latency,
        "input_tokens": iteration_results[0]["input_tokens"],
        "output_tokens": iteration_results[0]["output_tokens"],
        "generated_text": iteration_results[0]["generated_text"],
        "prompt": prompt_text,
        "per_token_latencies": iteration_results[0]["per_token_latencies"],
        "iterations": iteration_results,
        "num_iterations": num_iterations,
    }

    # Add prompt-specific metadata
    if prompt_metadata:
        averaged_metrics.update(prompt_metadata)

    print(f"\n{'='*80}")
    print(f"AVERAGED RESULTS")
    print(f"{'='*80}")
    print(f"Average TTFT: {avg_ttft*1000:.2f} ms")
    print(f"Average Tokens/sec: {avg_tokens_per_sec:.2f}")
    print(f"Average Query Latency: {avg_query_latency*1000:.2f} ms")
    print(f"Input Tokens: {averaged_metrics['input_tokens']}")
    print(f"Output Tokens: {averaged_metrics['output_tokens']}")
    print(f"{'='*80}\n")

    return averaged_metrics


def setup_llm(
    model_path: str,
    max_seq_len: int = 4096,
    max_num_tokens: int = 8192,
    trust_remote_code: bool = False,
):
    """
    Setup TensorRT-LLM model

    This is a simplified version that runs inside the container.
    """
    try:
        from tensorrt_llm import LLM, SamplingParams
        from tensorrt_llm.llmapi import KvCacheConfig
    except ImportError as e:
        raise ImportError(
            "TensorRT-LLM not available. Make sure you're running inside the TensorRT-LLM container."
        ) from e

    kv_cache_config = KvCacheConfig(
        enable_block_reuse=True,
        free_gpu_memory_fraction=None,
        dtype="auto",
    )

    llm = LLM(
        model=model_path,
        backend="pytorch",
        kv_cache_config=kv_cache_config,
        max_seq_len=max_seq_len,
        max_batch_size=1,
        max_num_tokens=max_num_tokens,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        trust_remote_code=trust_remote_code,
    )

    return llm


def calculate_aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate aggregate metrics across all prompts"""
    if not results:
        return {}

    ttfts = [r["ttft"] for r in results]
    latencies = [r["query_latency"] for r in results]
    tps_values = [r["tokens_per_sec"] for r in results]
    input_tokens = [r["input_tokens"] for r in results]
    output_tokens = [r["output_tokens"] for r in results]

    total_output_tokens = sum(output_tokens)
    total_time = sum(latencies)

    aggregate = {
        "num_queries": len(results),
        "average_ttft": sum(ttfts) / len(ttfts),
        "min_ttft": min(ttfts),
        "max_ttft": max(ttfts),
        "average_tokens_per_sec": sum(tps_values) / len(tps_values),
        "min_tokens_per_sec": min(tps_values),
        "max_tokens_per_sec": max(tps_values),
        "average_latency": sum(latencies) / len(latencies),
        "total_latency": total_time,
        "average_input_tokens": sum(input_tokens) / len(input_tokens),
        "average_output_tokens": sum(output_tokens) / len(output_tokens),
        "total_output_tokens": total_output_tokens,
        "overall_throughput": total_output_tokens / total_time if total_time > 0 else 0,
    }

    return aggregate


async def run_benchmark(
    model_path: str,
    prompts: List[Dict[str, Any]],
    num_iterations: int = 3,
    num_warmup: int = 2,
    max_tokens: int = 512,
    max_seq_len: int = 4096,
    max_num_tokens: int = 8192,
    trust_remote_code: bool = False,
    output_file: str = None,
) -> Dict[str, Any]:
    """
    Main benchmark function

    Args:
        model_path: Path to model checkpoint
        prompts: List of prompt dicts with 'prompt' and optional 'max_tokens', 'filename' keys
        num_iterations: Number of iterations per prompt
        num_warmup: Number of warmup runs
        max_tokens: Default max output tokens
        max_seq_len: Max sequence length
        max_num_tokens: Max num tokens
        trust_remote_code: Trust remote code
        output_file: Optional path to save results JSON

    Returns:
        Results dictionary
    """
    try:
        from tensorrt_llm import SamplingParams
    except ImportError as e:
        raise ImportError(
            "TensorRT-LLM not available. Make sure you're running inside the TensorRT-LLM container."
        ) from e

    print(f"{'='*80}")
    print(f"Model: {model_path}")
    print(f"Iterations per prompt: {num_iterations}")
    print(f"{'='*80}\n")

    # Load model initially
    print("Loading model (initial)...")
    model_load_start = time.perf_counter()
    llm = setup_llm(model_path, max_seq_len, max_num_tokens, trust_remote_code)
    model_load_time = time.perf_counter() - model_load_start
    print(f"Model loaded in {model_load_time:.2f} seconds\n")

    # Base sampling params
    base_sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.7,
    )

    # Run benchmarks for all prompts
    results = []
    for idx, prompt_data in enumerate(prompts, 1):
        prompt_text = prompt_data["prompt"]
        prompt_max_tokens = prompt_data.get("max_tokens", max_tokens)

        sampling_params = SamplingParams(
            max_tokens=prompt_max_tokens,
            temperature=0.7,
        )

        print(f"\n{'='*80}")
        print(f"Processing query {idx}/{len(prompts)} (max_tokens={prompt_max_tokens})")
        if "filename" in prompt_data:
            print(f"  File: {prompt_data['filename']}")
        print(f"{'='*80}")

        # Extract metadata
        metadata = {
            k: v for k, v in prompt_data.items() if k not in ["prompt", "max_tokens"]
        }

        metrics = await run_prompt_iterations(
            llm, prompt_text, sampling_params, num_iterations, num_warmup, metadata
        )
        results.append(metrics)

    # Calculate aggregate metrics
    aggregate_metrics = calculate_aggregate_metrics(results)

    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"Number of queries: {aggregate_metrics['num_queries']}")
    print(f"Average TTFT: {aggregate_metrics['average_ttft']*1000:.2f} ms")
    print(f"Average Tokens/sec: {aggregate_metrics['average_tokens_per_sec']:.2f}")
    print(f"Average Latency: {aggregate_metrics['average_latency']*1000:.2f} ms")
    print(f"Total Output Tokens: {aggregate_metrics['total_output_tokens']}")
    print(
        f"Overall Throughput: {aggregate_metrics['overall_throughput']:.2f} tokens/sec"
    )
    print(f"{'='*80}\n")

    # Prepare output data
    output_data = {
        "model": model_path,
        "model_load_time": model_load_time,
        "streaming_mode": True,
        "configuration": {
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "max_seq_len": max_seq_len,
            "max_batch_size": 1,
            "num_iterations": num_iterations,
        },
        "aggregate_metrics": aggregate_metrics,
        "per_query_results": [
            {
                "query_index": idx + 1,
                "filename": r.get("filename", None),
                "num_iterations": r.get("num_iterations", 1),
                "averaged_ttft": r["ttft"],
                "averaged_ttft_ms": r["ttft"] * 1000,
                "averaged_query_latency": r["query_latency"],
                "averaged_query_latency_ms": r["query_latency"] * 1000,
                "averaged_tokens_per_sec": r["tokens_per_sec"],
                "input_tokens": r["input_tokens"],
                "output_tokens": r["output_tokens"],
                "prompt": (
                    r["prompt"][:100] + "..." if len(r["prompt"]) > 100 else r["prompt"]
                ),
                "generated_text": (
                    r["generated_text"][:100] + "..."
                    if len(r["generated_text"]) > 100
                    else r["generated_text"]
                ),
                "individual_iterations": (
                    [
                        {
                            "iteration": i + 1,
                            "ttft": it["ttft"],
                            "ttft_ms": it["ttft"] * 1000,
                            "query_latency": it["query_latency"],
                            "query_latency_ms": it["query_latency"] * 1000,
                            "tokens_per_sec": it["tokens_per_sec"],
                            "input_tokens": it["input_tokens"],
                            "output_tokens": it["output_tokens"],
                            "warmup_metrics": it.get("warmup_metrics", []),
                        }
                        for i, it in enumerate(r.get("iterations", []))
                    ]
                    if "iterations" in r
                    else []
                ),
            }
            for idx, r in enumerate(results)
        ],
    }

    # Save to file if specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to: {output_path.absolute()}")

    return output_data


# Copyright (c) 2025 AMD
