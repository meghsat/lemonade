import os
import re
import subprocess
import sys
import datetime
import platform
import csv
import statistics
import math

if len(sys.argv) < 7 or len(sys.argv) > 8:
    print(
        "Usage: python script.py <vendor> <model [ hf_checkpoint | directory ]> <prompt_dir_path> <prompt_prefix [ mlperf | textgen | phi | llama3 ]> <iterations> <warmups> [backend]"
    )
    print("  backend (optional): llamacpp (default), vllm, trtllm, mlx, openvino, oga")
    print("  Hardware-specific backends:")
    print("    - trtllm: NVIDIA only")
    print("    - mlx: APPLE only")
    print("    - openvino: INTEL only")
    print("    - oga: AMD only")
    sys.exit(1)

VENDOR = sys.argv[1].upper()
MODEL_PATH = sys.argv[2]
PROMPTS_PATH = sys.argv[3]
FILE_PREFIX = sys.argv[4]
ITERATIONS = sys.argv[5]
WARMUPS = sys.argv[6]
BACKEND = sys.argv[7].lower() if len(sys.argv) == 8 else "llamacpp"

# Validate backend compatibility with vendor
BACKEND_VENDOR_MAP = {
    "trtllm": "NVIDIA",
    "mlx": "APPLE",
    "openvino": "INTEL",
    "oga": "AMD",
}

if BACKEND in BACKEND_VENDOR_MAP:
    required_vendor = BACKEND_VENDOR_MAP[BACKEND]
    if VENDOR != required_vendor:
        print(
            f"Error: Backend '{BACKEND}' is only supported on {required_vendor} hardware."
        )
        print(f"Current vendor: {VENDOR}")
        sys.exit(1)

# Extract model name from path
if os.path.isdir(MODEL_PATH):
    model_name = os.path.basename(MODEL_PATH.rstrip(os.sep))
else:
    # Extract model name from HF checkpoint (e.g., "org/model-name" -> "model-name")
    model_name = MODEL_PATH.split("/")[-1]

# Generate unique cache directory with naming scheme: vendor_modelname_backend_promptprefix_timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
cache_folder_name = f"{VENDOR}_{model_name}_{BACKEND}_{FILE_PREFIX}_{timestamp}"
CACHE_PATH = os.path.abspath(cache_folder_name)
os.makedirs(CACHE_PATH, exist_ok=True)
print(f"Cache directory: {CACHE_PATH}")

FILENAME_PATTERN = re.compile(rf"^{FILE_PREFIX}_p(\d+)(?:_in\d+)?_out(\d+)\.txt$")


def parse_filename(filename):
    match = FILENAME_PATTERN.match(filename)
    if match:
        part_number = int(match.group(1))
        out_value = int(match.group(2))
        return part_number, out_value
    return None


def get_model_from_path(hf_path: str) -> str:
    """
    Returns the local model path given either:
      - a directory path (used as-is), or
      - a Hugging Face checkpoint.

    Behavior depends on the VENDOR variable:
      - INTEL → export with optimum-cli (unless already exported).
      - AMD   → if hf_path is a checkpoint, just return it (no export).
    """
    model_path = hf_path

    if os.path.isdir(hf_path):
        print(f"Detected directory: {hf_path} → using as-is")
    else:
        checkpoint = hf_path
        output_dir = f"{checkpoint.split('/')[-1]}_ov-int4-CHw"

        if VENDOR == "INTEL":
            if os.path.exists(output_dir):
                print(f"Model already exported at {output_dir} → skipping export")
            else:
                cmd = [
                    "optimum-cli",
                    "export",
                    "openvino",
                    "-m",
                    checkpoint,
                    "--weight-format",
                    "int4",
                    "--sym",
                    "--group-size",
                    "-1",
                    "--ratio",
                    "1.0",
                    "--all-layers",
                    "--awq",
                    "--scale-estimation",
                    "--dataset=wikitext2",
                    output_dir,
                ]
                print(f"[INTEL] Running command: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
            model_path = output_dir

        elif VENDOR == "AMD":
            print(f"[AMD] Using Hugging Face checkpoint directly: {checkpoint}")
            model_path = checkpoint
        elif VENDOR == "NVIDIA":
            print(f"[NVIDIA] Using Hugging Face checkpoint directly: {checkpoint}")
            model_path = checkpoint

        elif VENDOR == "APPLE":
            print(f"[APPLE] Using Hugging Face checkpoint directly: {checkpoint}")
            model_path = checkpoint

        else:
            raise ValueError(
                f"Unsupported or missing VENDOR '{VENDOR}'. Expected 'INTEL', 'AMD', 'NVIDIA', or 'APPLE'."
            )

    return model_path


def main():
    tasks = []

    for fname in os.listdir(PROMPTS_PATH):
        parsed = parse_filename(fname)
        if parsed:
            part_num, out_value = parsed
            tasks.append((part_num, out_value, fname))

    tasks.sort(key=lambda x: x[0])
    print(tasks)
    failed_tasks = []

    model_path = get_model_from_path(MODEL_PATH)

    for part_num, out_value, fname in tasks:
        full_path = os.path.join(PROMPTS_PATH, fname)

        try:
            if VENDOR == "NVIDIA":
                print(f"Running lemonade with prompt file: {fname}")

                if BACKEND == "trtllm":
                    cmd = [
                        "lemonade",
                        "-d",
                        CACHE_PATH,
                        "-i",
                        model_path,
                        "--power-nvidia",
                        "--memory",
                        "trtllm-load",
                        "--device",
                        "cuda",
                        "trtllm-bench",
                        "--iterations",
                        str(ITERATIONS),
                        "--warmup-iterations",
                        str(WARMUPS),
                        "--prompts",
                        full_path,
                        "--output-tokens",
                        str(out_value),
                        "--prompt-label",
                        fname,
                    ]
                #  elif BACKEND == "vllm":

                else:  # default llamacpp
                    cmd = [
                        "lemonade",
                        "-d",
                        CACHE_PATH,
                        "-i",
                        model_path,
                        "--power-nvidia",
                        "--memory",
                        "llamacpp-load",
                        "--device",
                        "igpu",
                        "llamacpp-bench",
                        "--cli",  # Required for text prompts!
                        "--iterations",
                        str(ITERATIONS),
                        "--warmup-iterations",
                        str(WARMUPS),
                        "--prompts",
                        full_path,
                        "--output-tokens",
                        str(out_value),
                        "--prompt-label",
                        fname,
                    ]

            elif VENDOR == "AMD":
                print(
                    f"Running: lemonade -i {MODEL_PATH} -d {CACHE_PATH} ... with {full_path}"
                )
                if BACKEND == "oga":
                    cmd = [
                        "lemonade",
                        "-d",
                        CACHE_PATH,
                        "-i",
                        model_path,
                        "--power-agt",
                        "oga-load",
                        "--device",
                        "hybrid",
                        "--dtype",
                        "int4",
                        "oga-bench",
                        "--prompts",
                        full_path,
                        "--iterations",
                        str(ITERATIONS),
                        "--warmup-iterations",
                        str(WARMUPS),
                        "--output-tokens",
                        str(out_value),
                    ]
                #   elif BACKEND == "vllm":
                else:  # default llamacpp
                    cmd = [
                        "lemonade",
                        "-d",
                        CACHE_PATH,
                        "-i",
                        model_path,
                        "--power-agt",
                        "--memory",
                        "llamacpp-load",
                        "--device",
                        "igpu",
                        "llamacpp-bench",
                        "--cli",
                        "--prompts",
                        full_path,
                        "--iterations",
                        str(ITERATIONS),
                        "--warmup-iterations",
                        str(WARMUPS),
                        "--output-tokens",
                        str(out_value),
                        "--prompt-label",
                        fname,
                    ]
            elif VENDOR == "INTEL":
                print(
                    f"Running: lemonade -i {MODEL_PATH} -d {CACHE_PATH} ... with {full_path}"
                )
                if BACKEND == "openvino":
                    cmd = [
                        "lemonade",
                        "-d",
                        CACHE_PATH,
                        "-i",
                        model_path,
                        "openvino-load",
                        "--device",
                        "NPU",
                        "-bp",
                        full_path,
                        "-r",
                        str(out_value),
                        "openvino-bench",
                        "--iterations",
                        str(ITERATIONS),
                        "--warmup-iterations",
                        str(WARMUPS),
                        "--output-tokens",
                        str(out_value),
                        "--prompts",
                        full_path,
                    ]
                # elif BACKEND == "vllm":

                else:  # default llamacpp
                    cmd = [
                        "lemonade",
                        "-d",
                        CACHE_PATH,
                        "-i",
                        model_path,
                        "--power-hwinfo",
                        "--memory",
                        "llamacpp-load",
                        "--device",
                        "igpu",
                        "llamacpp-bench",
                        "--cli",
                        "--prompts",
                        full_path,
                        "--iterations",
                        str(ITERATIONS),
                        "--warmup-iterations",
                        str(WARMUPS),
                        "--output-tokens",
                        str(out_value),
                        "--prompt-label",
                        fname,
                    ]

            elif VENDOR == "APPLE":
                print(
                    f"Running: lemonade -i {MODEL_PATH} -d {CACHE_PATH} ... with {full_path}"
                )
                if BACKEND == "mlx":
                    cmd = [
                        "lemonade",
                        "-d",
                        CACHE_PATH,
                        "-i",
                        model_path,
                        "--power-apple",
                        "--memory",
                        "mlx-load",
                        "--device",
                        "gpu",
                        "mlx-bench",
                        "--iterations",
                        str(ITERATIONS),
                        "--warmup-iterations",
                        str(WARMUPS),
                        "--prompts",
                        full_path,
                        "--output-tokens",
                        str(out_value),
                        "--prompt-label",
                        fname,
                    ]
                #  elif BACKEND == "vllm":

                else:  # default llamacpp
                    cmd = [
                        "lemonade",
                        "-d",
                        CACHE_PATH,
                        "-i",
                        model_path,
                        "--power-apple",
                        "llamacpp-load",
                        "--device",
                        "igpu",
                        "llamacpp-bench",
                        "--cli",
                        "--iterations",
                        str(ITERATIONS),
                        "--warmup-iterations",
                        str(WARMUPS),
                        "--prompts",
                        full_path,
                        "--output-tokens",
                        str(out_value),
                        "--prompt-label",
                        fname,
                    ]

            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

            print(f"✓ Completed {fname}")

        except subprocess.CalledProcessError as e:
            print(f"Failed on {fname} (part {part_num}): {e}")
            failed_tasks.append(fname)
            continue

    csv_path = os.path.join(CACHE_PATH, "benchmark_results.csv")
    if os.path.exists(csv_path):
        print(f"\nCSV results saved to: {csv_path}")
        print(f"  Total prompts processed: {len(tasks)}")
        print(f"  Successful: {len(tasks) - len(failed_tasks)}")
        print(f"  Failed: {len(failed_tasks)}")

    # Only generate report if at least one run succeeded
    if len(failed_tasks) < len(tasks):
        print(f"\nGenerating report from cache: {CACHE_PATH}")
        try:
            subprocess.run(
                [
                    "lemonade",
                    "report",
                    "-i",
                    CACHE_PATH,
                    "--no-save",
                    "--perf",
                    "--lean",
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Report generation failed: {e}")
    else:
        print("Skipping report: all tasks failed.")

    if failed_tasks:
        print("\nSummary of failed tasks:")
        for f in failed_tasks:
            print(f" - {f}")


if __name__ == "__main__":
    main()
