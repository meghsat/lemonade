import os
import re
import subprocess
import sys
import datetime
import platform

if len(sys.argv) != 8:
    print("Usage: python script.py <vendor> <model [ hf_checkpoint | directory ]> <prompt_dir_path> <cache_base_path> <prompt_prefix [ mlperf | textgen ]> <iterations> <warmups>")
    sys.exit(1)

VENDOR = sys.argv[1]
MODEL_PATH = sys.argv[2]
PROMPTS_PATH = sys.argv[3]
CACHE_BASE = sys.argv[4]
FILE_PREFIX = sys.argv[5]
ITERATIONS = sys.argv[6]
WARMUPS = sys.argv[7]

# Generate unique cache directory using timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CACHE_PATH = os.path.join(CACHE_BASE, f"run_{timestamp}")
os.makedirs(CACHE_PATH, exist_ok=True)

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
                    "optimum-cli", "export", "openvino",
                    "-m", checkpoint,
                    "--weight-format", "int4",
                    "--sym",
                    "--group-size", "-1",
                    "--ratio", "1.0",
                    "--all-layers",
                    "--awq",
                    "--scale-estimation",
                    "--dataset=wikitext2",
                    output_dir
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

        else:
            raise ValueError(
                f"Unsupported or missing VENDOR '{VENDOR}'. Expected 'INTEL' or 'AMD' or 'NVIDIA'."
            )

    return model_path

def is_nvidia_arm64():
    """Check if running on ARM64 architecture with Nvidia GPU"""
    is_arm64 = platform.machine().lower() in ['aarch64', 'arm64']

    # Check for nvidia-smi availability
    has_nvidia = False
    try:
        result = subprocess.run(['nvidia-smi'],
                              capture_output=True,
                              timeout=5)
        has_nvidia = result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return is_arm64 and has_nvidia

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

    # Detect if we're on Nvidia ARM64 platform
    use_llamacpp_with_prompts = is_nvidia_arm64()

    if use_llamacpp_with_prompts:
        print(f"Detected Nvidia ARM64 platform - using llamacpp with prompt files")

    for part_num, out_value, fname in tasks:
        full_path = os.path.join(PROMPTS_PATH, fname)

        try:
            if use_llamacpp_with_prompts:
                # Read prompt file content
                with open(full_path, 'r', encoding='utf-8') as f:
                    prompt_content = f.read().strip()

                # Check if the prompt already has quotes (like mlperf files)
                # If not, we need to keep newlines as-is (don't escape them)
                # The prompt content will be passed as a single argument to subprocess

                print(f"Running lemonade with prompt file: {fname}")

                # Use llamacpp-bench with prompt content (requires --cli flag for text prompts)
                cmd = [
                    "lemonade", "-d", CACHE_PATH,
                    "-i", model_path,
                    "--power-nvidia", "--memory",
                    "llamacpp-load", "--device", "igpu",
                    "llamacpp-bench",
                    "--cli",  # Required for text prompts!
                    "--iterations", str(ITERATIONS),
                    "--warmup-iterations", str(WARMUPS),
                    "--prompts", full_path,
                    "--output-tokens", str(out_value),
                    "--prompt-label", fname
                ]

            elif VENDOR == "AMD":
                print(f"Running: lemonade -i {MODEL_PATH} -d {CACHE_PATH} ... with {full_path}")
                cmd = [
                    "lemonade", "-d", CACHE_PATH,
                    "-i", model_path,
                    "oga-load", "--device", "hybrid", "--dtype", "int4",
                    "oga-bench", "--prompts", full_path,
                    "--iterations", str(ITERATIONS), "--warmup-iterations", str(WARMUPS),
                    "--output-tokens", str(out_value)
                ]
            elif VENDOR == "INTEL":
                print(f"Running: lemonade -i {MODEL_PATH} -d {CACHE_PATH} ... with {full_path}")
                cmd = [
                    "lemonade", "-d", CACHE_PATH,
                    "-i", model_path,
                    "openvino-load", "--device", "NPU", "-bp", full_path,
                    "-r", str(out_value),
                    "openvino-bench",
                    "--iterations", str(ITERATIONS), "--warmup-iterations", str(WARMUPS),
                    "--output-tokens", str(out_value),
                    "--prompts", full_path,
                ]

            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed on {fname} (part {part_num}): {e}")
            failed_tasks.append(fname)
            continue

    # Only generate report if at least one run succeeded
    if len(failed_tasks) < len(tasks):
        print(f"\nGenerating report from cache: {CACHE_PATH}")
        try:
            subprocess.run([
                "lemonade", "report",
                "-i", CACHE_PATH,
                "--no-save", "--perf", "--lean"
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Report generation failed: {e}")
    else:
        print("⚠️ Skipping report: all tasks failed.")

    if failed_tasks:
        print("\nSummary of failed tasks:")
        for f in failed_tasks:
            print(f" - {f}")

if __name__ == "__main__":
    main()
