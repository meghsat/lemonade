import os
import re
import subprocess
import sys
import datetime
import platform
import csv
import statistics
import math

if len(sys.argv) < 8 or len(sys.argv) > 9:
    print(
        "Usage: python script.py <vendor> <model [ hf_checkpoint | directory ]> <prompt_dir_path> <cache_base_path> <prompt_prefix [ mlperf | textgen ]> <iterations> <warmups> [backend]"
    )
    print("  backend (optional): llamacpp (default), oga, or openvino")
    sys.exit(1)

VENDOR = sys.argv[1]
MODEL_PATH = sys.argv[2]
PROMPTS_PATH = sys.argv[3]
CACHE_BASE = sys.argv[4]
FILE_PREFIX = sys.argv[5]
ITERATIONS = sys.argv[6]
WARMUPS = sys.argv[7]
BACKEND = sys.argv[8].lower() if len(sys.argv) == 9 else "llamacpp"

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


def is_nvidia_arm64():
    is_arm64 = platform.machine().lower() in ["aarch64", "arm64"]

    # Check for nvidia-smi availability
    has_nvidia = False
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        has_nvidia = result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return is_arm64 and has_nvidia


def process_csv_with_scores(csv_path, model_path):
    """
    Read the benchmark CSV, calculate averages and scores, and add new columns.

    Args:
        csv_path: Path to the benchmark_results.csv file
        model_path: Path to the model (used to determine c1 and c2 values)
    """
    if not os.path.exists(csv_path):
        print(f"Warning: CSV file not found at {csv_path}")
        return

    # Determine c1 and c2 based on model name
    model_name_lower = model_path.lower()
    if "phi" in model_name_lower:
        c1 = 1250
        c2 = 20.49
        model_type = "Phi"
    elif "llama" in model_name_lower:
        c1 = 1300
        c2 = 31.25
        model_type = "Llama"
    else:
        # Default values if model type not recognized
        c1 = 1250
        c2 = 20.49
        model_type = "Unknown"

    print(f"\nProcessing CSV with model type: {model_type} (C1={c1}, C2={c2})")

    # Read the CSV
    rows = []
    with open(csv_path, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    if not rows:
        print("Warning: CSV file is empty")
        return

    # Calculate averages across all prompts
    ttft_values = []
    tps_values = []

    for row in rows:
        try:
            ttft = float(row.get("Seconds To First Token", 0))
            tps = float(row.get("Token Generation Tokens Per Second", 0))
            if ttft > 0:
                ttft_values.append(ttft)
            if tps > 0:
                tps_values.append(tps)
        except (ValueError, TypeError):
            continue

    if not ttft_values or not tps_values:
        print("Warning: No valid TTFT or TPS values found in CSV")
        return

    average_ttft = statistics.mean(ttft_values)
    average_tps = statistics.mean(tps_values)

    print(f"  Average TTFT: {average_ttft:.3f} seconds")
    print(f"  Average TPS: {average_tps:.3f} tokens/second")

    ttft_score = c1 / average_ttft if average_ttft > 0 else 0
    ots_score = c2 * average_tps
    overall_score = (
        math.sqrt(ttft_score * ots_score) if ttft_score > 0 and ots_score > 0 else 0
    )

    print(f"  TTFT Score: {ttft_score:.3f}")
    print(f"  OTS Score: {ots_score:.3f}")
    print(f"  Overall Score: {overall_score:.3f}")

    new_fieldnames = list(fieldnames) + [
        "C1",
        "C2",
        "Average TTFT",
        "Average TPS",
        "TTFT Score",
        "OTS Score",
        "Overall Score",
    ]

    for row in rows:
        row["C1"] = c1
        row["C2"] = c2
        row["Average TTFT"] = f"{average_ttft:.3f}"
        row["Average TPS"] = f"{average_tps:.3f}"
        row["TTFT Score"] = f"{ttft_score:.3f}"
        row["OTS Score"] = f"{ots_score:.3f}"
        row["Overall Score"] = f"{overall_score:.3f}"

    # Write back to CSV
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✓ Updated CSV with averages and scores: {csv_path}")


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
            if is_nvidia_arm64():
                # Read prompt file content
                with open(full_path, "r", encoding="utf-8") as f:
                    prompt_content = f.read().strip()

                print(f"Running lemonade with prompt file: {fname}")

                # Use llamacpp-bench with prompt content (requires --cli flag for text prompts)
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
                        "lemonade", "-d", CACHE_PATH,
                        "-i", model_path, "--power-agt",
                        "oga-load", "--device", "hybrid", "--dtype", "int4",
                        "oga-bench", "--prompts", full_path,
                        "--iterations", str(ITERATIONS), "--warmup-iterations", str(WARMUPS),
                        "--output-tokens", str(out_value)
                    ]
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

        process_csv_with_scores(csv_path, model_path)

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
