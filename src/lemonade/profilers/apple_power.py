import os
import platform
import textwrap
import time
import threading
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lemonade.common.printing as printing
from lemonade.profilers import Profiler
from lemonade.tools.report.table import LemonadePerfTable, DictListStat

DEFAULT_TRACK_POWER_INTERVAL_S = 0.05  
DEFAULT_TRACK_POWER_WARMUP_PERIOD = 5

POWER_USAGE_CSV_FILENAME = "power_usage_apple.csv"
POWER_USAGE_PNG_FILENAME = "power_usage_apple.png"


class Keys:
    POWER_USAGE_PLOT = "power_usage_plot_apple"
    POWER_USAGE_DATA = "power_usage_data_apple"
    POWER_USAGE_DATA_CSV = "power_usage_data_file_apple"

    # powermetrics metrics
    PEAK_GPU_POWER = "peak_gpu_power_apple"
    AVG_GPU_POWER = "avg_gpu_power_apple"

# Add column to the Lemonade performance report table for the power data
LemonadePerfTable.table_descriptor["stat_columns"].append(
    DictListStat(
        "Power Usage (Apple)",
        Keys.POWER_USAGE_DATA,
        [
            ("name", "{0}:"),
            ("duration", "{0:.1f}s,"),
            ("energy consumed", "{0:.1f} J"),
        ],
    )
)


class ApplePowerProfiler(Profiler):

    unique_name = "power-apple"

    @staticmethod
    def add_arguments_to_parser(parser):
        parser.add_argument(
            f"--{ApplePowerProfiler.unique_name}",
            nargs="?",
            metavar="WARMUP_PERIOD",
            type=int,
            default=None,
            const=DEFAULT_TRACK_POWER_WARMUP_PERIOD,
            help="Track Apple GPU power consumption using powermetrics "
            "and plot the results. Requires sudo access. "
            "Optionally, set the warmup period in seconds "
            f"(default: {DEFAULT_TRACK_POWER_WARMUP_PERIOD}). "
            "This works on macOS systems with Apple Silicon. "
            "You may be prompted for your sudo password.",
        )

    def __init__(self, parser_arg_value):
        super().__init__()
        self.warmup_period = parser_arg_value
        self.status_stats += [
            Keys.PEAK_GPU_POWER,
            Keys.AVG_GPU_POWER,
            Keys.POWER_USAGE_PLOT,
        ]
        self.tracking_active = False
        self.build_dir = None
        self.csv_path = None
        self.data = None
        self.powermetrics_data = []
        self.powermetrics_thread = None

    def _monitor_powermetrics(self):
        """Background thread that monitors GPU/CPU power using powermetrics."""
        start_time = time.time()
        sample_count = 0

        while self.tracking_active:
            try:
                current_time = time.time() - start_time

                cmd = [
                    "sudo",
                    "-n",  # non-interactive (won't prompt for password if already authenticated)
                    "powermetrics",
                    "-n", "1",  # single sample
                    "-i", str(int(DEFAULT_TRACK_POWER_INTERVAL_S * 1000)),
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

                    if sample_count == 0:
                        debug_file = os.path.join(self.build_dir, "powermetrics_sample.txt")
                        with open(debug_file, 'w') as f:
                            f.write(output)
                        printing.log_info(f"Saved first powermetrics sample to {debug_file}")

                    # Extract power values
                    gpu_power = self._extract_value(output, "GPU Power")
                    cpu_power = self._extract_value(output, "CPU Power")
                    ane_power = self._extract_value(output, "ANE Power")
                    combined_power = self._extract_value(output, "Combined Power (CPU + GPU + ANE)")

                    # Debug logging for first few samples
                    if sample_count < 3:
                        printing.log_info(
                            f"Sample {sample_count}: gpu={gpu_power}W, cpu={cpu_power}W, ane={ane_power}W"
                        )

                    if gpu_power is not None or cpu_power is not None:
                        sample_data = {
                            'time': current_time,
                            'gpu_power': gpu_power if gpu_power is not None else 0,
                            'cpu_power': cpu_power if cpu_power is not None else 0,
                            'ane_power': ane_power if ane_power is not None else 0,
                            'combined_power': combined_power,
                        }

                        self.powermetrics_data.append(sample_data)
                        sample_count += 1
                    else:
                        if sample_count < 3:
                            printing.log_info(f"Warning: Both GPU and CPU power were None, skipping sample")
                else:
                    printing.log_info(f"powermetrics returned non-zero exit code: {result.returncode}")
                    if result.stderr:
                        printing.log_info(f"stderr: {result.stderr[:200]}")

                time.sleep(DEFAULT_TRACK_POWER_INTERVAL_S)

            except (subprocess.TimeoutExpired, ValueError) as e:
                printing.log_info(f"Error reading powermetrics: {e}")
                time.sleep(DEFAULT_TRACK_POWER_INTERVAL_S)
            except Exception as e:
                printing.log_info(f"Unexpected error in powermetrics monitoring: {e}")
                break

        printing.log_info(f"Monitoring stopped. Collected {len(self.powermetrics_data)} samples")

    def _extract_value(self, text, key):
        try:
            import re

            # Text format: "GPU Power: 123 mW" or "CPU Power: 456 mW"
            pattern_text = rf"{re.escape(key)}:\s*([0-9.]+)\s*mW"
            match = re.search(pattern_text, text)
            if match:
                value = float(match.group(1))
                # powermetrics reports in milliwatts, convert to watts
                return value / 1000.0

            return None
        except Exception:
            return None

    def start(self, build_dir):
        if self.tracking_active:
            raise RuntimeError("Cannot start power tracking while already tracking")

        if platform.system() != "Darwin":
            raise RuntimeError("Apple power usage tracking is only enabled on macOS.")

        self.build_dir = build_dir

        self.csv_path = os.path.join(build_dir, POWER_USAGE_CSV_FILENAME)

        try:
            result = subprocess.run(
                ["which", "powermetrics"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "powermetrics not found. This tool is typically available on macOS 10.9+."
                )
        except Exception as e:
            raise RuntimeError(f"Error checking for powermetrics: {e}")

        try:
            test_cmd = [
                "sudo",
                "powermetrics",
                "-n", "1",
                "-i", "1000",
                "--samplers", "cpu_power",
                "--show-usage-summary",
            ]
            result = subprocess.run(
                test_cmd,
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Failed to run powermetrics with sudo. Error: {result.stderr}"
                )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Timeout waiting for sudo access. Please ensure you can run 'sudo powermetrics'.")
        except Exception as e:
            raise RuntimeError(f"Failed to test powermetrics: {e}")

        printing.log_info("Monitoring Apple GPU using powermetrics")

        # Start monitoring in background thread
        self.tracking_active = True
        self.powermetrics_data = []

        # Start powermetrics monitoring thread
        self.powermetrics_thread = threading.Thread(target=self._monitor_powermetrics, daemon=True)
        self.powermetrics_thread.start()

        # Warmup period
        printing.log_info(f"Warming up for {self.warmup_period}s...")
        time.sleep(self.warmup_period)

    def stop(self):
        if self.tracking_active:
            self.tracking_active = False

            # Wait for monitoring thread to finish
            if self.powermetrics_thread:
                self.powermetrics_thread.join(timeout=5)

            # Cooldown period
            printing.log_info(f"Cooling down for {self.warmup_period}s...")
            time.sleep(self.warmup_period)

    def generate_results(self, state, timestamp, start_times):
        if not self.powermetrics_data:
            printing.log_info("No power data collected")
            state.save_stat(Keys.POWER_USAGE_PLOT, "NONE")
            return

        if self.tracking_active:
            self.stop()

        # Convert powermetrics data to DataFrame
        df = pd.DataFrame(self.powermetrics_data)

        # Save CSV
        df.to_csv(self.csv_path, index=False)

        # Remap time to start at 0 when first tool starts
        if start_times:
            tool_start_times = sorted(start_times.values())
            # First tool after warmup
            first_tool_time = tool_start_times[1] if len(tool_start_times) > 1 else tool_start_times[0]

            # For simplicity, just make the first measurement time 0
            df['time'] = df['time'] - df['time'].iloc[0]

        # Calculate statistics from powermetrics data
        peak_power = df['gpu_power'].max()
        avg_power = df['gpu_power'].mean()

        printing.log_info(f"powermetrics: Peak GPU Power={peak_power:.1f}W, "
                        f"Avg GPU Power={avg_power:.1f}W")

        # Create a figure with 2 subplots (GPU power, CPU/ANE power)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))

        if start_times:
            tool_starts = sorted(start_times.items(), key=lambda item: item[1])
            tool_name_list = [item[0] for item in tool_starts]

            # Adjust to common time frame as power measurements
            tool_start_list = [
                max(df['time'].iloc[0], item[1] - (tool_starts[1][1] if len(tool_starts) > 1 else 0))
                for item in tool_starts
            ]
            tool_stop_list = tool_start_list[1:] + [df['time'].values[-1]]

            # Extract power data time series
            x_time = df['time'].to_numpy()
            y_power = df['gpu_power'].to_numpy()

            # Extract data for each stage in the build
            self.data = []
            for name, t0, tf in zip(tool_name_list, tool_start_list, tool_stop_list):
                x = x_time[(x_time >= t0) * (x_time <= tf)]
                if len(x) == 0:
                    continue
                x = np.insert(x, 0, t0)
                x = np.insert(x, len(x), tf)
                y = np.interp(x, x_time, y_power)
                energy = np.trapz(y, x)
                avg_power_stage = energy / (tf - t0) if (tf - t0) > 0 else 0
                stage = {
                    "name": name,
                    "t": x.tolist(),
                    "power": y.tolist(),
                    "duration": float(tf - t0),
                    "energy consumed": float(energy),
                    "average power": float(avg_power_stage),
                }
                self.data.append(stage)

            # Plot power usage for each stage
            for stage in self.data:
                p = ax1.plot(
                    stage["t"],
                    stage["power"],
                    label=f"{stage['name']} ({stage['duration']:.1f}s, "
                    f"{stage['energy consumed']:0.1f} J)",
                )
                # Add a dashed line to show average power
                ax1.plot(
                    [stage["t"][0], stage["t"][-1]],
                    [stage["average power"], stage["average power"]],
                    linestyle="--",
                    c=p[0].get_c(),
                )
                # Add average power text to plot
                ax1.text(
                    stage["t"][0],
                    stage["average power"],
                    f"{stage['average power']:.1f} W ",
                    horizontalalignment="right",
                    verticalalignment="center",
                    c=p[0].get_c(),
                )
        else:
            ax1.plot(df['time'], df['gpu_power'])

        # Add title and labels to first plot
        ax1.set_ylabel("GPU Power Draw [W]")
        title_str = "Apple GPU Power Stats\n" + "\n".join(textwrap.wrap(state.build_name, 60))
        ax1.set_title(title_str)
        ax1.legend()
        ax1.grid(True)

        # Second plot: CPU and ANE power
        ax2.plot(df['time'], df['cpu_power'], label='CPU Power [W]', c='orange')
        ax2.plot(df['time'], df['ane_power'], label='ANE Power [W]', c='green')

        # If combined power is available, show it as well
        if 'combined_power' in df.columns and df['combined_power'].notna().any():
            ax2.plot(df['time'], df['combined_power'], label='Combined Power [W]',
                    c='purple', linestyle='--', alpha=0.7)

        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Power [W]')
        ax2.legend(loc='best')
        ax2.grid(True)

        # Save plot to current folder AND save to cache
        plot_path = os.path.join(
            self.build_dir, f"{timestamp}_{POWER_USAGE_PNG_FILENAME}"
        )
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plot_path = os.path.join(os.getcwd(), f"{timestamp}_{POWER_USAGE_PNG_FILENAME}")
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        state.save_stat(Keys.POWER_USAGE_PLOT, plot_path)
        state.save_stat(Keys.POWER_USAGE_DATA, self.data)
        state.save_stat(Keys.POWER_USAGE_DATA_CSV, self.csv_path)

        # Save statistics
        state.save_stat(Keys.PEAK_GPU_POWER, f"{peak_power:0.1f} W")
        state.save_stat(Keys.AVG_GPU_POWER, f"{avg_power:0.1f} W")

        printing.log_info(f"Power usage plot saved to: {plot_path}")


# Copyright (c) 2025 AMD
