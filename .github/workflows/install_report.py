import argparse
import importlib.metadata
import json
import sys
from pathlib import Path
from tabulate import tabulate


def dist_size(dist):
    total = 0
    for f in dist.files or []:
        p = dist.locate_file(f)
        if p.is_file():
            total += p.stat().st_size
    return total


def dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


parser = argparse.ArgumentParser()
parser.add_argument("--stats", help="Write JSON stats to this file")
parser.add_argument("--install-time", type=int, help="Installation time in seconds")
args = parser.parse_args()

rows = [(d.metadata["Name"], dist_size(d)) for d in importlib.metadata.distributions()]

base_dir = Path(sys.executable).resolve().parent
for name in ("vulkan", "rocm"):
    extra = base_dir / name
    if extra.is_dir():
        rows.append((name, dir_size(extra)))

rows.sort(key=lambda x: x[1], reverse=True)

total = sum(size for _, size in rows)

print(f"Total installation size: {total/1024/1024:.2f} MB\n")
print("### Installation Size Breakdown\n")

table_data = []
for name, size in rows:
    mb = size / 1024 / 1024
    pct = (size / total * 100) if total else 0
    table_data.append((name, mb, pct))
table_data.append(("**Total**", total / 1024 / 1024, 100.0))

print(
    tabulate(
        table_data,
        headers=["Package", "Size (MB)", "% of Total"],
        tablefmt="github",
        floatfmt=".2f",
    )
)

if args.stats:
    with open(args.stats, "w") as f:
        json.dump({"size": total, "time": args.install_time}, f)
