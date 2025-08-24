import importlib.metadata

def dist_size(dist):
    total = 0
    for f in dist.files or []:
        p = dist.locate_file(f)
        if p.is_file():
            total += p.stat().st_size
    return total

rows = []
for d in importlib.metadata.distributions():
    rows.append((d.metadata['Name'], dist_size(d)))
rows.sort(key=lambda x: x[1], reverse=True)

total = sum(size for _, size in rows)
print("\n### Installation Size Breakdown\n")
print("| Package | Size (MB) |")
print("| --- | ---: |")
for name, size in rows:
    print(f"| {name} | {size/1024/1024:.2f} |")
print(f"| **Total** | **{total/1024/1024:.2f}** |")
