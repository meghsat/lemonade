#!/bin/bash
# Script to remove Jinja2 template syntax from prompt files

PROMPT_DIR="/home/user/Downloads/ai-benchmarks-main/utilities/llm_benchmark_estimator/prompt_files"
BACKUP_DIR="${PROMPT_DIR}/originals_$(date +%Y%m%d_%H%M%S)"

echo "Creating backup directory: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

echo "Backing up original files..."
cp "$PROMPT_DIR"/llama3_*.txt "$BACKUP_DIR/" 2>/dev/null || echo "No llama3_*.txt files found"

echo "Processing prompt files..."
for file in "$PROMPT_DIR"/llama3_*.txt; do
    if [ -f "$file" ]; then
        echo "Processing: $(basename "$file")"
        # Remove {{ and }} from the file
        sed -i.bak 's/{{ //g; s/ }}//g' "$file"
        # Remove the leading <|begin_of_text|> to avoid double BOS tokens
        sed -i.bak 's/^<|begin_of_text|>//' "$file"
        # Remove the .bak file created by sed
        rm -f "${file}.bak"
    fi
done

echo "Done! Original files backed up to: $BACKUP_DIR"
echo "You can now run your benchmark script."
