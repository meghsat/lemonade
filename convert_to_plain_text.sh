#!/bin/bash
# Script to convert Llama 3.1 chat template prompts to plain text

PROMPT_DIR="/home/user/Downloads/ai-benchmarks-main/utilities/llm_benchmark_estimator/prompt_files"
OUTPUT_DIR="${PROMPT_DIR}/plain_text"
BACKUP_DIR="${PROMPT_DIR}/originals_$(date +%Y%m%d_%H%M%S)"

echo "Creating directories..."
mkdir -p "$OUTPUT_DIR"
mkdir -p "$BACKUP_DIR"

echo "Backing up original files..."
cp "$PROMPT_DIR"/llama3_*.txt "$BACKUP_DIR/" 2>/dev/null || echo "No llama3_*.txt files found"

echo "Converting prompt files to plain text..."
for file in "$PROMPT_DIR"/llama3_*.txt; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        echo "Processing: $filename"

        # Extract system message and user message, remove all special tokens
        cat "$file" | \
            sed 's/{{ //g; s/ }}//g' | \
            sed 's/<|begin_of_text|>//g' | \
            sed 's/<|start_header_id|>system<|end_header_id|>//g' | \
            sed 's/<|start_header_id|>user<|end_header_id|>//g' | \
            sed 's/<|start_header_id|>assistant<|end_header_id|>//g' | \
            sed 's/<|eot_id|>//g' | \
            sed 's/<|eom_id|>//g' | \
            sed 's/<|end_of_text|>//g' | \
            sed '/^$/N;/^\n$/d' \
            > "$OUTPUT_DIR/$filename"

        echo "Created: $OUTPUT_DIR/$filename"
    fi
done

echo ""
echo "Done! Converted files are in: $OUTPUT_DIR"
echo "Original files backed up to: $BACKUP_DIR"
echo ""
echo "To use the converted files, update your benchmark script to use:"
echo "  $OUTPUT_DIR"
