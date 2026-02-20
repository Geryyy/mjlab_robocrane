#!/bin/bash
INPUT_DIR="input"
OUTPUT_DIR="output"
SCRIPT="simplify.mlx"

mkdir -p "$OUTPUT_DIR"

for file in "$INPUT_DIR"/*.{stl,obj}; do
  [ -e "$file" ] || continue
  filename=$(basename -- "$file")
  output="$OUTPUT_DIR/$filename"
  meshlabserver -i "$file" -o "$output" -s "$SCRIPT"
done
