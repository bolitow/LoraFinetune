#!/bin/bash
# Script to check data file on RunPod

echo "=== Checking data/instruction.json ==="

# Check if file exists
if [ ! -f "data/instruction.json" ]; then
    echo "ERROR: data/instruction.json not found!"
    echo "Checking if it's a Git LFS pointer..."
    ls -la data/
    exit 1
fi

# Check file size
echo "File size:"
ls -lh data/instruction.json

# Check if it's a Git LFS pointer file
if head -n 1 data/instruction.json | grep -q "version https://git-lfs.github.com"; then
    echo "ERROR: File is a Git LFS pointer, not the actual data!"
    echo "Running git lfs pull..."
    git lfs pull
    echo "Checking again..."
    ls -lh data/instruction.json
fi

# Check first few bytes
echo -e "\nFirst 200 characters of file:"
head -c 200 data/instruction.json

# Check if valid JSON
echo -e "\n\nValidating JSON..."
python3 -c "import json; json.load(open('data/instruction.json'))" && echo "JSON is valid!" || echo "JSON is invalid!"

# Count records
echo -e "\nCounting records..."
python3 -c "import json; data=json.load(open('data/instruction.json')); print(f'Total records: {len(data)}')"