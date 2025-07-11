#!/bin/bash
# RunPod deployment script for Qwen Coder v2.5 7B LoRA fine-tuning

echo "=== RunPod Deployment Script for Qwen Coder v2.5 7B LoRA ==="

# Step 1: Install uv package manager
echo "Installing uv package manager..."
pip install uv

# Step 2: Initialize uv project
echo "Initializing uv project..."
uv init

# Step 3: Copy the RunPod-specific pyproject.toml
echo "Setting up RunPod dependencies..."
cp pyproject_use_this_one_on_runpod.toml pyproject.toml

# Step 4: Sync dependencies
echo "Installing dependencies..."
uv sync

# Step 5: Set HuggingFace token (if not already set)
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN environment variable not set!"
    echo "Please run: export HF_TOKEN='your-huggingface-token'"
    echo "You can get a token from: https://huggingface.co/settings/tokens"
fi

# Step 6: Check for data directory
if [ ! -d "data" ]; then
    echo "ERROR: data directory not found!"
    echo "Please upload your data folder with instruction.json"
    exit 1
fi

# Step 7: Create output directories
echo "Creating output directories..."
mkdir -p Qwen2.5-Coder-7B-LoRA
mkdir -p complete_checkpoint
mkdir -p final_model
mkdir -p workspace

# Step 8: Show GPU info
echo "GPU Information:"
nvidia-smi

# Step 9: Ready to train
echo "=== Setup Complete ==="
echo "To start training, run:"
echo "  export HF_TOKEN='your-huggingface-token'"
echo "  uv run train.py"
echo ""
echo "Monitor GPU usage with: watch -n 1 nvidia-smi"