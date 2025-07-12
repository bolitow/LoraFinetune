#!/bin/bash
# RunPod deployment script for Qwen Coder v2.5 7B LoRA fine-tuning

echo "=== RunPod Deployment Script for Qwen Coder v2.5 7B LoRA ==="

# Step 1: Install Git LFS and uv package manager
echo "Installing Git LFS..."
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install -y git-lfs
git lfs install

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

# Step 4b: Install flash-attn separately with special build flags
echo "Installing Flash Attention 2 for optimized performance..."
uv pip install flash-attn --no-build-isolation

# Step 5: Set HuggingFace token and username (if not already set)
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN environment variable not set!"
    echo "Please run: export HF_TOKEN='your-huggingface-token'"
    echo "You can get a token from: https://huggingface.co/settings/tokens"
fi

if [ -z "$HF_USERNAME" ]; then
    echo "INFO: HF_USERNAME not set. Model will not be published to HuggingFace."
    echo "To enable automatic publishing, run: export HF_USERNAME='your-huggingface-username'"
fi

# Step 6: Clone repository with Git LFS if needed
if [ ! -d ".git" ]; then
    echo "Cloning repository with Git LFS support..."
    git clone https://github.com/bolitow/LoraFinetune.git .
    git lfs pull
fi

# Step 7: Check for data directory and Git LFS
if [ ! -d "data" ]; then
    echo "ERROR: data directory not found!"
    echo "Please upload your data folder with instruction.json"
    exit 1
fi

# Check if instruction.json is a Git LFS pointer
if [ -f "data/instruction.json" ]; then
    if head -n 1 data/instruction.json | grep -q "version https://git-lfs.github.com"; then
        echo "Detected Git LFS pointer for data/instruction.json"
        echo "Pulling LFS data..."
        git lfs pull
    fi
    echo "Data file size: $(ls -lh data/instruction.json | awk '{print $5}')"
fi

# Step 8: Create output directories
echo "Creating output directories..."
mkdir -p final_merged_model
mkdir -p workspace

# Step 9: Show GPU info
echo "GPU Information:"
nvidia-smi

# Step 10: Ready to train
echo "=== Setup Complete ==="
echo "To start training, run:"
echo "  export HF_TOKEN='your-huggingface-token'"
echo "  export HF_USERNAME='your-huggingface-username'  # Optional: for auto-publishing"
echo "  uv run train.py"
echo ""
echo "Monitor GPU usage with: watch -n 1 nvidia-smi"