#!/bin/bash

##########################################################################
# Setup script for AMD HPC cluster environment
# Run this once after SSH'ing into the cluster
#
# Usage:
#   ssh USER@hpcfund.amd.com
#   cd ~/finetune
#   ./setup_cluster.sh
#
# The venv is created under $WORK (not $HOME) because PyTorch ROCm
# requires ~8 GB and $HOME has only a 24 GB quota.
##########################################################################

set -e

echo "=========================================="
echo "AMD HPC Cluster - Environment Setup"
echo "=========================================="
echo ""

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# Venv lives under $WORK to avoid filling $HOME (24 GB quota)
VENV_DIR="${WORK}/finetune-venv"

# Check Python availability
echo "[1/5] Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "Python3 not found. Please load Python module:"
    echo "   module load python/3.10"
    exit 1
fi
echo "Python found: $(python3 --version)"
echo ""

# Check PyTorch availability
echo "[2/5] Checking PyTorch..."
if python3 -c "import torch" 2>/dev/null; then
    echo "PyTorch already available"
    python3 -c "import torch; print('  Version:', torch.__version__); print('  ROCm:', torch.cuda.is_available()); print('  GPUs:', torch.cuda.device_count() if torch.cuda.is_available() else 'N/A')"
else
    echo "PyTorch not found. Will install during venv creation."
fi
echo ""

# Create virtual environment under $WORK
echo "[3/5] Creating Python virtual environment in \$WORK..."
echo "  Path: $VENV_DIR"
if [[ -d "$VENV_DIR" ]]; then
    echo "Virtual environment already exists"
    read -p "  Recreate? (y/N) " -r
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
        python3 -m venv "$VENV_DIR"
        echo "Virtual environment created"
    fi
else
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment created"
fi
echo ""

# Activate venv
echo "[4/5] Activating virtual environment..."
source "$VENV_DIR/bin/activate"
echo "Virtual environment activated"
echo ""

# Install requirements
echo "[5/5] Installing Python packages..."
echo "  This may take a few minutes..."
pip install --upgrade pip wheel setuptools > /dev/null 2>&1
# Install PyTorch with ROCm support (AMD GPUs use HIP/ROCm, not CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
# Install remaining dependencies without overwriting the ROCm torch
grep -v "^torch" experiments/requirements.txt | pip install -r /dev/stdin
echo "Dependencies installed"
echo ""

# Verify installation
echo "=========================================="
echo "Verification"
echo "=========================================="
echo ""

echo "Python: $(python --version)"
echo "Venv:   $VENV_DIR"
echo ""
echo "PyTorch:"
python -c "
import torch
print(f'  Version: {torch.__version__}')
print(f'  ROCm available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  Device name: {torch.cuda.get_device_name(0)}')
    print(f'  Device count: {torch.cuda.device_count()}')
else:
    print('  (Will use CPU for now)')
"

echo ""
echo "Transformers:"
python -c "import transformers; print(f'  Version: {transformers.__version__}')"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the venv manually:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Next steps:"
echo "  ./experiments/submit.sh bench-base mi250 4"
echo ""
