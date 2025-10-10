#!/bin/bash

# ====================================================================
# ProteinMPNN Automated Setup Script
# ====================================================================
# This script automatically:
# - Detects GPU/CPU availability
# - Installs PyTorch with appropriate CUDA support
# - Installs all required dependencies
# - Verifies the installation
# ====================================================================

set -e  # Exit on error

echo "============================================"
echo "ProteinMPNN Setup Script"
echo "============================================"
echo ""

# Function to check if conda environment is activated
check_conda_env() {
    if [[ "$CONDA_DEFAULT_ENV" != "proteinmpnn" ]]; then
        echo "ERROR: Please activate the proteinmpnn conda environment first:"
        echo "  conda activate proteinmpnn"
        exit 1
    fi
}

# Check if we're in the right environment
check_conda_env

echo "✓ Conda environment 'proteinmpnn' is activated"
echo ""

# Detect operating system
OS_TYPE=$(uname -s)
echo "Detected OS: $OS_TYPE"

# Detect GPU availability
echo ""
echo "Checking for GPU availability..."
GPU_AVAILABLE=false

if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        GPU_AVAILABLE=true
        echo "✓ NVIDIA GPU detected!"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    else
        echo "○ NVIDIA driver found but GPU not accessible"
    fi
else
    echo "○ No NVIDIA GPU detected"
fi

# Detect CUDA version if available
CUDA_VERSION=""
if [ "$GPU_AVAILABLE" = true ]; then
    CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9.]+")
    echo "CUDA Version: $CUDA_VERSION"
fi

echo ""
echo "Installing dependencies..."
echo ""

# Install PyTorch based on GPU availability
if [ "$GPU_AVAILABLE" = true ]; then
    echo "Installing PyTorch with CUDA support..."
    # Install PyTorch with CUDA 11.8 (widely compatible)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "Installing PyTorch (CPU-only version)..."
    if [ "$OS_TYPE" = "Darwin" ]; then
        # macOS - use default PyTorch
        pip install torch torchvision torchaudio
    else
        # Linux/Windows CPU
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
fi

echo ""
echo "Installing additional dependencies..."

# Install other required packages
pip install numpy biopython scipy tqdm

echo ""
echo "Verifying installation..."
echo ""

# Verify PyTorch installation
python -c "
import torch
import sys

print('Python version:', sys.version)
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())

if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('GPU device:', torch.cuda.get_device_name(0))
    print('Number of GPUs:', torch.cuda.device_count())
else:
    print('Running on CPU')

# Verify other packages
try:
    import numpy as np
    print('NumPy version:', np.__version__)
except ImportError:
    print('ERROR: NumPy not found!')
    sys.exit(1)

try:
    from Bio import SeqIO
    print('Biopython installed ✓')
except ImportError:
    print('ERROR: Biopython not found!')
    sys.exit(1)

try:
    import scipy
    print('SciPy version:', scipy.__version__)
except ImportError:
    print('ERROR: SciPy not found!')
    sys.exit(1)

try:
    import tqdm
    print('tqdm installed ✓')
except ImportError:
    print('ERROR: tqdm not found!')
    sys.exit(1)

print('')
print('✓ All dependencies installed successfully!')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "✓ Setup completed successfully!"
    echo "============================================"
    echo ""
    echo "You can now run ProteinMPNN. Example:"
    echo "  python protein_mpnn_run.py --pdb_path ./inputs/PDB_monomers/pdbs/ --out_folder ./outputs/test_run"
    echo ""
else
    echo ""
    echo "============================================"
    echo "✗ Setup failed. Please check the errors above."
    echo "============================================"
    exit 1
fi

