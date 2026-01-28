#!/usr/bin/env python3
"""
ProteinMPNN Quick Installation Script
======================================
This Python script provides a cross-platform installation solution
that works on Linux, macOS, and Windows.

Usage:
    python quick_install.py

Requirements:
    - Python 3.9+ environment activated
    - Internet connection
"""

import subprocess
import sys
import platform
import os


def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {e}")
        print(e.stdout)
        print(e.stderr)
        return False


def check_gpu():
    """Check if NVIDIA GPU is available."""
    print("\nChecking for GPU availability...")
    
    try:
        result = subprocess.run(
            "nvidia-smi",
            shell=True,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✓ NVIDIA GPU detected!")
            print(result.stdout.split('\n')[0:10])  # Print first 10 lines
            return True
    except:
        pass
    
    print("○ No NVIDIA GPU detected - will use CPU")
    return False


def install_pytorch(gpu_available):
    """Install PyTorch with appropriate backend."""
    os_type = platform.system()
    
    if gpu_available:
        print("\nInstalling PyTorch with CUDA support...")
        cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    else:
        print("\nInstalling PyTorch (CPU-only version)...")
        if os_type == "Darwin":  # macOS
            cmd = "pip install torch torchvision torchaudio"
        else:  # Linux/Windows
            cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    
    return run_command(cmd, "Installing PyTorch")


def install_dependencies():
    """Install additional required packages."""
    packages = ["numpy", "biopython", "scipy", "tqdm"]
    cmd = f"pip install {' '.join(packages)}"
    return run_command(cmd, "Installing additional dependencies")


def verify_installation():
    """Verify that all packages are correctly installed."""
    print("\n" + "="*60)
    print("Verifying installation...")
    print("="*60)
    
    try:
        import torch
        import numpy as np
        from Bio import SeqIO
        import scipy
        import tqdm
        
        print(f"\n✓ Python version: {sys.version}")
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ GPU device: {torch.cuda.get_device_name(0)}")
            print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
        else:
            print("✓ Running on CPU")
        
        print(f"✓ NumPy version: {np.__version__}")
        print(f"✓ Biopython installed")
        print(f"✓ SciPy version: {scipy.__version__}")
        print(f"✓ tqdm installed")
        
        print("\n" + "="*60)
        print("✓ All dependencies installed successfully!")
        print("="*60)
        return True
        
    except ImportError as e:
        print(f"\n✗ Import Error: {e}")
        print("\nPlease ensure all packages are installed correctly.")
        return False


def main():
    """Main installation routine."""
    print("="*60)
    print("ProteinMPNN Installation Script")
    print("="*60)
    print(f"\nOperating System: {platform.system()}")
    print(f"Python Version: {sys.version}")
    
    # Check Python version
    if sys.version_info < (3, 9):
        print("\n✗ ERROR: Python 3.9 or higher is required!")
        print("Please create a conda environment with Python 3.9:")
        print("  conda create -n proteinmpnn python=3.9 -y")
        print("  conda activate proteinmpnn")
        sys.exit(1)
    
    # Check for GPU
    gpu_available = check_gpu()
    
    # Install PyTorch
    if not install_pytorch(gpu_available):
        print("\n✗ Failed to install PyTorch")
        sys.exit(1)
    
    # Install other dependencies
    if not install_dependencies():
        print("\n✗ Failed to install dependencies")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("\n✗ Installation verification failed")
        sys.exit(1)
    
    # Success message
    print("\n" + "="*60)
    print("Installation Complete!")
    print("="*60)
    print("\nYou can now run ProteinMPNN. Example commands:")
    print("\n  # Basic usage:")
    print("  python protein_mpnn_run.py \\")
    print("      --pdb_path ./inputs/PDB_monomers/pdbs/ \\")
    print("      --out_folder ./outputs/my_run")
    print("\n  # With specific PDB file:")
    print("  python protein_mpnn_run.py \\")
    print("      --pdb_path ./inputs/PDB_monomers/pdbs/5L33.pdb \\")
    print("      --out_folder ./outputs/5L33_run \\")
    print("      --num_seq_per_target 10")
    print("")


if __name__ == "__main__":
    main()

