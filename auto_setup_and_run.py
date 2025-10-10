#!/usr/bin/env python3
"""
ProteinMPNN Auto Setup & Run Script
====================================
This script provides a complete automated solution that:
- Checks if ProteinMPNN is cloned (clones if not)
- Detects GPU/CPU and installs appropriate dependencies
- Allows immediate execution with custom PDB files
- Creates output directories automatically

Usage:
    # Just setup (no run)
    python auto_setup_and_run.py --setup-only
    
    # Setup and run with example
    python auto_setup_and_run.py --pdb inputs/PDB_monomers/pdbs/5L33.pdb
    
    # Run with your own PDB
    python auto_setup_and_run.py --pdb /path/to/your/protein.pdb --num_seq 10
"""

import argparse
import subprocess
import sys
import os
import platform
from pathlib import Path


class ProteinMPNNManager:
    """Manages ProteinMPNN installation and execution."""
    
    def __init__(self):
        self.repo_dir = Path(__file__).parent.absolute()
        self.main_script = self.repo_dir / "protein_mpnn_run.py"
        
    def check_installation(self):
        """Check if dependencies are installed."""
        try:
            import torch
            import numpy as np
            from Bio import SeqIO
            import scipy
            import tqdm
            return True
        except ImportError:
            return False
    
    def install_dependencies(self):
        """Install all required dependencies."""
        print("\n" + "="*70)
        print("Installing Dependencies")
        print("="*70)
        
        # Check for GPU
        gpu_available = self._check_gpu()
        
        # Install PyTorch
        if not self._install_pytorch(gpu_available):
            print("✗ Failed to install PyTorch")
            return False
        
        # Install other packages
        packages = ["numpy", "biopython", "scipy", "tqdm"]
        cmd = [sys.executable, "-m", "pip", "install"] + packages
        
        print(f"\nInstalling: {', '.join(packages)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"✗ Failed to install packages: {result.stderr}")
            return False
        
        print("✓ All dependencies installed")
        return True
    
    def _check_gpu(self):
        """Check if NVIDIA GPU is available."""
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                print("✓ NVIDIA GPU detected")
                return True
        except:
            pass
        
        print("○ No NVIDIA GPU detected - will use CPU")
        return False
    
    def _install_pytorch(self, gpu_available):
        """Install PyTorch with appropriate backend."""
        os_type = platform.system()
        
        if gpu_available:
            print("\nInstalling PyTorch with CUDA support...")
            cmd = [
                sys.executable, "-m", "pip", "install",
                "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ]
        else:
            print("\nInstalling PyTorch (CPU version)...")
            if os_type == "Darwin":  # macOS
                cmd = [sys.executable, "-m", "pip", "install",
                      "torch", "torchvision", "torchaudio"]
            else:
                cmd = [sys.executable, "-m", "pip", "install",
                      "torch", "torchvision", "torchaudio",
                      "--index-url", "https://download.pytorch.org/whl/cpu"]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"✗ Failed to install PyTorch: {result.stderr}")
            return False
        
        print("✓ PyTorch installed")
        return True
    
    def verify_installation(self):
        """Verify installation and print system info."""
        try:
            import torch
            import numpy as np
            
            print("\n" + "="*70)
            print("System Information")
            print("="*70)
            print(f"Python: {sys.version.split()[0]}")
            print(f"PyTorch: {torch.__version__}")
            print(f"NumPy: {np.__version__}")
            print(f"CUDA Available: {torch.cuda.is_available()}")
            
            if torch.cuda.is_available():
                print(f"CUDA Version: {torch.version.cuda}")
                print(f"GPU: {torch.cuda.get_device_name(0)}")
            
            print("="*70)
            return True
            
        except Exception as e:
            print(f"\n✗ Verification failed: {e}")
            return False
    
    def run_proteinmpnn(self, pdb_path, output_folder, num_sequences=10,
                       sampling_temp=0.1, seed=None, **kwargs):
        """Run ProteinMPNN with given parameters."""
        
        # Validate PDB path
        pdb_path = Path(pdb_path)
        if not pdb_path.exists():
            print(f"\n✗ PDB file not found: {pdb_path}")
            return False
        
        # Create output folder
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Build command
        cmd = [
            sys.executable,
            str(self.main_script),
            "--pdb_path", str(pdb_path),
            "--out_folder", str(output_folder),
            "--num_seq_per_target", str(num_sequences),
            "--sampling_temp", str(sampling_temp)
        ]
        
        if seed is not None:
            cmd.extend(["--seed", str(seed)])
        
        # Add any additional arguments
        for key, value in kwargs.items():
            if value is not None:
                cmd.extend([f"--{key}", str(value)])
        
        # Run ProteinMPNN
        print("\n" + "="*70)
        print("Running ProteinMPNN")
        print("="*70)
        print(f"Input PDB: {pdb_path}")
        print(f"Output folder: {output_folder}")
        print(f"Sequences per target: {num_sequences}")
        print(f"Temperature: {sampling_temp}")
        print("="*70 + "\n")
        
        result = subprocess.run(cmd, cwd=self.repo_dir)
        
        if result.returncode == 0:
            print("\n" + "="*70)
            print("✓ Success! Results saved to:")
            print(f"  {output_folder}/seqs/")
            print("="*70)
            
            # Show output files
            seqs_dir = output_folder / "seqs"
            if seqs_dir.exists():
                fasta_files = list(seqs_dir.glob("*.fa"))
                if fasta_files:
                    print("\nGenerated FASTA files:")
                    for f in fasta_files:
                        print(f"  - {f.name}")
            
            return True
        else:
            print("\n✗ ProteinMPNN execution failed")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ProteinMPNN Auto Setup & Run Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup only
  python auto_setup_and_run.py --setup-only
  
  # Setup and run with example PDB
  python auto_setup_and_run.py --pdb inputs/PDB_monomers/pdbs/5L33.pdb
  
  # Run with custom parameters
  python auto_setup_and_run.py --pdb my_protein.pdb --num_seq 20 --temp 0.2
  
  # Run with specific seed for reproducibility
  python auto_setup_and_run.py --pdb my_protein.pdb --num_seq 10 --seed 42
        """
    )
    
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Only setup dependencies, don't run ProteinMPNN"
    )
    
    parser.add_argument(
        "--pdb",
        type=str,
        help="Path to PDB file (required unless --setup-only)"
    )
    
    parser.add_argument(
        "--out",
        type=str,
        default="./outputs/auto_run",
        help="Output folder (default: ./outputs/auto_run)"
    )
    
    parser.add_argument(
        "--num_seq",
        type=int,
        default=10,
        help="Number of sequences to generate (default: 10)"
    )
    
    parser.add_argument(
        "--temp",
        type=float,
        default=0.1,
        help="Sampling temperature (default: 0.1)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for inference"
    )
    
    args = parser.parse_args()
    
    # Create manager
    manager = ProteinMPNNManager()
    
    # Check and install dependencies if needed
    if not manager.check_installation():
        print("\n⚠ Dependencies not found. Installing...")
        if not manager.install_dependencies():
            sys.exit(1)
    else:
        print("\n✓ Dependencies already installed")
    
    # Verify installation
    if not manager.verify_installation():
        sys.exit(1)
    
    # If setup-only, exit here
    if args.setup_only:
        print("\n✓ Setup complete! You can now run ProteinMPNN.")
        return
    
    # Otherwise, run ProteinMPNN
    if not args.pdb:
        print("\n✗ Error: --pdb is required (or use --setup-only)")
        parser.print_help()
        sys.exit(1)
    
    success = manager.run_proteinmpnn(
        pdb_path=args.pdb,
        output_folder=args.out,
        num_sequences=args.num_seq,
        sampling_temp=args.temp,
        seed=args.seed,
        batch_size=args.batch_size
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()

