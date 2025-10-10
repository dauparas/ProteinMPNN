#!/usr/bin/env python3
"""
Simple ProteinMPNN Runner
=========================
Ultra-simple interface for running ProteinMPNN.

Usage Option 1 (Edit config below and just run):
    python run_proteinmpnn.py
    
Usage Option 2 (Command line arguments):
    python run_proteinmpnn.py my_protein.pdb --num 20 --seed 42
"""

# ============================================================================
# 📝 CONFIGURATION - Edit these settings, then just run: python run_proteinmpnn.py
# ============================================================================

# Input PDB file (required)
PDB_FILE = "inputs/PDB_monomers/pdbs/5L33.pdb"

# Number of sequences to generate
NUM_SEQUENCES = 10

# Temperature mode: "conservative" (0.05), "default" (0.1), "diverse" (0.3), or custom number
TEMPERATURE_MODE = "default"  # Options: "conservative", "default", "diverse", or a number like 0.15

# Random seed for reproducibility (use None for random)
SEED = 42

# Output folder (use None for auto-generated with timestamp)
OUTPUT_FOLDER = None

# Batch size (reduce if out of memory, use None for default)
BATCH_SIZE = None

# ============================================================================

import argparse
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime


def check_dependencies():
    """Quick check if dependencies are installed."""
    try:
        import torch
        import numpy as np
        from Bio import SeqIO
        return True
    except ImportError:
        return False


def install_dependencies():
    """Install dependencies if missing."""
    print("\n" + "="*70)
    print("First-time setup: Installing dependencies...")
    print("This will take ~2-3 minutes")
    print("="*70)
    
    quick_install = Path(__file__).parent / "quick_install.py"
    if quick_install.exists():
        result = subprocess.run([sys.executable, str(quick_install)])
        return result.returncode == 0
    else:
        print("\n⚠ quick_install.py not found. Installing manually...")
        # Try basic installation
        packages = ["torch", "torchvision", "torchaudio", "numpy", "biopython", "scipy", "tqdm"]
        cmd = [sys.executable, "-m", "pip", "install"] + packages
        result = subprocess.run(cmd)
        return result.returncode == 0


def get_output_folder(pdb_path):
    """Generate a sensible output folder name."""
    pdb_name = Path(pdb_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"outputs/{pdb_name}_{timestamp}"


def run_proteinmpnn(pdb_path, output_folder, num_sequences, temperature, seed=None, batch_size=None):
    """Run ProteinMPNN with the given parameters."""
    
    script_dir = Path(__file__).parent
    main_script = script_dir / "protein_mpnn_run.py"
    
    if not main_script.exists():
        print(f"\n✗ Error: protein_mpnn_run.py not found in {script_dir}")
        return False
    
    # Build command
    cmd = [
        sys.executable,
        str(main_script),
        "--pdb_path", str(pdb_path),
        "--out_folder", str(output_folder),
        "--num_seq_per_target", str(num_sequences),
        "--sampling_temp", str(temperature)
    ]
    
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    
    if batch_size is not None:
        cmd.extend(["--batch_size", str(batch_size)])
    
    # Print info
    print("\n" + "="*70)
    print("Running ProteinMPNN")
    print("="*70)
    print(f"Input:       {pdb_path}")
    print(f"Output:      {output_folder}/seqs/")
    print(f"Sequences:   {num_sequences}")
    print(f"Temperature: {temperature}")
    if seed is not None:
        print(f"Seed:        {seed}")
    print("="*70 + "\n")
    
    # Run
    result = subprocess.run(cmd, cwd=script_dir)
    
    if result.returncode == 0:
        # Show results
        seq_dir = Path(output_folder) / "seqs"
        fasta_files = list(seq_dir.glob("*.fa")) if seq_dir.exists() else []
        
        print("\n" + "="*70)
        print("✓ SUCCESS!")
        print("="*70)
        print(f"\nResults saved to: {output_folder}/seqs/")
        
        if fasta_files:
            print(f"\nGenerated files:")
            for f in fasta_files:
                size = f.stat().st_size
                print(f"  • {f.name} ({size} bytes)")
                
            # Quick preview of first few lines
            first_file = fasta_files[0]
            print(f"\nPreview of {first_file.name}:")
            print("-" * 70)
            with open(first_file) as f:
                for i, line in enumerate(f):
                    if i >= 6:  # Show first 6 lines
                        print("  ...")
                        break
                    print(f"  {line.rstrip()}")
            print("-" * 70)
        
        print(f"\n💡 To view full results: cat {seq_dir}/*.fa")
        print("="*70)
        return True
    else:
        print("\n" + "="*70)
        print("✗ Failed to run ProteinMPNN")
        print("="*70)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Simple ProteinMPNN Runner - Design protein sequences from structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Option 1 (Config file - EASIEST):
  Edit the configuration section at the top of this file, then run:
  python run_proteinmpnn.py

Usage Option 2 (Command line):
  # Basic usage
  python run_proteinmpnn.py my_protein.pdb
  
  # Generate 20 sequences
  python run_proteinmpnn.py my_protein.pdb --num 20
  
  # Conservative design (high similarity to input)
  python run_proteinmpnn.py my_protein.pdb --conservative
  
  # Diverse designs
  python run_proteinmpnn.py my_protein.pdb --diverse --num 50
  
  # Reproducible run
  python run_proteinmpnn.py my_protein.pdb --num 10 --seed 42

Temperature Guide:
  --conservative  → temp 0.05  (very similar to input)
  default         → temp 0.1   (balanced)
  --diverse       → temp 0.3   (more exploration)
        """
    )
    
    # Positional argument (now optional if config is set)
    parser.add_argument(
        "pdb",
        type=str,
        nargs='?',  # Make it optional
        help="Path to PDB file (e.g., my_protein.pdb) - optional if set in config"
    )
    
    # Optional arguments
    parser.add_argument(
        "--num",
        type=int,
        default=10,
        metavar="N",
        help="Number of sequences to generate (default: 10)"
    )
    
    parser.add_argument(
        "--out",
        type=str,
        metavar="DIR",
        help="Output directory (default: auto-generated with timestamp)"
    )
    
    parser.add_argument(
        "--temp",
        type=float,
        metavar="T",
        help="Sampling temperature 0.05-0.5 (default: 0.1)"
    )
    
    parser.add_argument(
        "--conservative",
        action="store_true",
        help="Use conservative settings (temp=0.05)"
    )
    
    parser.add_argument(
        "--diverse",
        action="store_true",
        help="Use diverse settings (temp=0.3)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        metavar="SEED",
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        metavar="N",
        help="Batch size (reduce if out of memory)"
    )
    
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Force re-run dependency installation"
    )
    
    args = parser.parse_args()
    
    # Use config values if command line arguments not provided
    pdb_input = args.pdb if args.pdb else PDB_FILE
    num_sequences = args.num if args.num != 10 or args.pdb else NUM_SEQUENCES
    seed = args.seed if args.seed is not None else SEED
    output = args.out if args.out else OUTPUT_FOLDER
    batch = args.batch_size if args.batch_size is not None else BATCH_SIZE
    
    # Determine temperature
    if args.conservative:
        temperature = 0.05
    elif args.diverse:
        temperature = 0.3
    elif args.temp is not None:
        temperature = args.temp
    else:
        # Use config temperature
        if TEMPERATURE_MODE == "conservative":
            temperature = 0.05
        elif TEMPERATURE_MODE == "diverse":
            temperature = 0.3
        elif TEMPERATURE_MODE == "default":
            temperature = 0.1
        elif isinstance(TEMPERATURE_MODE, (int, float)):
            temperature = TEMPERATURE_MODE
        else:
            temperature = 0.1
    
    # Check PDB file exists
    if not pdb_input:
        print("\n✗ Error: No PDB file specified!")
        print("\nEither:")
        print("  1. Set PDB_FILE in the configuration section at the top of this file")
        print("  2. Provide PDB file as argument: python run_proteinmpnn.py my_protein.pdb")
        sys.exit(1)
    
    pdb_path = Path(pdb_input)
    if not pdb_path.exists():
        print(f"\n✗ Error: PDB file not found: {pdb_path}")
        print(f"\nLooking in: {pdb_path.absolute()}")
        print("\nTip: Set PDB_FILE in the config or use an example:")
        print("  python run_proteinmpnn.py inputs/PDB_monomers/pdbs/5L33.pdb")
        sys.exit(1)
    
    # Print configuration being used
    print("\n" + "="*70)
    print("ProteinMPNN Configuration")
    print("="*70)
    print(f"PDB File:    {pdb_path}")
    print(f"Sequences:   {num_sequences}")
    print(f"Temperature: {temperature}")
    if seed is not None:
        print(f"Seed:        {seed}")
    else:
        print(f"Seed:        Random")
    print("="*70)
    
    # Determine output folder
    if output:
        output_folder = output
    else:
        output_folder = get_output_folder(pdb_path)
    
    # Check dependencies (skip if --setup)
    if not args.setup and not check_dependencies():
        print("\n⚠ Dependencies not installed.")
        response = input("Install now? [Y/n]: ").strip().lower()
        if response in ['', 'y', 'yes']:
            if not install_dependencies():
                print("\n✗ Installation failed")
                sys.exit(1)
        else:
            print("\nPlease install dependencies first:")
            print("  python quick_install.py")
            sys.exit(1)
    elif args.setup:
        if not install_dependencies():
            print("\n✗ Installation failed")
            sys.exit(1)
    
    # Run ProteinMPNN
    success = run_proteinmpnn(
        pdb_path=pdb_path,
        output_folder=output_folder,
        num_sequences=num_sequences,
        temperature=temperature,
        seed=seed,
        batch_size=batch
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()

