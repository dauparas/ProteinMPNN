# ProteinMPNN

Deep learning-based protein sequence design using ProteinMPNN.

## Installation

```bash
# Create conda environment
conda create -n proteinmpnn python=3.9 -y
conda activate proteinmpnn

# Install dependencies
python quick_install.py
```

## Usage

### Simple Method (Recommended)

Edit the configuration in `run_proteinmpnn.py`:

```python
# Configuration
PDB_FILE = "inputs/PDB_monomers/pdbs/5L33.pdb"
NUM_SEQUENCES = 10
TEMPERATURE_MODE = "default"  # "conservative", "default", "diverse", or custom number
SEED = 42
```

Then run:

```bash
python run_proteinmpnn.py
```

### Command Line Method

```bash
python run_proteinmpnn.py my_protein.pdb --num 20 --seed 42
```

### Advanced Method

```bash
python protein_mpnn_run.py \
    --pdb_path inputs/PDB_monomers/pdbs/5L33.pdb \
    --out_folder outputs/my_design \
    --num_seq_per_target 10 \
    --sampling_temp 0.1
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--num` | Number of sequences to generate | 10 |
| `--temp` | Sampling temperature (0.05-0.5) | 0.1 |
| `--seed` | Random seed for reproducibility | None |
| `--conservative` | Use conservative mode (temp 0.05) | - |
| `--diverse` | Use diverse mode (temp 0.3) | - |

## Temperature Guide

- **0.05** (conservative): High similarity to input
- **0.1** (default): Balanced design
- **0.3** (diverse): More sequence exploration

## Output

Results are saved in `outputs/` as FASTA files containing:
- Original sequence
- Designed sequences with scores
- Sequence recovery metrics

## Citation

```bibtex
@article{dauparas2022robust,
  title={Robust deep learning--based protein sequence design using ProteinMPNN},
  author={Dauparas, Justas and Anishchenko, Ivan and Bennett, Nathaniel and Bai, Hua and Ragotte, Robert J and Milles, Lukas F and Wicky, Basile IM and Courbet, Alexis and de Haas, Rob J and Bethel, Neville and others},
  journal={Science},
  volume={378},
  number={6615},
  pages={49--56},
  year={2022},
  publisher={American Association for the Advancement of Science}
}
```

## License

MIT License

