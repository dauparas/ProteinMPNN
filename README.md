# ProteinMPNN
![ProteinMPNN](https://docs.google.com/drawings/d/e/2PACX-1vTtnMBDOq8TpHIctUfGN8Vl32x5ISNcPKlxjcQJF2q70PlaH2uFlj2Ac4s3khnZqG1YxppdMr0iTyk-/pub?w=889&h=358)
Read [ProteinMPNN paper](https://www.biorxiv.org/content/10.1101/2022.06.03.494563v1).

> **Fork notice:** This is a usability-focused fork of the original ProteinMPNN repository. The model, helper scripts, examples, and upstream reference material are retained below. This fork adds a direct command-line workflow so routine designs do not require calling `protein_mpnn_utils.py` or scripts in `helper_scripts/` from the command line.

## Fork usage

Run commands from the repository root. For a zero-setup command-line workflow, use the `pmpnn` launcher described below; it creates a Conda environment and installs this repository's requirements on first use. `protein_mpnn_run.py` remains available when you prefer to manage Python yourself.

### Design a single structure

```bash
python protein_mpnn_run.py \
  --pdb-path inputs/PDB_complexes/pdbs/3HTN.pdb \
  --out_folder outputs/my_design \
  --num_seq_per_target 8 \
  --sampling_temp "0.1" \
  --batch_size 1
```

All chains are designed by default. Limit design to selected chains with `--design-chains`:

```bash
python protein_mpnn_run.py \
  --pdb-path inputs/PDB_complexes/pdbs/3HTN.pdb \
  --design-chains "A C" \
  --out_folder outputs/my_design
```

### Add constraints directly

Use the following options in the same command; no intermediate JSONL configuration file is needed.

| Goal | Option | Example |
| --- | --- | --- |
| Fix residues | `--fixed-positions` | `"A:1-8,23,25;C:10-20,40"` |
| Tie residues for symmetric sampling | `--tied-positions` | `"A:1=C:1;A:2=C:2"` |
| Bias amino acids | `--aa-bias` | `"D:1.39,E:1.39,H:1.39"` |

Residue indices are one-based positions within each chain, not PDB residue numbers. Separate chains with `;`, positions with `,`, and inclusive ranges with `-`. Each semicolon-separated tied group is sampled together.

For example:

```bash
python protein_mpnn_run.py \
  --pdb-path inputs/PDB_complexes/pdbs/3HTN.pdb \
  --design-chains "A C" \
  --fixed-positions "A:1-8,23,25;C:10-20,40" \
  --tied-positions "A:1=C:1;A:2=C:2" \
  --aa-bias "D:1.39,E:1.39,H:1.39,K:1.39" \
  --out_folder outputs/my_design \
  --num_seq_per_target 8 --sampling_temp "0.1" --batch_size 1
```

### Batch and advanced inputs

Use `--jsonl-path parsed_pdbs.jsonl` to design an existing parsed batch. Inline constraints apply to every target in that batch. The original `*_jsonl` arguments remain available for advanced cases where every target needs different constraints, or when using PSSM/per-residue bias data. They are not required for normal single-PDB runs.

Run `python protein_mpnn_run.py --help` for the complete option list. The existing underscore spellings remain compatible: `--pdb_path`, `--jsonl_path`, and `--pdb_path_chains` (an alias for `--design-chains`).

### Optional command-line launcher

Add this clone's `bin` directory to your system `PATH` to use `pmpnn` from any working directory. The launchers find the repository from their own location, so moving or recloning the repository does not require editing them.

On first use, the launcher creates a Conda environment named `proteinmpnn` and installs [`requirements.txt`](requirements.txt). Conda must be installed and available on `PATH`; no environment activation or manual dependency installation is required. Set `PMPNN_ENV` to use a different environment name, or set `PMPNN_PYTHON` to bypass the automatic setup with a specific Python interpreter.

```bash
# macOS/Linux
export PATH="/path/to/ProteinMPNN/bin:$PATH"
pmpnn --pdb-path /path/to/input.pdb --out_folder ./my_design

# Use a specific interpreter instead of automatic Conda setup
PMPNN_PYTHON=/path/to/python pmpnn --pdb-path /path/to/input.pdb --out_folder ./my_design
```

On Windows, add `C:\path\to\ProteinMPNN\bin` to `Path`, then run `pmpnn` from Command Prompt or PowerShell. For example: `pmpnn --pdb-path C:\path\to\input.pdb --out_folder .\my_design`. Set `PMPNN_PYTHON` to bypass automatic setup with a specific interpreter when needed.

`pmpnn-util` remains only for the legacy batch-parser workflow. Normal designs use `pmpnn` and do not need a utility command.

## Upstream README

To run ProteinMPNN clone this github repo and install Python>=3.0, PyTorch, Numpy. 

Full protein backbone models: `vanilla_model_weights/v_48_002.pt, v_48_010.pt, v_48_020.pt, v_48_030.pt`, `soluble_model_weights/v_48_010.pt, v_48_020.pt`.

CA only models: `ca_model_weights/v_48_002.pt, v_48_010.pt, v_48_020.pt`. Enable flag `--ca_only` to use these models.

Helper scripts: `helper_scripts` - helper functions for advanced workflows and per-target batch configuration. Routine command-line use in this fork does not require invoking them.

Code organization:
* `protein_mpnn_run.py` - the main script to initialialize and run the model.
* `protein_mpnn_utils.py` - utility functions for the main script.
* `examples/` - simple code examples.
* `inputs/` - input PDB files for examples
* `outputs/` - outputs from examples
* `colab_notebooks/` - Google Colab examples
* `training/` - code and data to retrain the model
For example to make a conda environment to run ProteinMPNN:
* `conda create --name mlfold` - this creates conda environment called `mlfold`
* `source activate mlfold` - this activate environment
* `conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch` - install pytorch following steps from https://pytorch.org/
-----------------------------------------------------------------------------------------------------
These are provided `examples/`:
* `submit_example_1.sh` - simple monomer example 
* `submit_example_2.sh` - simple multi-chain example
* `submit_example_3.sh` - directly from the .pdb path
* `submit_example_3_score_only.sh` - return score only (model's uncertainty)
* `submit_example_3_score_only_from_fasta.sh` - return score only (model's uncertainty) loading sequence from fasta files
* `submit_example_4.sh` - fix some residue positions
* `submit_example_4_non_fixed.sh` - specify which positions to design
* `submit_example_5.sh` - tie some positions together (symmetry)
* `submit_example_6.sh` - homooligomer example
* `submit_example_7.sh` - return sequence unconditional probabilities (PSSM like)
* `submit_example_8.sh` - add amino acid bias
* `submit_example_pssm.sh` - use PSSM bias when designing sequences
-----------------------------------------------------------------------------------------------------
Output example:
```
>3HTN, score=1.1705, global_score=1.2045, fixed_chains=['B'], designed_chains=['A', 'C'], model_name=v_48_020, git_hash=015ff820b9b5741ead6ba6795258f35a9c15e94b, seed=37
NMYSYKKIGNKYIVSINNHTEIVKALNAFCKEKGILSGSINGIGAIGELTLRFFNPKTKAYDDKTFREQMEISNLTGNISSMNEQVYLHLHITVGRSDYSALAGHLLSAIQNGAGEFVVEDYSERISRTYNPDLGLNIYDFER/NMYSYKKIGNKYIVSINNHTEIVKALNAFCKEKGILSGSINGIGAIGELTLRFFNPKTKAYDDKTFREQMEISNLTGNISSMNEQVYLHLHITVGRSDYSALAGHLLSAIQNGAGEFVVEDYSERISRTYNPDLGLNIYDFER
>T=0.1, sample=1, score=0.7291, global_score=0.9330, seq_recovery=0.5736
NMYSYKKIGNKYIVSINNHTEIVKALKKFCEEKNIKSGSVNGIGSIGSVTLKFYNLETKEEELKTFNANFEISNLTGFISMHDNKVFLDLHITIGDENFSALAGHLVSAVVNGTCELIVEDFNELVSTKYNEELGLWLLDFEK/NMYSYKKIGNKYIVSINNHTDIVTAIKKFCEDKKIKSGTINGIGQVKEVTLEFRNFETGEKEEKTFKKQFTISNLTGFISTKDGKVFLDLHITFGDENFSALAGHLISAIVDGKCELIIEDYNEEINVKYNEELGLYLLDFNK
>T=0.1, sample=2, score=0.7414, global_score=0.9355, seq_recovery=0.6075
NMYKYKKIGNKYIVSINNHTEIVKAIKEFCKEKNIKSGTINGIGQVGKVTLRFYNPETKEYTEKTFNDNFEISNLTGFISTYKNEVFLHLHITFGKSDFSALAGHLLSAIVNGICELIVEDFKENLSMKYDEKTGLYLLDFEK/NMYKYKKIGNKYVVSINNHTEIVEALKAFCEDKKIKSGTVNGIGQVSKVTLKFFNIETKESKEKTFNKNFEISNLTGFISEINGEVFLHLHITIGDENFSALAGHLLSAVVNGEAILIVEDYKEKVNRKYNEELGLNLLDFNL
```
* `score` - average over residues that were designed negative log probability of sampled amino acids
* `global score` - average over all residues in all chains negative log probability of sampled/fixed amino acids
* `fixed_chains` - chains that were not designed (fixed)
* `designed_chains` - chains that were redesigned
* `model_name/CA_model_name` - model name that was used to generate results, e.g. `v_48_020`
* `git_hash` - github version that was used to generate outputs
* `seed` - random seed
* `T=0.1` - temperature equal to 0.1 was used to sample sequences
* `sample` - sequence sample number 1, 2, 3...etc
-----------------------------------------------------------------------------------------------------
```
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
-----------------------------------------------------------------------------------------------------
