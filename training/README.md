# ProteinMPNN
To train/retrain ProteinMPNN clone this github repo and install Python>=3.0, PyTorch, Numpy. 

The multi-chain training data (16.5 GB, PDB biounits, 2021 August 2) can be downloaded from here: `https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02.tar.gz`; The small subsample (47 MB) of this data for testing purposes can be downloaded from here: `https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02_sample.tar.gz`

```
Training set for ProteinMPNN curated by Ivan Anishchanko.

Each PDB entry is represented as a collection of .pt files:
    PDBID_CHAINID.pt - contains CHAINID chain from PDBID
    PDBID.pt         - metadata and information on biological assemblies

PDBID_CHAINID.pt has the following fields:
    seq  - amino acid sequence (string)
    xyz  - atomic coordinates [L,14,3]
    mask - boolean mask [L,14]
    bfac - temperature factors [L,14]
    occ  - occupancy [L,14] (is 1 for most atoms, <1 if alternative conformations are present)

PDBID.pt:
    method        - experimental method (str)
    date          - deposition date (str)
    resolution    - resolution (float)
    chains        - list of CHAINIDs (there is a corresponding PDBID_CHAINID.pt file for each of these)
    tm            - pairwise similarity between chains (TM-score,seq.id.,rmsd from TM-align) [num_chains,num_chains,3]
    asmb_ids      - biounit IDs as in the PDB (list of str)
    asmb_details  - how the assembly was identified: author, or software, or smth else (list of str)
    asmb_method   - PISA or smth else (list of str)

    asmb_chains    - list of chains which each biounit is composed of (list of str, each str contains comma separated CHAINIDs)
    asmb_xformIDX  - (one per biounit) xforms to be applied to chains from asmb_chains[IDX], [n,4,4]
                     [n,:3,:3] - rotation matrices
                     [n,3,:3] - translation vectors

list.csv:
   CHAINID    - chain label, PDBID_CHAINID
   DEPOSITION - deposition date
   RESOLUTION - structure resolution
   HASH       - unique 6-digit hash for the sequence
   CLUSTER    - sequence cluster the chain belongs to (clusters were generated at seqID=30%)
   SEQUENCE   - reference amino acid sequence

valid_clusters.txt - clusters used for validation

test_clusters.txt - clusters used for testing
```

Code organization:
* `training.py` - the main script to train the model
* `model_utils.py` - utility functions and classes for the model
* `utils.py` - utility functions and classes for data loading
* `exp_020/` - sample outputs
* `submit_exp_020.sh` - sample SLURM submit script
-----------------------------------------------------------------------------------------------------
Input flags for `training.py`:
```
    argparser.add_argument("--path_for_training_data", type=str, default="my_path/pdb_2021aug02", help="path for loading training data")
    argparser.add_argument("--path_for_outputs", type=str, default="./test", help="path for logs and model weights")
    argparser.add_argument("--previous_checkpoint", type=str, default="", help="path for previous model weights, e.g. file.pt")
    argparser.add_argument("--num_epochs", type=int, default=200, help="number of epochs to train for")
    argparser.add_argument("--save_model_every_n_epochs", type=int, default=10, help="save model weights every n epochs")
    argparser.add_argument("--reload_data_every_n_epochs", type=int, default=2, help="reload training data every n epochs")
    argparser.add_argument("--num_examples_per_epoch", type=int, default=1000000, help="number of training example to load for one epoch")
    argparser.add_argument("--batch_size", type=int, default=10000, help="number of tokens for one batch")
    argparser.add_argument("--max_protein_length", type=int, default=10000, help="maximum length of the protein complext")
    argparser.add_argument("--hidden_dim", type=int, default=128, help="hidden model dimension")
    argparser.add_argument("--num_encoder_layers", type=int, default=3, help="number of encoder layers")
    argparser.add_argument("--num_decoder_layers", type=int, default=3, help="number of decoder layers")
    argparser.add_argument("--num_neighbors", type=int, default=48, help="number of neighbors for the sparse graph")
    argparser.add_argument("--dropout", type=float, default=0.1, help="dropout level; 0.0 means no dropout")
    argparser.add_argument("--backbone_noise", type=float, default=0.2, help="amount of noise added to backbone during training")
    argparser.add_argument("--rescut", type=float, default=3.5, help="PDB resolution cutoff")
    argparser.add_argument("--debug", type=bool, default=False, help="minimal data loading for debugging")
    argparser.add_argument("--gradient_norm", type=float, default=-1.0, help="clip gradient norm, set to negative to omit clipping")
    argparser.add_argument("--mixed_precision", type=bool, default=True, help="train with mixed precision")
```
-----------------------------------------------------------------------------------------------------
For example to make a conda environment to run ProteinMPNN:
* `conda create --name mlfold` - this creates conda environment called `mlfold`
* `source activate mlfold` - this activate environment
* `conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch` - install pytorch following steps from https://pytorch.org/
-----------------------------------------------------------------------------------------------------
Models provided for the vanilla MPNN were trained with default flags:
* `v_48_002.pt` - `--num_neighbors 48 --backbone_noise 0.02 --num_epochs 150`
* `v_48_010.pt` - `--num_neighbors 48 --backbone_noise 0.10 --num_epochs 150`
* `v_48_020.pt` - `--num_neighbors 48 --backbone_noise 0.20 --num_epochs 150`
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
