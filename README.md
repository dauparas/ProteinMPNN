# ProteinMPNN
![ProteinMPNN](https://docs.google.com/drawings/d/e/2PACX-1vTtnMBDOq8TpHIctUfGN8Vl32x5ISNcPKlxjcQJF2q70PlaH2uFlj2Ac4s3khnZqG1YxppdMr0iTyk-/pub?w=889&h=358)
Read [ProteinMPNN paper](https://www.biorxiv.org/content/10.1101/2022.06.03.494563v1).

To run ProteinMPNN install PyTorch. 

Full protein backbone models: `vanilla_proteinmpnn`.

Code organization:
* `vanilla_proteinmpnn/protein_mpnn_run.py` - the main script to initialialize and run the model
* `vanilla_proteinmpnn/protein_mpnn_utils.py` - utility functions for the main script
* `vanilla_proteinmpnn/helper_scripts/` - helper functions to parse PDBs, assign which chains to design, which residues to fix, adding AA bias, tying residues etc.
* `vanilla_proteinmpnn/examples/` - simple code examples
-----------------------------------------------------------------------------------------------------
Input flags:
```
argparser.add_argument("--path_to_model_weights", type=str, default="", help="Path to model weights folder;")
argparser.add_argument("--model_name", type=str, default="v_48_020", help="ProteinMPNN model name: v_48_002, v_48_010, v_48_020, v_48_030; v_48_010=version with 48 edges 0.10A noise")

argparser.add_argument("--save_score", type=int, default=0, help="0 for False, 1 for True; save score=-mean[log_probs] to npy files")
argparser.add_argument("--save_probs", type=int, default=0, help="0 for False, 1 for True; save MPNN predicted probabilites per position")
argparser.add_argument("--score_only", type=int, default=0, help="0 for False, 1 for True; score input backbone-sequence pairs")
argparser.add_argument("--conditional_probs_only", type=int, default=0, help="0 for False, 1 for True; output conditional probabilities p(s_i given the rest of the sequence and backbone)")
argparser.add_argument("--conditional_probs_only_backbone", type=int, default=0, help="0 for False, 1 for True; if true output conditional probabilities p(s_i given backbone)")

argparser.add_argument("--backbone_noise", type=float, default=0.00, help="Standard deviation of Gaussian noise to add to backbone atoms during the inference.")
argparser.add_argument("--num_seq_per_target", type=int, default=1, help="Number of sequences to generate per target.")
argparser.add_argument("--batch_size", type=int, default=1, help="Batch size when using GPUs.")
argparser.add_argument("--max_length", type=int, default=20000, help="Maximum sequence length.")
argparser.add_argument("--sampling_temp", type=str, default="0.1", help="A string of temperatures, 0.1 0.3 0.5. Sampling temperature for amino acids, T=0.0 means taking argmax, T>>1.0 means sampling randomly.")

argparser.add_argument("--out_folder", type=str, help="Path to a folder to output sequences, e.g. /home/out/")
argparser.add_argument("--pdb_path", type=str, default='', help="Path to a single PDB to be designed.")
argparser.add_argument("--pdb_path_chains", type=str, default='', help="Define which chains need to be designed for a single PDB.")
argparser.add_argument("--jsonl_path", type=str, help="Path to a folder with parsed PDBs into jsonl.")
argparser.add_argument("--chain_id_jsonl",type=str, default='', help="Path to a dictionary specifying which chains need to be designed and which ones are fixed, if not specied all chains will be designed.")
argparser.add_argument("--fixed_positions_jsonl", type=str, default='', help="Path to a dictionary with fixed positions.")
argparser.add_argument("--omit_AAs", type=list, default='X', help="Specify which amino acids should be omitted in the generated sequence, e.g. 'AC' would omit alanine and cystine.")
argparser.add_argument("--bias_AA_jsonl", type=str, default='', help="Path to a dictionary which specifies AA composion bias, e.g. {A: -1.1, F: 0.7} would make A less likely and F more likely.")
argparser.add_argument("--bias_by_res_jsonl", default='', help="Path to dictionary with per position bias.")
argparser.add_argument("--omit_AA_jsonl", type=str, default='', help="Path to a dictionary which specifies which amino acids need to be omited from design at specific chain indices.")
argparser.add_argument("--pssm_jsonl", type=str, default='', help="Path to a dictionary with pssm.")
argparser.add_argument("--pssm_multi", type=float, default=0.0, help="A value between [0.0, 1.0], 0.0 means do not use pssm, 1.0 ignore MPNN predictions.")
argparser.add_argument("--pssm_threshold", type=float, default=0.0, help="A value between -inf + inf to restric per position AAs.")
argparser.add_argument("--pssm_log_odds_flag", type=int, default=0, help="0 for False, 1 for True.")
argparser.add_argument("--pssm_bias_flag", type=int, default=0, help="0 for False, 1 for True.")
argparser.add_argument("--tied_positions_jsonl", type=str, default='', help="Path to a dictionary with tied positions for symmetric design.")
```
-----------------------------------------------------------------------------------------------------
Example to design some monomers:
```
folder_with_pdbs="../PDB_monomers/pdbs/"

output_dir="../PDB_monomers/example_1_outputs"
if [ ! -d $output_dir ]
then
    mkdir -p $output_dir
fi

path_for_parsed_chains=$output_dir"/parsed_pdbs.jsonl"

python ../helper_scripts/parse_multiple_chains.py --input_path=$folder_with_pdbs --output_path=$path_for_parsed_chains

python ../protein_mpnn_run.py \
        --jsonl_path $path_for_parsed_chains \
        --out_folder $output_dir \
        --num_seq_per_target 2 \
        --sampling_temp "0.1" \
        --batch_size 1
```
