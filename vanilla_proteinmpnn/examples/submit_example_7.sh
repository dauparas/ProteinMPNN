#!/bin/bash
#SBATCH -p gpu
#SBATCH --mem=32g
#SBATCH --gres=gpu:rtx2080:1
#SBATCH -c 3
#SBATCH --output=example_7.out

source activate mlfold

folder_with_pdbs="../PDB_complexes/pdbs/"

output_dir="../PDB_complexes/example_7_outputs"
if [ ! -d $output_dir ]
then
    mkdir -p $output_dir
fi


path_for_parsed_chains=$output_dir"/parsed_pdbs.jsonl"
path_for_assigned_chains=$output_dir"/PDB_complexes/assigned_pdbs.jsonl"
path_for_bias=$output_dir"/bias_pdbs.jsonl"
AA_list="G P A"
bias_list="40.1 0.3 -0.05" #for G P A respectively; global AA bias in the logit space
chains_to_design="A B"


python ../helper_scripts/parse_multiple_chains.py --input_path=$folder_with_pdbs --output_path=$path_for_parsed_chains

python ../helper_scripts/assign_fixed_chains.py --input_path=$path_for_parsed_chains --output_path=$path_for_assigned_chains --chain_list "$chains_to_design"

python ../helper_scripts/make_bias_AA.py --output_path=$path_for_bias --AA_list="$AA_list" --bias_list="$bias_list"

python ../protein_mpnn_run.py \
        --jsonl_path $path_for_parsed_chains \
        --chain_id_jsonl $path_for_assigned_chains \
        --out_folder $output_dir \
        --bias_AA_jsonl $path_for_bias \
        --num_seq_per_target 2 \
        --sampling_temp "0.1" \
        --batch_size 1
