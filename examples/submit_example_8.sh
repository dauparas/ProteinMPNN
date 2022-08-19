#!/bin/bash
#SBATCH -p gpu
#SBATCH --mem=32g
#SBATCH --gres=gpu:rtx2080:1
#SBATCH -c 2
#SBATCH --output=example_8.out

source activate mlfold

folder_with_pdbs="../inputs/PDB_monomers/pdbs/"

output_dir="../outputs/example_8_outputs"
if [ ! -d $output_dir ]
then
    mkdir -p $output_dir
fi

path_for_bias=$output_dir"/bias_pdbs.jsonl"
#Adding global polar amino acid bias (Doug Tischer)
AA_list="D E H K N Q R S T W Y"
bias_list="1.39 1.39 1.39 1.39 1.39 1.39 1.39 1.39 1.39 1.39 1.39"
python ../helper_scripts/make_bias_AA.py --output_path=$path_for_bias --AA_list="$AA_list" --bias_list="$bias_list"

path_for_parsed_chains=$output_dir"/parsed_pdbs.jsonl"
python ../helper_scripts/parse_multiple_chains.py --input_path=$folder_with_pdbs --output_path=$path_for_parsed_chains

python ../protein_mpnn_run.py \
        --jsonl_path $path_for_parsed_chains \
        --out_folder $output_dir \
        --bias_AA_jsonl $path_for_bias \
        --num_seq_per_target 2 \
        --sampling_temp "0.1" \
        --seed 37 \
        --batch_size 1
