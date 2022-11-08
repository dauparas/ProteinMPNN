#!/bin/bash
#SBATCH -p gpu
#SBATCH --mem=32g
#SBATCH --gres=gpu:rtx2080:1
#SBATCH -c 2
#SBATCH --output=example_1.out

source activate mlfold

folder_with_pdbs="../inputs/mRFP#132777/"

output_dir="../outputs/RFP_design"
if [ ! -d $output_dir ]; then
    mkdir -p $output_dir
fi

path_for_parsed_chains=$output_dir"/parsed_pdbs.jsonl"

PYTHON=~/anaconda3/envs/DeepCpf1_torch/bin/python3

${PYTHON} ../helper_scripts/parse_multiple_chains.py --input_path=$folder_with_pdbs --output_path=$path_for_parsed_chains

${PYTHON} ../protein_mpnn_run.py \
    --jsonl_path $path_for_parsed_chains \
    --out_folder $output_dir \
    --num_seq_per_target 10 \
    --sampling_temp "0.1" \
    --seed 37 \
    --batch_size 1
