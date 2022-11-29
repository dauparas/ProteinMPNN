#!/bin/bash
#SBATCH -p gpu
#SBATCH --mem=32g
#SBATCH --gres=gpu:rtx2080:1
#SBATCH -c 3
#SBATCH --output=example_3_from_fasta.out

source activate mlfold

path_to_PDB="../inputs/PDB_complexes/pdbs/3HTN.pdb"
path_to_fasta="/home/justas/projects/github/ProteinMPNN/outputs/example_3_outputs/seqs/3HTN.fa"

output_dir="../outputs/example_3_score_only_from_fasta_outputs"
if [ ! -d $output_dir ]
then
    mkdir -p $output_dir
fi

chains_to_design="A B"

python ../protein_mpnn_run.py \
        --path_to_fasta $path_to_fasta \
        --pdb_path $path_to_PDB \
        --pdb_path_chains "$chains_to_design" \
        --out_folder $output_dir \
        --num_seq_per_target 5 \
        --sampling_temp "0.1" \
        --score_only 1 \
        --seed 13 \
        --batch_size 1
