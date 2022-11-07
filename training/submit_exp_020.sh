#!/bin/bash

#SBATCH -p gpu
#SBATCH --mem=128g
#SBATCH --gres=gpu:a100:1
#SBATCH -c 12
#SBATCH -t 7-00:00:00
#SBATCH --output=exp_020.out

source activate mlfold-test
python ./training.py \
           --path_for_outputs "./exp_020" \
           --path_for_training_data "path_to/pdb_2021aug02" \
           --num_examples_per_epoch 1000 \
           --save_model_every_n_epochs 50
