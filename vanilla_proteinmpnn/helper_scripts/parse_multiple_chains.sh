#!/bin/bash
#SBATCH --mem=32g
#SBATCH -c 2
#SBATCH --output=parse_multiple_chains.out

source activate mlfold
python parse_multiple_chains.py --input_path='../PDB_complexes/pdbs/' --output_path='../PDB_complexes/parsed_pdbs.jsonl'
