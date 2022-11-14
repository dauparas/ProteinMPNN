# run multiple fa on alphafold automatically
import glob
import subprocess
import shlex
import sys

""" 
python3 docker/run_docker.py --fasta_paths=./protein_sequence/REC3.fasta --max_template_date=2021-11-01  --model_preset=monomer --data_dir=/mnt/WD/alphafold_DB/
"""
python = "/home/dengarden/anaconda3/envs/mlfold_v3.9/bin/python"
alphafold_script_dir = "/mnt/P41/Repositories/alphafold/docker/run_docker.py"
alphafold_db_dir = "/mnt/WD/alphafold_DB/"
fasta_dir = "./vanila_model/separated/"
max_template_date = "2021-11-01"
model_preset = "monomer"
output_dir = (
    "/mnt/P41/Repositories/ProteinMPNN/AI_project/vanila_model/AlphaFold_prediction"
)

# print(glob.glob(fasta_dir + "*.fa"))

for fasta in sorted(glob.glob(fasta_dir + "*.fa")):
    # print(fasta)
    cmd = f"{python} {alphafold_script_dir} --fasta_paths={fasta} --max_template_date={max_template_date} --model_preset={model_preset} --data_dir={alphafold_db_dir} --output_dir={output_dir}"
    print(cmd)
    # p = subprocess.Popen(['ls'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for c in iter(lambda: p.stdout.read(1), b""):
        # TODO : https://stackoverflow.com/questions/60106146/catching-logger-info-output-in-python-subprocess
        sys.stdout.buffer.write(c)
