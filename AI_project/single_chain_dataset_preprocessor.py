import glob
import os
import pathlib
import pickle

import pandas as pd
from mmtf import fetch

from tqdm import tqdm

PDB_DIR = "/mnt/P41/Repositories/ProteinMPNN/AI_project/input_preprocessing/"


delete_list = []
deviants = []

headers = []
targets = []
for pdb in sorted(glob.glob(PDB_DIR + "*.pdb")):
    with open(pdb, "r") as p:

        temp = p.readline().split()

        # DEV
        if len(temp) != 5:
            # handler Z
            deviants.append(temp)
            delete_list.append(pdb)
            # headers.append(temp[:1] + [temp[1]+"_"+temp[2]] + temp[3:])
            continue
        headers.append(temp)
        targets.append(os.path.splitext(pathlib.Path(pdb).name)[0].split("_")[0])


# 1st removal
while len(delete_list) > 0:
    os.remove(delete_list.pop())


for item in headers:
    if len(item) != 5:
        print(item)

df = pd.DataFrame(headers, columns=["HEADER", "PROTEIN1", "PROTEIN2", "DATE", "ID"])
print(df.head())

# print(df.groupby(["PROTEIN1"]).count())

for i, row in df.iterrows():
    if row["PROTEIN1"] in ["FLUORESCENT", "LUMINESCENT"]:
        print(row)
    else:
        delete_list.append(row["ID"])

for fname in os.listdir(PDB_DIR):
    if fname.split("_")[0] in set(delete_list):
        os.remove(os.path.join(PDB_DIR, fname))


# for mmtf
s = set(targets)
output = []
for pdb in tqdm(s):
    try:
        decoded_data = fetch(pdb)
        print(
            "PDB Code: "
            + str(decoded_data.structure_id)
            + " has "
            + str(decoded_data.num_chains)
            + " chains"
        )
        for chain in set(decoded_data.chain_name_list):
            output.append(f"{pdb.lower()}{chain}")
    except Exception as e:
        print(e)

with open("mmtf_dataset.list", "w") as f:
    f.writelines([f"{t}\n" for t in output])
