import glob

import pandas as pd

import os

PDB_DIR = "/mnt/P41/Repositories/ProteinMPNN/AI_project/input_preprocessing/"


delete_list = []
headers = []
deviants = []
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
