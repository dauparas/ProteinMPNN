import glob
import os
import pathlib
import pandas as pd

from tqdm import tqdm

PDB_PATH = "/mnt/P41/Repositories/ProteinMPNN/training/pdb_2021aug02/pdb/"

match = 0

target_pdbids = []
with open("/mnt/P41/Repositories/ProteinMPNN/training/mmtf_dataset.list", "r") as f:
    targets = f.read().split("\n")[:-1]  # remove last empty line
    target_pdbids = [*set([t[:4].lower() for t in targets])]

    for path in pathlib.Path(PDB_PATH).rglob("*.pt"):
        # print(path)
        candidate = os.path.splitext(path.name)[0].split("_")[0].lower()
        if candidate in target_pdbids:
            # print(candidate)
            match += 1

        else:
            os.remove(path)

        # DEV
        # print(path)

print(match)


"""
Module to remove empty folders recursively. Can be used as standalone script or be imported into existing script.
"""
# https://gist.github.com/jacobtomlinson/9031697
import os, sys


def removeEmptyFolders(path, removeRoot=True):
    "Function to remove empty folders"
    if not os.path.isdir(path):
        return

    # remove empty subfolders
    files = os.listdir(path)
    if len(files):
        for f in files:
            fullpath = os.path.join(path, f)
            if os.path.isdir(fullpath):
                removeEmptyFolders(fullpath)

    # if folder empty, delete it
    files = os.listdir(path)
    if len(files) == 0 and removeRoot:
        print("Removing empty folder:", path)
        os.rmdir(path)


def usageString():
    "Return usage string to be output in error cases"
    return "Usage: %s directory [removeRoot]" % sys.argv[0]


# if __name__ == "__main__":
#   removeRoot = True

#   if len(sys.argv) < 1:
#     print ("Not enough arguments")
#     sys.exit(usageString())

#   if not os.path.isdir(sys.argv[1]):
#     print ("No such directory %s" % sys.argv[1])
#     sys.exit(usageString())

#   if len(sys.argv) == 2 and sys.argv[2] != "False":
#     print ("removeRoot must be 'False' or not set")
#     sys.exit(usageString())
#   else:
#     removeRoot = False

removeEmptyFolders(PDB_PATH, False)


# Modify list.csv for give dataset information to the model training function
df = pd.read_csv(
    "/mnt/P41/Repositories/ProteinMPNN/training/pdb_2021aug02/list.csv", index_col=False
)

# print(df.describe())
# print(df.head())

hits = []
for idx, row in tqdm(df.iterrows()):
    if row["CHAINID"][:4].lower() in [t.lower() for t in target_pdbids]:
        # print(row)
        hits.append(row)
    else:
        df.drop(idx, inplace=True)

df.to_csv(
    "/mnt/P41/Repositories/ProteinMPNN/training/pdb_2021aug02/list_filtered.csv",
    index=False,
)
print()
