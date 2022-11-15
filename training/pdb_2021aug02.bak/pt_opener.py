import torch
import pickle

PDBID = "/mnt/P41/Repositories/ProteinMPNN/training/pdb_2021aug02/pdb/g7/1g7k.pt"
PDBID_CHAINID = "/mnt/P41/Repositories/ProteinMPNN/training/pdb_2021aug02/pdb/g7/1g7k_A.pt"

# metadata
meta = torch.load(PDBID)
chain = torch.load(PDBID_CHAINID)

print()