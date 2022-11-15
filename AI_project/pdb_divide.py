# https://stackoverflow.com/questions/11685716/how-to-extract-chains-from-a-pdb-file

import glob
import pathlib

PDB_DIR = "/mnt/P41/Repositories/ProteinMPNN/RFP_designing/RFP_query/"

import os
from Bio.PDB import *

from tqdm import tqdm


class FlatFile:
    # Instance variables
    def __init__(self, id=str(), path="."):
        self.id = id
        self.path = path
        self.lines = list()

    # methods : downloading and reading flat files (PDB file or CSV)
    def download_pdb(self, pdb_id, output_dir="."):
        """Download a PDB file with Biopython PDBList class. Returns the donwloaded
        file path.
        /!\ the Biopython function assings the format name : 'pdb<pdb_id>.ent'
        """
        if self.path != ".":
            output_dir = self.path
        pdb_file = PDBList()
        pdb_file.retrieve_pdb_file(pdb_id, pdir=output_dir, file_format="pdb")
        file_name = "pdb" + pdb_id.lower() + ".ent"
        self.id = pdb_id
        self.path = output_dir + file_name

    def read_file(self, path="."):
        """Read a flat file. Assigns a lines list to a lines attribute. This
        fonction is used by CSV and PDB files.
        """
        if path != ".":
            self.path = path
        f = open(self.path, "r")
        lines = f.readlines()
        f.close()
        self.lines = lines

    def split_PDBfile_by_chains(
        self, output_dir=".", chains="all", all_sections=True, log_file=None
    ):
        """Split a pdb file in different pdb files by chains. data is a list of
        pdb file lines. chains must be a list of PDB ids (e.g. ['A', 'B'])
        """
        # Defensive programming4
        if self.path != ".":
            output_dir = self.path

        pdblines = self.lines
        # file split :
        initial_sections = list()
        dict_chains = dict()
        final_sections = list()
        i = 0
        while i < len(pdblines):
            line = pdblines[i]
            if line[0:4] != "ATOM" and line[0:3] != "TER":
                initial_sections.append(line)
                i += 1
            else:
                break
        while i < len(pdblines):
            line = pdblines[i]
            possible_sections = ["ATOM  ", "ANISOU", "TER   ", "HETATM"]
            if line[0:6] in possible_sections:
                chain_id = line[21]
                if not (chain_id in dict_chains):
                    dict_chains[chain_id] = [line]
                else:
                    dict_chains[chain_id].append(line)
                i += 1
            else:
                break
        while i < len(pdblines):
            line = pdblines[i]
            final_sections.append(line)
            i += 1

        # Chains selection :
        if chains == "all":
            chains_id_list = dict_chains.keys()
            print("splitted", dict_chains.keys())
        else:
            chains_id_list = sorted(chains)
        pdb_id = self.id

        target_to_remove = self.path

        self.id = list()
        self.path = list()

        # Write the different files
        for chain_id in chains_id_list:
            # sub_file_id = pdb_id + "_" + chain_id
            # sub_file_name = "pdb" + sub_file_id + ".pdb"
            # sub_file_path = output_dir + sub_file_name
            sub_file_id = pdb_id + "_" + chain_id
            # sub_file_name = "pdb" + sub_file_id + ".pdb"
            sub_file_path = pathlib.Path(output_dir).parent / f"{sub_file_id}.pdb"
            f = open(sub_file_path, "w")
            if all_sections:
                f.writelines(initial_sections)
            f.writelines(dict_chains[chain_id])
            if all_sections:
                f.writelines(final_sections)
            f.close()
            self.id.append((pdb_id, chain_id))
            self.path.append(sub_file_path)

        # os.remove(target_to_remove)
        if log_file is not None:
            log_file.write(f"{target_to_remove}\n")


def main():
    """Parses PDB id's desired chains, and creates new PDB structures."""
    # import sys

    # if not len(sys.argv) == 2:
    #     print("Usage: $ python %s 'pdb.txt'" % __file__)
    #     sys.exit()

    # pdb_textfn = sys.argv[1]

    import glob
    import pathlib
    import MDAnalysis

    # load multiple pdb files
    input_pdb = glob.glob(PDB_DIR + "*.pdb")

    pdb_ids = open("./AI_project/input_preprocessing/mmtf_dataset.list", "w")
    for pdb in tqdm(input_pdb):
        pdb_id = os.path.splitext(pathlib.Path(pdb).name)[0]

        f = FlatFile(path="./AI_project/input_preprocessing/")

        # DEV
        # u = MDAnalysis.Universe(pdb)
        # chains = u.atoms.fragments
        # for ch in chains:
        #     print(ch)
        f.download_pdb(pdb_id)
        f.read_file()
        f.split_PDBfile_by_chains(chains="all", log_file=pdb_ids)
    pdb_ids.close()


if __name__ == "__main__":
    main()


# '''
# from Bio.PDB import parse_pdb_header

# with open("1XAE.pdb","r") as handle:
#     header_dict = parse_pdb_header(handle)

# print(header_dict)

# #first_model = structure[0]
# '''
# '''
# from Bio.PDB.PDBParser import PDBParser
# p = PDBParser(PERMISSIVE=1)
# structure_id = "1XAE"
# filename = "1XAE.pdb"
# structure = p.get_structure(structure_id, filename)

# resolution = structure.header['resolution']
# keywords = structure.header['keywords']
# '''

# from Bio.PDB import Select, PDBIO
# from Bio.PDB.PDBParser import PDBParser
# from Bio.PDB import parse_pdb_header
# class ChainSelect(Select):
#     def __init__(self, chain):
#         self.chain = chain

#     def accept_chain(self, chain):
#         if chain.get_id() == self.chain:
#             return 1
#         else:
#             return 0

# #chains = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
# p = PDBParser(PERMISSIVE=1)


# filename = '2GX2.pdb'
# structure = p.get_structure(filename, filename)
# with open("2GX2.pdb","r") as handle:
#     header_dict = parse_pdb_header(handle)

# tmp = header_dict['compound']
# tmpp = tmp['1']
# kl = []
# #print(tmpp['chain'])

# for tmp in tmpp['chain']:
#     if(tmp<'a' or tmp>'z'):
#         continue
#     kl.append(tmp.upper())
# #print(kl)
# for chain in kl:
#     pdb_chain_file ='{}_{}.pdb'.format(filename,chain)
#     io_w_no_h = PDBIO()
#     io_w_no_h.set_structure(structure)
#     io_w_no_h.save('{}'.format(pdb_chain_file), ChainSelect(chain))
