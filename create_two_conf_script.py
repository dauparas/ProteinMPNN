"""from sampling.core.data import blob_storage
from multiconfmpnn.data.dataset import MultiConfCATHDataset"""
from itertools import combinations
import mdtraj
import os
import shutil
import pandas as pd
import argparse
import numpy as np
import h5py
from bioemu_benchmarks.benchmarks import Benchmark
from Bio import Align
from Bio.PDB import MMCIFParser, Structure, Model, PDBIO
import string
from Bio.PDB.PDBList import PDBList
from glob import glob

all_ids = list(string.ascii_uppercase + string.ascii_lowercase + string.digits)


def combine_two_pdb_files(file_path_one: str, file_path_two: str, chain_i: str, chain_j: str, output_path:str) -> tuple[str, str]:
    """
    Combines structures for two input bioassembly cif files and returns the renamed chain ids for provided chains

    Args:
        chain_i: chain id of interest in the first cif file
        chain_j: chain id of interest in the second cif file
    returns:
        new_chain_i: the renamed chain id for chain of interest i in the combined structure
        new_chain_j: the renamed chain id for chain of interset j in the combined structure
    """

    parser = MMCIFParser(auth_chains = False)

    structure_one = parser.get_structure("struct_one", filename=file_path_one)

    model_one = structure_one.get_list()[0]

    structure_two = parser.get_structure("struct_two", filename=file_path_two)

    for atom in structure_two.get_atoms():
        atom.transform([[1,0,0],[0,1,0],[0,0,1]], [500, 500, 500]) #500 angstroms

    model_two = structure_two.get_list()[0]

    new_structure = Structure.Structure("Combined")

    model = Model.Model(0)

    total_chains = 0

    for chain in model_one.get_chains():
        new_chain = chain.copy()
        new_chain.id = all_ids[total_chains]
        if chain.id == chain_i:
            new_chain_i = new_chain.id
        model.add(new_chain)
        total_chains += 1

    for chain in model_two.get_chains():
        new_chain = chain.copy()
        new_chain.id = all_ids[total_chains]
        if chain.id == chain_j:
            new_chain_j = new_chain.id
        model.add(new_chain)
        total_chains += 1

    new_structure.add(model)

    io = PDBIO()
    io.set_structure(new_structure)
    io.save(output_path)

    return new_chain_i, new_chain_j


def download_pdb_file(pdb_id:str, output_dir:str, file_format:str="mmCif", retrieve_assembly:bool=False, retrieve_struct = True, assembly_num:int=1):
    """ Downloads a PDB file from the RCSB PDB database """

    pdbl =PDBList()

    if retrieve_struct:
        pdbl.retrieve_pdb_file(
                pdb_id, pdir=output_dir, file_format= file_format
            )
        
    if retrieve_assembly:
        pdbl.retrieve_assembly_file(
            pdb_id, pdir=output_dir, file_format= file_format, 
            assembly_num= assembly_num
        )
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed_data_location",
        type=str,
        default=".",
        help="Directory storing processed data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to store proteinMPNN data",
    )
    parser.add_argument(
        "--unprocessed_struct_dir",
        type=str,
        default=".",
        help="Directory containing unprocessed structures for multiconf dataset",
    )
    args = parser.parse_args()


    # some pdb entries have multiple assemblies, which do not always contain the chain of interest. TRAJ loads by chains, so should not need this
    dict_id_to_assembly_idx = {
        # domain motion
        "1k23": 1,  # ->chain 1 in assembly 2
        # cryptic pocket
        "1pzy": 1,
        "2hka": 2,
        "1ha3": 1,
        "1k5h": 1,
        "1ecj": 1,
        "3ixj": 2,
        "3hqp": 15,
        "3peo": 1,
        "2bys": 1,
        "1ank": 1,
    }

    ### load test cases from bioemu-benchmarks ###
    # some benchmarks sets have repeat ids with different chains of interest
    domain_motion_df = pd.read_csv(
        os.path.join(Benchmark("multiconf_domainmotion").asset_dir, "references.csv")
    )
    ID_TO_CHAIN = dict(
        {
            row["pdbidchain_i"].split("_")[0].lower(): row["pdbidchain_i"].split("_")[1]
            for _, row in domain_motion_df.iterrows()
        },
        **{
            row["pdbidchain_j"].split("_")[0].lower(): row["pdbidchain_j"].split("_")[1]
            for _, row in domain_motion_df.iterrows()
        },
    )

    ### load benchmark metadata ###
    dataset_multiconf = pd.read_csv(
        os.path.join(
            args.processed_data_location,
            "processed_bioemu_benchmark",
            "domain_motion",
            "multiconf_metadata.csv",
        )
    )

    id_pairs = dataset_multiconf["pdb_id_i_j"].to_numpy()

    polymer_idx_i = [i.split(",") for i in dataset_multiconf["polymer_alignment_idx_i"].to_numpy()]
    polymer_idx_j = [i.split(",") for i in dataset_multiconf["polymer_alignment_idx_j"].to_numpy()]
    sequence_idx_i = [i.split(",") for i in dataset_multiconf["sequence_alignment_idx_i"].to_numpy()]
    sequence_idx_j = [i.split(",") for i in dataset_multiconf["sequence_alignment_idx_j"].to_numpy()]


    output_directory_2 = "two_conf_pdb_files"
    script_directory_2 = "two_conf_pdb_scripts"

    os.makedirs(output_directory_2, exist_ok=True)
    os.makedirs(script_directory_2, exist_ok=True)
    os.makedirs(args.unprocessed_struct_dir, exist_ok=True)


    chains_to_design: list[tuple[str, str]] = []  

    # download the bioassembly cif files
    for id in ID_TO_CHAIN.keys():
        download_pdb_file(id, args.unprocessed_struct_dir, file_format="mmCif", retrieve_assembly=True, retrieve_struct=False, assembly_num=dict_id_to_assembly_idx.get(id, 0) + 1)


    for idx, id in enumerate(id_pairs):
        id_i = id.split("_")[0]
        id_j = id.split("_")[1]

        print(f"creating script for {id}")

        assembly_pair = [glob(os.path.join(args.unprocessed_struct_dir, id_i + "-assembly*.cif")),
                            glob(os.path.join(args.unprocessed_struct_dir, id_j + "-assembly*.cif"))]
        
        # combine structures and get chain id to design
        os.makedirs(os.path.join(output_directory_2, id), exist_ok=True)
        chain_i, chain_j = combine_two_pdb_files(assembly_pair[0][0], assembly_pair[1][0], ID_TO_CHAIN[id_i], ID_TO_CHAIN[id_j], output_directory_2 + f"/{id}" + f"/{id}.pdb")


        # create tied residue positons
        tied_pos_i = [ 
            polymer_idx_i[idx][i]
            for i, val in enumerate(sequence_idx_i[idx])
            if val in sequence_idx_j[idx]
        ]  
        tied_pos_j = [
            polymer_idx_j[idx][j]
            for j, val in enumerate(sequence_idx_j[idx])
            if val in sequence_idx_i[idx]
        ]  

        tied_pos_i = [str(int(i) + 1) for i in tied_pos_i]
        tied_pos_j = [str(int(i) + 1) for i in tied_pos_j]

        # copy the script to the output directory
        shutil.copyfile(
            "template_proteinmpnn_job.sh",
            os.path.join(script_directory_2, f"proteinmpnn_job_{id}.sh"),
        )
        # read the contents of the script into a string
        with open(
            os.path.join(script_directory_2, f"proteinmpnn_job_{id}.sh"), "r"
        ) as file:
            script_contents = file.read()
        script_contents = script_contents.replace(
            "folder_with_pdbs=",
            "folder_with_pdbs=" + os.path.join(".", output_directory_2, id),
        )
        script_contents = script_contents.replace(
            "output_dir=", "output_dir=" + os.path.join(".", args.output_dir, id)
        )
        script_contents = script_contents.replace(
            "\ntied_positions=",
            '\ntied_positions="'
            + ", ".join([" ".join(tied_pos_i), " ".join(tied_pos_j)])
            + '"',
        )
        script_contents = script_contents.replace(
            "chains_to_design=",
            f'chains_to_design="{chain_i} {chain_j}"',
        )  ### CHECK AND TEST THIS sq

        # write script contents
        with open(
            os.path.join(script_directory_2, f"proteinmpnn_job_{id}.sh"), "w"
        ) as file:
            file.write(script_contents)
