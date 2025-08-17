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
from pathlib import Path
from glob import glob 
import string
from Bio.PDB import MMCIFParser,PDBIO, PDBList, Structure, Model

all_ids = list(string.ascii_uppercase + string.ascii_lowercase + string.digits)

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
    


def save_pdb_file(
    file_path: str, chain_of_interest:str, output_path: str):
    """
    Save as .pdb with renamed chains to label_asym id instead of auth_asym id.

    Args:
        chain_i: chain id of interest in the first cif file
        chain_j: chain id of interest in the second cif file
    returns:
        new_chain_i: the renamed chain id for chain of interest i in the combined structure
        new_chain_j: the renamed chain id for chain of interset j in the combined structure
    """
    parser = MMCIFParser(auth_chains=False)

    structure = parser.get_structure("struct", filename=file_path)

    model = structure.get_list()[0]
    
    new_structure = Structure.Structure("Single")

    model_new = Model.Model(0)

    total_chains = 0

    for chain in model.get_chains():
        new_chain = chain.copy()
        new_chain.id = all_ids[total_chains]
        if chain.id == chain_of_interest:
            new_chain_of_interest = new_chain.id
        model_new.add(new_chain)
        total_chains += 1

    new_structure.add(model_new)

    io = PDBIO()
    io.set_structure(new_structure)
    io.save(output_path)

    return new_chain_of_interest



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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

    DICT_ID_TO_ASSEMBLY_IDX = {
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

    script_directory_2 = "one_conf_pdb_scripts"
    output_directory = "one_conf_pdb_files"

    os.makedirs(script_directory_2, exist_ok=True)
    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(args.unprocessed_struct_dir, exist_ok=True)
    
    # download the bioassembly cif files
    for id in ID_TO_CHAIN.keys():
        os.makedirs(os.path.join(output_directory, id), exist_ok=True)
        download_pdb_file(id, os.path.join(output_directory, id), file_format="mmCif", retrieve_assembly=True, retrieve_struct=False, assembly_num=DICT_ID_TO_ASSEMBLY_IDX.get(id, 0) + 1)

        """ #rename with ".pdb" extension
        for file in Path(os.path.join(output_directory, id)).iterdir():
            print("here")
            name = file.name
            id, extension = name.split(".pdb", 1)
            if extension: 
                name = id + ".pdb"
                path = file.with_name(name)
                file.rename(path)"""
       

    for idx, id in enumerate(ID_TO_CHAIN.keys()):
        print(f"creating script for {id}")

        assembly = glob(os.path.join(output_directory, id, id + "-assembly*.cif"))

        chain_to_design = save_pdb_file(
            assembly[0],
            ID_TO_CHAIN[id],
            os.path.join(output_directory, id, id + ".pdb"),
        )
        
        # copy the script to the output directory
        shutil.copyfile(
            "template_proteinmpnn_job_single.sh",
            os.path.join(script_directory_2, f"proteinmpnn_job_{id}.sh"),
        )
        # read the contents of the script into a string
        with open(
            os.path.join(script_directory_2, f"proteinmpnn_job_{id}.sh"), "r"
        ) as file:
            script_contents = file.read()
        script_contents = script_contents.replace(
            "folder_with_pdbs=",
            "folder_with_pdbs=" + os.path.join(".", output_directory, id),
        )
        script_contents = script_contents.replace(
            "output_dir=", "output_dir=" + os.path.join(".", args.output_dir, id)
        )
        script_contents = script_contents.replace(
            "chains_to_design=", f'chains_to_design="{chain_to_design}"',
        )
        # write script contents
        with open(
            os.path.join(script_directory_2, f"proteinmpnn_job_{id}.sh"), "w"
        ) as file:
            file.write(script_contents)
