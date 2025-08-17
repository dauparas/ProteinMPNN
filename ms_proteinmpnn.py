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


def save_list_of_trajs(trajs, add_amounts, output_path):
    """
    Save a list of trajectories to a single PDB file.
    """
    # get protein indexes in each trajectory
    protein_idxs = [traj.topology.select("protein") for traj in trajs] ## list of indexs where proteins are 

    # translate chains and place them in the same trajectory
    coords = [traj.atom_slice(protein_idxs[i]).xyz + add_amounts[i] for i, traj in enumerate(trajs)] 
    topologies = [traj.atom_slice(protein_idxs[i]).topology for i, traj in enumerate(trajs)] 
    
    # save the new trajectory
    full_coords = np.concatenate(coords, axis=1)
    topology = topologies[0]

    for topology_extra in topologies[1:]:
        topology = topology.join(topology_extra)

    traj = mdtraj.Trajectory(full_coords, topology)
    traj.save(output_path)




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
    """
    # download .pdb file
    download_pdb_entries(
        reference_csv_path=os.path.join(
            args.processed_data_location, "multiconf_metadata.csv"
        ),
        output_dir=args.unprocessed_struct_dir,
        file_type="pdb",
    )"""

      # some pdb entries have multiple assemblies, which do not always contain the chain of interest. TRAJ loads by chains, so should not need this
    dict_id_to_assembly_idx = {
        # domain motion
        "1k23": 1, #->chain 1 in assembly 2
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

    # used to get the correct alignmetn index. Alignment index is based on single bioassembly, so we cannot index using chain names #chains are also not always numbered alphbetically. ex: 2bpg has chain "E" as its chain of interest (furst protein chain)
    CHAIN_TO_REL_IDX = {"1k23": 0, "2bpg":0} 

    # used to get the chain to design based on mdtraj load. mdtraj names chains in order of appearance. Ex: 2bpg chain E is the first protein chain in the structure, so it is chain "A" in mdtraj (if loaded on its own)
    ID_TO_CHAIN_TO_DESIGN ={"2bpg": 0} 

    ### load test cases from bioemu-benchmarks ###
    # some benchmarks sets have repeat ids with different chains of interest
    domain_motion_df =pd.read_csv(os.path.join(Benchmark("multiconf_domainmotion").asset_dir, 'references.csv')) 
    ID_TO_CHAIN = dict({row['pdbidchain_i'].split('_')[0].lower(): row['pdbidchain_i'].split('_')[1] for _, row in domain_motion_df.iterrows()},  **{row['pdbidchain_j'].split('_')[0].lower(): row['pdbidchain_j'].split('_')[1] for _, row in domain_motion_df.iterrows()}) 

    ### load benchmark metadata ###
    dataset_multiconf = pd.read_csv(
        os.path.join(args.processed_data_location, "processed_bioemu_benchmark", "domain_motion", "multiconf_metadata.csv")
    )

    ids = dataset_multiconf["pdb_id_i_j"].to_numpy()
    polymer_indexs_i = dataset_multiconf["polymer_indexs_i"].to_numpy()
    polymer_indexs_j = dataset_multiconf["polymer_indexs_j"].to_numpy()
    sequence_indexs_i = dataset_multiconf["sequence_indexs_i"].to_numpy()
    sequence_indexs_j = dataset_multiconf["sequence_indexs_j"].to_numpy()

    output_directory_2 = "two_conf_pdb_files"
    script_directory_2 = "two_conf_pdb_scripts"

    os.makedirs(output_directory_2, exist_ok=True)
    os.makedirs(script_directory_2, exist_ok=True)

    add_amounts = [0, 5, 10, 15, 20]
    chains_to_design: list[tuple[str, str]] = []  ## list of tuples, each tuple is the chain to design for each conformation
   
    ### load trajs, only protein chains ### 
    for i, id_pairs in enumerate(ids):
        trajs = []

        print("loading id_pairs", id_pairs)
        id_pair = [id for id in id_pairs.split("_")]

        for id in id_pair:
            pdb_dir = os.path.join(args.unprocessed_struct_dir, id+ ".pdb")
            traj = mdtraj.load(os.path.join(pdb_dir)) # load the pdb file with the assembly idx if it exists

            print(f"non spliced traj for {id} ", traj)

            traj_protein = traj.atom_slice(traj.topology.select("protein"))
            trajs.append(traj_protein) 

            print(" traj topology after selecting proteins", [len(list(chain.residues)) for chain in traj_protein.topology.chains])
    
        
        polymer_idx_i = [item.split(",") for item in polymer_indexs_i[i].split("|")]
        polymer_idx_j = [item.split(",") for item in polymer_indexs_j[i].split("|")]
        sequence_idx_i =[item.split(",") for item in sequence_indexs_i[i].split("|")]
        sequence_idx_j = [item.split(",") for item in sequence_indexs_j[i].split("|")] 


        ###retrieve the alignment indexs and chains to design for based on the chain of interest ###
        if id_pair[0] not in CHAIN_TO_REL_IDX.keys():
            idx_i =  ord(ID_TO_CHAIN[id_pair[0]]) - ord('A') # get the polymer indexs for the chain of interest
            print("chain of interest for i:", ID_TO_CHAIN[id_pair[0]])
        else:
            idx_i = CHAIN_TO_REL_IDX[id_pair[0]]
            print("chain of interest for i IF in CHAIN_TO_REL IDX:", idx_i)
        print(f"idx for {id_pair[0]} is", idx_i)

        if id_pair[0] not in ID_TO_CHAIN_TO_DESIGN.keys():
            chain_to_design_i = ID_TO_CHAIN[id_pair[0]]
        else:
            chain_to_design_i = chr(ID_TO_CHAIN_TO_DESIGN[id_pair[0]] + ord("A"))
        print(f"chain to design for {id_pair[0]}, ", chain_to_design_i)
       

        if id_pair[1] not in CHAIN_TO_REL_IDX.keys():
            idx_j = ord(ID_TO_CHAIN[id_pair[1]]) - ord('A') 
            print("chain of interst for j:", ID_TO_CHAIN[id_pair[1]])
        else:
            idx_j = CHAIN_TO_REL_IDX[id_pair[1]]
            print("chain of interst for j IF in CHAIN_TO_REL IDX:", idx_j)
        print(f"idx for {id_pair[1]} is", idx_j)

        if id_pair[1] not in ID_TO_CHAIN_TO_DESIGN.keys():
            chain_to_design_j = chr(ord(ID_TO_CHAIN[id_pair[1]]) + trajs[0].n_chains)
        else:
            chain_to_design_j = chr((ID_TO_CHAIN_TO_DESIGN[id_pair[1]]) + trajs[0].n_chains + ord("A"))# get the polymer indexs for the chain of interest
        print(f"chain to design for {id_pair[1]}, ", chain_to_design_j)

       
        chains_to_design.append((chain_to_design_i, chain_to_design_j))

        ### create tied residue positons ###
        tied_pos_i = [polymer_idx_i[idx_i][i] for i, val in enumerate(sequence_idx_i[idx_i]) if val in sequence_idx_j[idx_j]] # overlapping indices
        tied_pos_j = [polymer_idx_j[idx_j][j] for j, val in enumerate(sequence_idx_j[idx_j]) if val in sequence_idx_i[idx_i]] # overlapping indices

        tied_pos_i = [str(int(i) + 1) for i in tied_pos_i]
        tied_pos_j = [str(int(i) + 1) for i in tied_pos_j]
        print("residx_i", len(tied_pos_j))
        print("residx_j", len(tied_pos_j))
       

    
        """      print("polymer_idx_i", polymer_idx_i[idx_i])
        print("polymer_idx_j", polymer_idx_j[idx_j])
        print("tied_pos_i", tied_pos_i)
        print("tied_pos_j", tied_pos_j)
"""
        """ # testing tied chains for i 
        pdb_dir = os.path.join(args.unprocessed_struct_dir, id_pair[0]+ ".pdb")
        traj_topology = mdtraj.load(os.path.join(pdb_dir)).topology
        print("chain lens:", [len(list(chain.residues)) for chain in traj_topology.chains], "n residues_ i:",
        traj_topology.n_residues)

        #print("residues types", [res.name for res in traj_topology.residues])
        

        # testing tied chains for j
        pdb_dir = os.path.join(args.unprocessed_struct_dir, id_pair[1]+ ".pdb")
        traj_topology = mdtraj.load(os.path.join(pdb_dir)).topology
        print("chain lens:", [len(list(chain.residues)) for chain in traj_topology.chains], "n residues_ j:", traj_topology.n_residues)"""

    
        os.makedirs(os.path.join(output_directory_2, id_pairs), exist_ok=True)

        trajs = []
        for id in id_pair:
            pdb_dir = os.path.join(args.unprocessed_struct_dir, id+ ".pdb")
            traj = mdtraj.load(os.path.join(pdb_dir))
            trajs.extend(mdtraj.load(os.path.join(pdb_dir)))

        # exhaustive combinations of two conformations
        combinations_of_two = list(combinations(trajs, 2))
        for i, combination in enumerate(combinations_of_two):
            save_list_of_trajs(
                combination,
                add_amounts,
                os.path.join(output_directory_2, id_pairs, f"combined_{id_pairs}.pdb"),
            )
        # copy the script to the output directory
        shutil.copyfile(
            "template_proteinmpnn_job.sh",
            os.path.join(script_directory_2, f"proteinmpnn_job_{id_pairs}.sh"),
        )
        # read the contents of the script into a string
        with open(
            os.path.join(script_directory_2, f"proteinmpnn_job_{id_pairs}.sh"), "r"
        ) as file:
            script_contents = file.read()
        script_contents = script_contents.replace(
            "folder_with_pdbs=",
            "folder_with_pdbs=" + os.path.join(".", output_directory_2, id_pairs),
        )
        script_contents = script_contents.replace(
            "output_dir=", "output_dir=" + os.path.join(".", args.output_dir, id_pairs)
        )
        script_contents = script_contents.replace(
            "\ntied_positions=",
            '\ntied_positions="'
            + ", ".join(
                [" ".join(tied_pos_i), " ".join(tied_pos_j)]
            )
            + '"',
        )
        script_contents = script_contents.replace("chains_to_design=", f"chains_to_design=\"{chains_to_design[-1][0]} {chains_to_design[-1][1]}\"") ### CHECK AND TEST THIS sq

        # write script contents
        with open(
            os.path.join(script_directory_2, f"proteinmpnn_job_{id_pairs}.sh"), "w"
        ) as file:
            file.write(script_contents)

