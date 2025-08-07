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

def save_list_of_trajs(trajs, add_amounts, output_path):
    """
    Save a list of trajectories to a single PDB file.
    """
    # translate chains and place them in the same trajectory
    coords = [traj.xyz + add_amounts[i] for i, traj in enumerate(trajs)]
    topologies = [traj.topology for traj in trajs]
    # save the new trajectory
    full_coords = np.concatenate(coords, axis=1)
    print("topologies", [i.n_residues for i in topologies[0].chains])
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

      # some pdb entries have multiple assemblies, which do not always contain the chain of interest.
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
    CHAIN_TO_REL_IDX = {"1k23": 0, "2bpg":0}
    

    ### load test cases from bioemu-benchmarks
    # some benchmarks sets have repeat ids with different chains of interest
    domain_motion_df =pd.read_csv(os.path.join(Benchmark("multiconf_domainmotion").asset_dir, 'references.csv')) 
    ID_TO_CHAIN = dict({row['pdbidchain_i'].split('_')[0].lower(): row['pdbidchain_i'].split('_')[1] for _, row in domain_motion_df.iterrows()},  **{row['pdbidchain_j'].split('_')[0].lower(): row['pdbidchain_j'].split('_')[1] for _, row in domain_motion_df.iterrows()}) 



    # load the metadata of datset: ie> domain motion
    dataset_multiconf = pd.read_csv(
        os.path.join(args.processed_data_location, "processed_bioemu_benchmark", "domain_motion", "multiconf_metadata.csv")
    )

    ids = dataset_multiconf["pdb_id_i_j"].to_numpy()
    polymer_indexs_i = dataset_multiconf["polymer_indexs_i"].to_numpy()
    polymer_indexs_j = dataset_multiconf["polymer_indexs_j"].to_numpy()
    sequence_indexs_i = dataset_multiconf["sequence_indexs_i"].to_numpy()
    sequence_indexs_j = dataset_multiconf["sequence_indexs_j"].to_numpy()


    handle = h5py.File(
        os.path.join(args.processed_data_location, "processed_bioemu_benchmark", "domain_motion", "multiconf.h5"),
        mode="r",
        libver="latest",
    )

    # load all ground truth sequences

    output_directory_2 = "two_conf_pdb_files"
    script_directory_2 = "two_conf_pdb_scripts"

    os.makedirs(output_directory_2, exist_ok=True)
    os.makedirs(script_directory_2, exist_ok=True)

    add_amounts = [0, 5, 10, 15, 20]

    # create global index for tied positions
    for i, id_pairs in enumerate(ids):

        print("id_pairs", id_pairs)
        group = handle[id_pairs]
        
        polymer_idx_i = [item.split(",") for item in polymer_indexs_i[i].split("|")]
        polymer_idx_j = [item.split(",") for item in polymer_indexs_j[i].split("|")]
        """sequence_idx_i =[item.split(",") for item in sequence_indexs_i[i].split("|")]
        sequence_idx_j = [item.split(",") for item in sequence_indexs_j[i].split("|")] #"""


        id_pair = [id for id in id_pairs.split("_")]
        
        ## getrs the alignment for the chain of interest (starts not global)

        print("polymer_idx_i", (polymer_idx_i))
        print("polymer_idx_j", len(polymer_idx_j))
       
        if id_pair[0] not in CHAIN_TO_REL_IDX.keys():
            polymer_idx_i = polymer_idx_i[ord(ID_TO_CHAIN[id_pair[0]]) - ord('A')] 
        else:
            polymer_idx_i = polymer_idx_i[CHAIN_TO_REL_IDX[id_pair[0]]]

        if id_pair[1] not in CHAIN_TO_REL_IDX.keys():
            polymer_idx_j = polymer_idx_j[ord(ID_TO_CHAIN[id_pair[1]]) - ord('A')] # get the polymer indexs for the chain of interest
        else:
            polymer_idx_j = polymer_idx_j[CHAIN_TO_REL_IDX[id_pair[1]]]

        global_starts= []

        #length of the traj seq up until the chain of interest
        for id in id_pair:
            pdb_dir = os.path.join(args.unprocessed_struct_dir, id+ ".pdb")
            traj_topology = mdtraj.load(os.path.join(pdb_dir)).topology

            global_chain_of_interest_start= 0

            for idx, chain in enumerate(traj_topology.chains):
                print("chain", chain)

                use_rel_idx = id in CHAIN_TO_REL_IDX.keys() # structure is not the first assembly

                if use_rel_idx and idx == CHAIN_TO_REL_IDX[id]:
                    global_starts.append(global_chain_of_interest_start - 1) # zero-indexed
                    break
                if not use_rel_idx and idx == (ord(ID_TO_CHAIN[id]) - ord('A')):
                    global_starts.append(global_chain_of_interest_start - 1)
                    break
                else:
                    global_chain_of_interest_start += len(list(chain.residues))

       
        tied_pos_i = [str(int(i) + global_starts[0] + 2) for i in polymer_idx_i]
        tied_pos_j = [str(int(i) + global_starts[1] + 2) for i in polymer_idx_j]

        print("tied_pos_i", tied_pos_i)
        print("tied_pos_j", tied_pos_j)





    
        """ alignment_idx_i = ord(ID_TOCHAIN[id_pair[0]]) - ord('A') #map chain to index map COI to the index you want ID_TO_CHAINS= {'7ubl': ['A', 'B']} -> we have ID to COI. = {'7ubl: ["A"]} Then,   ID_TO_CHAINS.index(ID_TO_COI[id_pair[0]][0]) returns 0
        alignment_idx_j = ord(ID_TOCHAIN[id_pair[1]]) - ord('A')

        """
        """  pdb_dir = os.path.join(args.unprocessed_struct_dir, id_pair[0]+ ".pdb")
        traj_i_topology = mdtraj.load(os.path.join(pdb_dir)).topology
        sequence = [res.name for res in (traj_i_topology.chain(alignment_idx_i)).residues] #ned the length of sequnces before coi
        """


        """residue_mask_i = group["residue_mask_i"][()]

        chain_of_interest_mask = group["chain_of_interest_mask"][()]

        print("chain_of_interest_mask", chain_of_interest_mask  )

        #global postion of chain of interest based on sequence. not poly,mer

        chain_of_interest_idx_i= np.where(chain_of_interest_mask[:len(residue_mask_i)] == 1,)[0]
        chain_of_interest_idx_j =np.where(chain_of_interest_mask[len(residue_mask_i):] == 1,)[0]

        #conver tsequence indexs to polymer indexs



        for i,k in zip(sequence_idx_i, polymer_idx_i):
            chain_of_interest_idx_i[i] =  


        chain_of_interest_idx_i[polymer_idx_i[alignment_idx_i]]



        chain_of_interest_start_i = int(chain_of_interest_idx_i[0]) + 1
        chain_of_interest_start_j = int(chain_of_interest_idx_j[0]) + 1


        
        
        ## assumes proteinMPNN 
        tied_pos_i = polymer_idx_i[alignment_idx_i] +  min(chain_of_interest_start_i) #one indexed start for the tied chains
        tied_pos_j = polymer_idx_j[alignment_idx_j] +  min(chain_of_interest_idx_j) """

        ## if it does, get the topology sequence length using alignment_idx mapp back to the chain name to get hte chain you want

        ####



        """ print("chain_of_interest_idx_i", chain_of_interest_idx_i)
        print("chain_of_interest_idx_j", chain_of_interest_idx_j)

        print("seq_idx_i before", ((sequence_idx_i[1])))
        print("polymer_idx_i before", (polymer_idx_i[1]))
        print(len(chain_of_interest_idx_i), len(chain_of_interest_idx_j))"""

        """   sequence_idx_i = sequence_idx_i[chain_of_interest_idx_i]
        sequence_idx_j = sequence_idx_j[chain_of_interest_idx_j]

        print("polymer_idx_i", polymer_idx_i)
        print("polymer_idx_j", polymer_idx_j)"""
       
    exit()
    os.makedirs(os.path.join(output_directory_2, id_pairs), exist_ok=True)

    trajs = []
    for id in id_pair:
        print("id", id)
        pdb_dir = os.path.join(args.unprocessed_struct_dir, id + ".pdb")
        trajs.extend(mdtraj.load(os.path.join(pdb_dir)))


        # add the amount to the coordinates of each trajectory
    # each traj list is for a single protein. one conformation == one pdb file
    num_residues_i = trajs[0].n_residues
    num_residues_j = trajs[1].n_residues




    print(id_pairs, num_residues_i, trajs[1].n_residues)
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
        "folder_with_pdbs=" + os.path.join("..", output_directory_2, id_pairs),
    )
    script_contents = script_contents.replace(
        "output_dir=", "output_dir=" + os.path.join("..", args.output_dir, id_pairs)
    )
    script_contents = script_contents.replace(
        "\ntied_positions=",
        '\ntied_positions="'
        + ", ".join(
            [" ".join(tied_pos_i), " ".join(tied_pos_j)]
        )
        + '"',
    )
    """script_contents = script_contents.replace(
        "\ntied_positions=",
        '\ntied_positions="'
        + ", ".join(
                [" ".join([str(i) for i in polymer_idx_i]), " ".join([str(i) for i in polymer_idx_j])]
        )
        + '"',
    )"""



    # write script contents
    with open(
        os.path.join(script_directory_2, f"proteinmpnn_job_{id_pairs}.sh"), "w"
    ) as file:
        file.write(script_contents)



    ## based on the ref seq used in processing alignment,
    # get the alignmnet indexs for each conformation, those indexs are your "tied positons" (the sturct used as ref will have all its postions in tied pos)

