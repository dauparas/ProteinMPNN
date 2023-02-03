import argparse

def main(args):
    import glob
    import random
    import numpy as np
    import json
    import itertools
    
    with open(args.input_path, 'r') as json_file:
        json_list = list(json_file)
    
    fixed_list = [[int(item) for item in one.split()] for one in args.position_list.split(",")]
    global_designed_chain_list = [str(item) for item in args.chain_list.split()]
    my_dict = {}
    
    if not args.specify_non_fixed:
        for json_str in json_list:
            result = json.loads(json_str)
            all_chain_list = [item[-1:] for item in list(result) if item[:9]=='seq_chain']
            fixed_position_dict = {}
            for i, chain in enumerate(global_designed_chain_list):
                fixed_position_dict[chain] = fixed_list[i]
            for chain in all_chain_list:
                if chain not in global_designed_chain_list:       
                    fixed_position_dict[chain] = []
            my_dict[result['name']] = fixed_position_dict
    else:
        for json_str in json_list:
            result = json.loads(json_str)
            all_chain_list = [item[-1:] for item in list(result) if item[:9]=='seq_chain']
            fixed_position_dict = {}   
            for chain in all_chain_list:
                seq_length = len(result[f'seq_chain_{chain}'])
                all_residue_list = (np.arange(seq_length)+1).tolist()
                if chain not in global_designed_chain_list:
                    fixed_position_dict[chain] = all_residue_list
                else:
                    idx = np.argwhere(np.array(global_designed_chain_list) == chain)[0][0]
                    fixed_position_dict[chain] = list(set(all_residue_list)-set(fixed_list[idx]))
            my_dict[result['name']] = fixed_position_dict

    with open(args.output_path, 'w') as f:
        f.write(json.dumps(my_dict) + '\n')
    
    #e.g. output
    #{"5TTA": {"A": [1, 2, 3, 7, 8, 9, 22, 25, 33], "B": []}, "3LIS": {"A": [], "B": []}}

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--input_path", type=str, help="Path to the parsed PDBs")
    argparser.add_argument("--output_path", type=str, help="Path to the output dictionary")
    argparser.add_argument("--chain_list", type=str, default='', help="List of the chains that need to be fixed")
    argparser.add_argument("--position_list", type=str, default='', help="Position lists, e.g. 11 12 14 18, 1 2 3 4 for first chain and the second chain")
    argparser.add_argument("--specify_non_fixed", action="store_true", default=False, help="Allows specifying just residues that need to be designed (default: false)")

    args = argparser.parse_args()
    main(args)

