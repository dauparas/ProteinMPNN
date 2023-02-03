import argparse

def main(args):

    import glob
    import random
    import numpy as np
    import json
    import itertools
    
    with open(args.input_path, 'r') as json_file:
        json_list = list(json_file)
    
    homooligomeric_state = args.homooligomer

    if homooligomeric_state == 0:
        tied_list = [[int(item) for item in one.split()] for one in args.position_list.split(",")]
        global_designed_chain_list = [str(item) for item in args.chain_list.split()]
        my_dict = {}
        for json_str in json_list:
            result = json.loads(json_str)
            all_chain_list = sorted([item[-1:] for item in list(result) if item[:9]=='seq_chain']) #A, B, C, ...
            tied_positions_list = []
            for i, pos in enumerate(tied_list[0]):
                temp_dict = {}
                for j, chain in enumerate(global_designed_chain_list):
                    temp_dict[chain] = [tied_list[j][i]] #needs to be a list
                tied_positions_list.append(temp_dict)
            my_dict[result['name']] = tied_positions_list
    else:
        if args.pos_neg_chain_list:
            chain_list_input = [[str(item) for item in one.split()] for one in args.pos_neg_chain_list.split(",")]
            chain_betas_input = [[float(item) for item in one.split()] for one in args.pos_neg_chain_betas.split(",")]
            chain_list_flat = [item for sublist in chain_list_input for item in sublist]
            chain_betas_flat = [item for sublist in chain_betas_input for item in sublist]
            chain_betas_dict = dict(zip(chain_list_flat, chain_betas_flat))
        my_dict = {}
        for json_str in json_list:
            result = json.loads(json_str)
            all_chain_list = sorted([item[-1:] for item in list(result) if item[:9]=='seq_chain']) #A, B, C, ...
            tied_positions_list = []
            chain_length = len(result[f"seq_chain_{all_chain_list[0]}"])
            for chains in chain_list_input:
                for i in range(1,chain_length+1):
                    temp_dict = {}
                    for j, chain in enumerate(chains):
                        if args.pos_neg_chain_list and chain in chain_list_flat:
                            temp_dict[chain] = [[i], [chain_betas_dict[chain]]]
                        else: 
                            temp_dict[chain] = [[i], [1.0]] #first list is for residue numbers, second list is for weights for the energy, +ive and -ive design
                    tied_positions_list.append(temp_dict)
            my_dict[result['name']] = tied_positions_list
 
    with open(args.output_path, 'w') as f:
        f.write(json.dumps(my_dict) + '\n')

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--input_path", type=str, help="Path to the parsed PDBs")
    argparser.add_argument("--output_path", type=str, help="Path to the output dictionary")
    argparser.add_argument("--chain_list", type=str, default='', help="List of the chains that need to be fixed")
    argparser.add_argument("--position_list", type=str, default='', help="Position lists, e.g. 11 12 14 18, 1 2 3 4 for first chain and the second chain")
    argparser.add_argument("--homooligomer", type=int, default=0, help="If 0 do not use, if 1 then design homooligomer")
    argparser.add_argument("--pos_neg_chain_list", type=str, default='', help="Chain lists to be tied together")
    argparser.add_argument("--pos_neg_chain_betas", type=str, default='', help="Chain beta list for the chain lists provided; 1.0 for the positive design, -0.1 or -0.5 for negative, 0.0 means do not use that chain info")

    args = argparser.parse_args()
    main(args)


#e.g. output
#{"5TTA": [], "3LIS": [{"A": [1], "B": [1]}, {"A": [2], "B": [2]}, {"A": [3], "B": [3]}, {"A": [4], "B": [4]}, {"A": [5], "B": [5]}, {"A": [6], "B": [6]}, {"A": [7], "B": [7]}, {"A": [8], "B": [8]}, {"A": [9], "B": [9]}, {"A": [10], "B": [10]}, {"A": [11], "B": [11]}, {"A": [12], "B": [12]}, {"A": [13], "B": [13]}, {"A": [14], "B": [14]}, {"A": [15], "B": [15]}, {"A": [16], "B": [16]}, {"A": [17], "B": [17]}, {"A": [18], "B": [18]}, {"A": [19], "B": [19]}, {"A": [20], "B": [20]}, {"A": [21], "B": [21]}, {"A": [22], "B": [22]}, {"A": [23], "B": [23]}, {"A": [24], "B": [24]}, {"A": [25], "B": [25]}, {"A": [26], "B": [26]}, {"A": [27], "B": [27]}, {"A": [28], "B": [28]}, {"A": [29], "B": [29]}, {"A": [30], "B": [30]}, {"A": [31], "B": [31]}, {"A": [32], "B": [32]}, {"A": [33], "B": [33]}, {"A": [34], "B": [34]}, {"A": [35], "B": [35]}, {"A": [36], "B": [36]}, {"A": [37], "B": [37]}, {"A": [38], "B": [38]}, {"A": [39], "B": [39]}, {"A": [40], "B": [40]}, {"A": [41], "B": [41]}, {"A": [42], "B": [42]}, {"A": [43], "B": [43]}, {"A": [44], "B": [44]}, {"A": [45], "B": [45]}, {"A": [46], "B": [46]}, {"A": [47], "B": [47]}, {"A": [48], "B": [48]}, {"A": [49], "B": [49]}, {"A": [50], "B": [50]}, {"A": [51], "B": [51]}, {"A": [52], "B": [52]}, {"A": [53], "B": [53]}, {"A": [54], "B": [54]}, {"A": [55], "B": [55]}, {"A": [56], "B": [56]}, {"A": [57], "B": [57]}, {"A": [58], "B": [58]}, {"A": [59], "B": [59]}, {"A": [60], "B": [60]}, {"A": [61], "B": [61]}, {"A": [62], "B": [62]}, {"A": [63], "B": [63]}, {"A": [64], "B": [64]}, {"A": [65], "B": [65]}, {"A": [66], "B": [66]}, {"A": [67], "B": [67]}, {"A": [68], "B": [68]}, {"A": [69], "B": [69]}, {"A": [70], "B": [70]}, {"A": [71], "B": [71]}, {"A": [72], "B": [72]}, {"A": [73], "B": [73]}, {"A": [74], "B": [74]}, {"A": [75], "B": [75]}, {"A": [76], "B": [76]}, {"A": [77], "B": [77]}, {"A": [78], "B": [78]}, {"A": [79], "B": [79]}, {"A": [80], "B": [80]}, {"A": [81], "B": [81]}, {"A": [82], "B": [82]}, {"A": [83], "B": [83]}, {"A": [84], "B": [84]}, {"A": [85], "B": [85]}, {"A": [86], "B": [86]}, {"A": [87], "B": [87]}, {"A": [88], "B": [88]}, {"A": [89], "B": [89]}, {"A": [90], "B": [90]}, {"A": [91], "B": [91]}, {"A": [92], "B": [92]}, {"A": [93], "B": [93]}, {"A": [94], "B": [94]}, {"A": [95], "B": [95]}, {"A": [96], "B": [96]}]}

