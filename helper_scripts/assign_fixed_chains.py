import argparse

def main(args):
    import json

    with open(args.input_path, 'r') as json_file:
        json_list = list(json_file)
    
    global_designed_chain_list = []
    if args.chain_list != '':
        global_designed_chain_list = [str(item) for item in args.chain_list.split()]
    my_dict = {}
    for json_str in json_list:
        result = json.loads(json_str)
        all_chain_list = [item[-1:] for item in list(result) if item[:9]=='seq_chain'] #['A','B', 'C',...]
        if len(global_designed_chain_list) > 0:
            designed_chain_list = global_designed_chain_list
        else:
            #manually specify, e.g.
            designed_chain_list = ["A"]
        fixed_chain_list = [letter for letter in all_chain_list if letter not in designed_chain_list] #fix/do not redesign these chains 
        my_dict[result['name']]= (designed_chain_list, fixed_chain_list)
    
    with open(args.output_path, 'w') as f:
        f.write(json.dumps(my_dict) + '\n')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--input_path", type=str, help="Path to the parsed PDBs")
    argparser.add_argument("--output_path", type=str, help="Path to the output dictionary")
    argparser.add_argument("--chain_list", type=str, default='', help="List of the chains that need to be designed")    

    args = argparser.parse_args()
    main(args) 

# Output looks like this:
# {"5TTA": [["A"], ["B"]], "3LIS": [["A"], ["B"]]}

