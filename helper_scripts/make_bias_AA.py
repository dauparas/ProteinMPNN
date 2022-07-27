import argparse 

def main(args):

    import numpy as np
    import json

    bias_list = [float(item) for item in args.bias_list.split()]
    AA_list = [str(item) for item in args.AA_list.split()]

    my_dict = dict(zip(AA_list, bias_list))

    with open(args.output_path, 'w') as f:
        f.write(json.dumps(my_dict) + '\n')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--output_path", type=str, help="Path to the output dictionary")
    argparser.add_argument("--AA_list", type=str, default='', help="List of AAs to be biased")
    argparser.add_argument("--bias_list", type=str, default='', help="AA bias strengths")

    args = argparser.parse_args()
    main(args)

#e.g. output
#{"A": -0.01, "G": 0.02}
