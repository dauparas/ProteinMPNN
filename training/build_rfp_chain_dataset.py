# Generative Models for Graph-Based Protein Design by John Ingraham, Vikas Garg, Regina Barzilay and Tommi Jaakkola, NeurIPS 2019.
# Adapt from https://github.com/jingraham/neurips19-graph-protein-design


import os, time, gzip, json
from mmtf_util import *
from collections import defaultdict

MAX_LENGTH = 500

# CATH base URL
# cath_base_url = 'http://download.cathdb.info/cath/releases/all-releases/v4_2_0/'
# cath_base_url = 'http://download.cathdb.info/cath/releases/latest-release/'

# # CATH non-redundant set at 40% identity
# cath_nr40_fn = 'cath-dataset-nonredundant-S40.list'
# cath_nr40_url = cath_base_url + 'non-redundant-data-sets/' + cath_nr40_fn
# cath_nr40_file = 'cath/' + cath_nr40_fn
# download_cached(cath_nr40_url, cath_nr40_file)

# with open(cath_nr40_file) as f:
#     cath_nr40_ids = f.read().split('\n')[:-1]
# cath_nr40_chains = list(set(cath_id[:5] for cath_id in cath_nr40_ids))

dataset = open("./mmtf_dataset.list", "r").read().split("\n")[:-1]
chain_set = sorted([(name[:4], name[4].upper()) for name in dataset])

# # CATH hierarchical classification
# cath_domain_fn = 'cath-domain-list.txt'
# cath_domain_url = cath_base_url + 'cath-classification-data/' + cath_domain_fn
# cath_domain_file = 'cath/cath-domain-list.txt'
# download_cached(cath_domain_url, cath_domain_file)

# # CATH topologies
# cath_nodes = defaultdict(list)
# with open(cath_domain_file,'r') as f:
#     lines = [line.strip() for line in f if not line.startswith('#')]
#     for line in lines:
#         entries = line.split()
#         cath_id, cath_node = entries[0], '.'.join(entries[1:4])
#         chain_name = cath_id[:4] + '.' + cath_id[4]
#         cath_nodes[chain_name].append(cath_node)
# cath_nodes = {key:list(set(val)) for key,val in cath_nodes.items()}

# Build dataset
dataset = []
for chain_ix, (pdb, chain) in enumerate(chain_set):
    try:
        # Load and parse coordinates
        print(chain_ix, pdb, chain)
        start = time.time()
        chain_dict = mmtf_parse(pdb, chain)
        stop = time.time() - start

        if len(chain_dict["seq"]) <= MAX_LENGTH:
            chain_name = pdb + "." + chain
            chain_dict["name"] = chain_name
            chain_dict["CATH"] = cath_nodes[chain_name]
            print(pdb, chain, chain_dict["num_chains"], chain_dict["seq"])
            dataset.append(chain_dict)
        else:
            print("Too long")
    except Exception as e:
        print(e)

outfile = "chain_set.jsonl"
with open(outfile, "w") as f:
    for entry in dataset:
        f.write(json.dumps(entry) + "\n")
