import pandas as pd
import numpy as np

import glob
import random
import numpy as np
import json


def softmax(x, T):
    return np.exp(x/T)/np.sum(np.exp(x/T), -1, keepdims=True)

def parse_pssm(path):
    data = pd.read_csv(path, skiprows=2)
    floats_list_list = []
    for i in range(data.values.shape[0]):
        str1 = data.values[i][0][4:]
        floats_list = []
        for item in str1.split():
            floats_list.append(float(item))
        floats_list_list.append(floats_list)
    np_lines = np.array(floats_list_list)
    return np_lines

np_lines = parse_pssm('/home/swang523/RLcage/capsid/monomersfordesign/8-16-21/pssm_rainity_final_8-16-21_int/build_0.2089_0.98_0.4653_19_2.00_0.005745.pssm')

mpnn_alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
input_alphabet = 'ARNDCQEGHILKMFPSTWYV'

permutation_matrix = np.zeros([20,21])
for i in range(20):
    letter1 = input_alphabet[i]
    for j in range(21):
        letter2 = mpnn_alphabet[j]
        if letter1 == letter2:
            permutation_matrix[i,j]=1.

pssm_log_odds = np_lines[:,:20] @ permutation_matrix
pssm_probs = np_lines[:,20:40] @ permutation_matrix

X_mask = np.concatenate([np.zeros([1,20]), np.ones([1,1])], -1)

def softmax(x, T):
    return np.exp(x/T)/np.sum(np.exp(x/T), -1, keepdims=True)

#Load parsed PDBs:  
with open('/home/justas/projects/cages/parsed/test.jsonl', 'r') as json_file:
    json_list = list(json_file)

my_dict = {}
for json_str in json_list:
    result = json.loads(json_str)
    all_chain_list = [item[-1:] for item in list(result) if item[:9]=='seq_chain']
    pssm_dict = {}
    for chain in all_chain_list:
        pssm_dict[chain] = {}
        pssm_dict[chain]['pssm_coef'] = (np.ones(len(result['seq_chain_A']))).tolist() #a number between 0.0 and 1.0 specifying how much attention put to PSSM, can be adjusted later as a flag
        pssm_dict[chain]['pssm_bias'] = (softmax(pssm_log_odds-X_mask*1e8, 1.0)).tolist() #PSSM like, [length, 21] such that sum over the last dimension adds up to 1.0
        pssm_dict[chain]['pssm_log_odds'] = (pssm_log_odds).tolist()
    my_dict[result['name']] = pssm_dict

#Write output to:    
with open('/home/justas/projects/lab_github/mpnn/data/pssm_dict.jsonl', 'w') as f:
    f.write(json.dumps(my_dict) + '\n')
