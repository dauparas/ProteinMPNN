import argparse

def main(args):

    import numpy as np
    import os, time, gzip, json
    import glob 
    
    folder_with_pdbs_path = args.input_path
    save_path = args.output_path
    ca_only = args.ca_only
    
    alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
    states = len(alpha_1)
    alpha_3 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
               'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','GAP']
    
    aa_1_N = {a:n for n,a in enumerate(alpha_1)}
    aa_3_N = {a:n for n,a in enumerate(alpha_3)}
    aa_N_1 = {n:a for n,a in enumerate(alpha_1)}
    aa_1_3 = {a:b for a,b in zip(alpha_1,alpha_3)}
    aa_3_1 = {b:a for a,b in zip(alpha_1,alpha_3)}
    
    def AA_to_N(x):
      # ["ARND"] -> [[0,1,2,3]]
      x = np.array(x);
      if x.ndim == 0: x = x[None]
      return [[aa_1_N.get(a, states-1) for a in y] for y in x]
    
    def N_to_AA(x):
      # [[0,1,2,3]] -> ["ARND"]
      x = np.array(x);
      if x.ndim == 1: x = x[None]
      return ["".join([aa_N_1.get(a,"-") for a in y]) for y in x]
    
    
    def parse_PDB_biounits(x, atoms=['N','CA','C'], chain=None):
      '''
      input:  x = PDB filename
              atoms = atoms to extract (optional)
      output: (length, atoms, coords=(x,y,z)), sequence
      '''
      xyz,seq,min_resn,max_resn = {},{},1e6,-1e6
      for line in open(x,"rb"):
        line = line.decode("utf-8","ignore").rstrip()
    
        if line[:6] == "HETATM" and line[17:17+3] == "MSE":
          line = line.replace("HETATM","ATOM  ")
          line = line.replace("MSE","MET")
    
        if line[:4] == "ATOM":
          ch = line[21:22]
          if ch == chain or chain is None:
            atom = line[12:12+4].strip()
            resi = line[17:17+3]
            resn = line[22:22+5].strip()
            x,y,z = [float(line[i:(i+8)]) for i in [30,38,46]]
    
            if resn[-1].isalpha(): 
                resa,resn = resn[-1],int(resn[:-1])-1
            else: 
                resa,resn = "",int(resn)-1
    #         resn = int(resn)
            if resn < min_resn: 
                min_resn = resn
            if resn > max_resn: 
                max_resn = resn
            if resn not in xyz: 
                xyz[resn] = {}
            if resa not in xyz[resn]: 
                xyz[resn][resa] = {}
            if resn not in seq: 
                seq[resn] = {}
            if resa not in seq[resn]: 
                seq[resn][resa] = resi
    
            if atom not in xyz[resn][resa]:
              xyz[resn][resa][atom] = np.array([x,y,z])
    
      # convert to numpy arrays, fill in missing values
      seq_,xyz_ = [],[]
      try:
          for resn in range(min_resn,max_resn+1):
            if resn in seq:
              for k in sorted(seq[resn]): seq_.append(aa_3_N.get(seq[resn][k],20))
            else: seq_.append(20)
            if resn in xyz:
              for k in sorted(xyz[resn]):
                for atom in atoms:
                  if atom in xyz[resn][k]: xyz_.append(xyz[resn][k][atom])
                  else: xyz_.append(np.full(3,np.nan))
            else:
              for atom in atoms: xyz_.append(np.full(3,np.nan))
          return np.array(xyz_).reshape(-1,len(atoms),3), N_to_AA(np.array(seq_))
      except TypeError:
          return 'no_chain', 'no_chain'
    
    
    
    pdb_dict_list = []
    c = 0
    
    if folder_with_pdbs_path[-1]!='/':
        folder_with_pdbs_path = folder_with_pdbs_path+'/'
    
    
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V','W','X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet
    
    biounit_names = glob.glob(folder_with_pdbs_path+'*.pdb')
    for biounit in biounit_names:
        my_dict = {}
        s = 0
        concat_seq = ''
        concat_N = []
        concat_CA = []
        concat_C = []
        concat_O = []
        concat_mask = []
        coords_dict = {}
        for letter in chain_alphabet:
            if ca_only:
                sidechain_atoms = ['CA']
            else:
                sidechain_atoms = ['N', 'CA', 'C', 'O']
            xyz, seq = parse_PDB_biounits(biounit, atoms=sidechain_atoms, chain=letter)
            if type(xyz) != str:
                concat_seq += seq[0]
                my_dict['seq_chain_'+letter]=seq[0]
                coords_dict_chain = {}
                if ca_only:
                    coords_dict_chain['CA_chain_'+letter]=xyz.tolist()
                else:
                    coords_dict_chain['N_chain_' + letter] = xyz[:, 0, :].tolist()
                    coords_dict_chain['CA_chain_' + letter] = xyz[:, 1, :].tolist()
                    coords_dict_chain['C_chain_' + letter] = xyz[:, 2, :].tolist()
                    coords_dict_chain['O_chain_' + letter] = xyz[:, 3, :].tolist()
                my_dict['coords_chain_'+letter]=coords_dict_chain
                s += 1
        fi = biounit.rfind("/")
        my_dict['name']=biounit[(fi+1):-4]
        my_dict['num_of_chains'] = s
        my_dict['seq'] = concat_seq
        if s < len(chain_alphabet):
            pdb_dict_list.append(my_dict)
            c+=1
            
            
    with open(save_path, 'w') as f:
        for entry in pdb_dict_list:
            f.write(json.dumps(entry) + '\n')
           

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--input_path", type=str, help="Path to a folder with pdb files, e.g. /home/my_pdbs/")
    argparser.add_argument("--output_path", type=str, help="Path where to save .jsonl dictionary of parsed pdbs")
    argparser.add_argument("--ca_only", action="store_true", default=False, help="parse a backbone-only structure (default: false)")

    args = argparser.parse_args()
    main(args)
