'''
from Bio.PDB import parse_pdb_header

with open("1XAE.pdb","r") as handle:
    header_dict = parse_pdb_header(handle)

print(header_dict)

#first_model = structure[0]
'''
'''
from Bio.PDB.PDBParser import PDBParser
p = PDBParser(PERMISSIVE=1)
structure_id = "1XAE"
filename = "1XAE.pdb"
structure = p.get_structure(structure_id, filename)

resolution = structure.header['resolution']
keywords = structure.header['keywords']
'''

from Bio.PDB import Select, PDBIO
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import parse_pdb_header
class ChainSelect(Select):
    def __init__(self, chain):
        self.chain = chain

    def accept_chain(self, chain):
        if chain.get_id() == self.chain:
            return 1
        else:          
            return 0

#chains = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
p = PDBParser(PERMISSIVE=1)       
filename = '2GX2.pdb'
structure = p.get_structure(filename, filename)
with open("2GX2.pdb","r") as handle:
    header_dict = parse_pdb_header(handle)

tmp = header_dict['compound']
tmpp = tmp['1']
kl = []
#print(tmpp['chain'])

for tmp in tmpp['chain']:
    if(tmp<'a' or tmp>'z'):
        continue
    kl.append(tmp.upper())
#print(kl)
for chain in kl:
    pdb_chain_file ='{}_{}.pdb'.format(filename,chain)                       
    io_w_no_h = PDBIO()               
    io_w_no_h.set_structure(structure)
    io_w_no_h.save('{}'.format(pdb_chain_file), ChainSelect(chain))