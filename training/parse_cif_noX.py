import pdbx
from pdbx.reader.PdbxReader import PdbxReader
from pdbx.reader.PdbxContainers import DataCategory
import gzip
import numpy as np
import torch
import os,sys
import glob
import re
from scipy.spatial import KDTree
from itertools import combinations,permutations
import tempfile
import subprocess

RES_NAMES = [
    'ALA','ARG','ASN','ASP','CYS',
    'GLN','GLU','GLY','HIS','ILE',
    'LEU','LYS','MET','PHE','PRO',
    'SER','THR','TRP','TYR','VAL'
]

RES_NAMES_1 = 'ARNDCQEGHILKMFPSTWYV'

to1letter = {aaa:a for a,aaa in zip(RES_NAMES_1,RES_NAMES)}
to3letter = {a:aaa for a,aaa in zip(RES_NAMES_1,RES_NAMES)}

ATOM_NAMES = [
    ("N", "CA", "C", "O", "CB"), # ala
    ("N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"), # arg
    ("N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"), # asn
    ("N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"), # asp
    ("N", "CA", "C", "O", "CB", "SG"), # cys
    ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"), # gln
    ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"), # glu
    ("N", "CA", "C", "O"), # gly
    ("N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"), # his
    ("N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"), # ile
    ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"), # leu
    ("N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"), # lys
    ("N", "CA", "C", "O", "CB", "CG", "SD", "CE"), # met
    ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"), # phe
    ("N", "CA", "C", "O", "CB", "CG", "CD"), # pro
    ("N", "CA", "C", "O", "CB", "OG"), # ser
    ("N", "CA", "C", "O", "CB", "OG1", "CG2"), # thr
    ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE2", "CE3", "NE1", "CZ2", "CZ3", "CH2"), # trp
    ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"), # tyr
    ("N", "CA", "C", "O", "CB", "CG1", "CG2") # val
]
        
idx2ra = {(RES_NAMES_1[i],j):(RES_NAMES[i],a) for i in range(20) for j,a in enumerate(ATOM_NAMES[i])}

aa2idx = {(r,a):i for r,atoms in zip(RES_NAMES,ATOM_NAMES) 
          for i,a in enumerate(atoms)}
aa2idx.update({(r,'OXT'):3 for r in RES_NAMES})


def writepdb(f, xyz, seq, bfac=None):

    #f = open(filename,"w")
    f.seek(0)
    
    ctr = 1
    seq = str(seq)
    L = len(seq)
    
    if bfac is None:
        bfac = np.zeros((L))

    idx = []
    for i in range(L):
        for j,xyz_ij in enumerate(xyz[i]):
            key = (seq[i],j)
            if key not in idx2ra.keys():
                continue
            if np.isnan(xyz_ij).sum()>0:
                continue
            r,a = idx2ra[key]
            f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                    "ATOM", ctr, a, r, 
                    "A", i+1, xyz_ij[0], xyz_ij[1], xyz_ij[2],
                    1.0, bfac[i,j] ) )
            if a == 'CA':
                idx.append(i)
            ctr += 1
            
    #f.close()
    f.flush()
    
    return np.array(idx)


def TMalign(chainA, chainB):
    
    # temp files to save the two input protein chains 
    # and TMalign transformation
    fA = tempfile.NamedTemporaryFile(mode='w+t', dir='/dev/shm')
    fB = tempfile.NamedTemporaryFile(mode='w+t', dir='/dev/shm')
    mtx = tempfile.NamedTemporaryFile(mode='w+t', dir='/dev/shm')

    # create temp PDB files keep track of residue indices which were saved
    idxA = writepdb(fA, chainA['xyz'], chainA['seq'], bfac=chainA['bfac'])
    idxB = writepdb(fB, chainB['xyz'], chainB['seq'], bfac=chainB['bfac'])
    
    # run TMalign
    tm = subprocess.Popen('/home/aivan/prog/TMalign %s %s -m %s'%(fA.name, fB.name, mtx.name), 
                          shell=True, 
                          stdout=subprocess.PIPE, 
                          stderr=subprocess.PIPE, 
                          encoding='utf-8')
    stdout,stderr = tm.communicate()
    lines = stdout.split('\n')
    
    # if TMalign failed
    if len(stderr) > 0:
        return None,None

    # parse transformation
    mtx.seek(0)
    tu = np.fromstring(''.join(l[2:] for l in mtx.readlines()[2:5]), 
                       dtype=float, sep=' ').reshape((3,4))
    t = tu[:,0]
    u = tu[:,1:]
    
    # parse rmsd, sequence identity, and two TM-scores 
    rmsd = float(lines[16].split()[4][:-1])
    seqid = float(lines[16].split()[-1])
    tm1 = float(lines[17].split()[1])
    tm2 = float(lines[18].split()[1])

    # parse alignment
    seq1 = lines[-5]
    seq2 = lines[-3]

    ss1 = np.array(list(seq1.strip()))!='-'
    ss2 = np.array(list(seq2.strip()))!='-'
    #print(ss1)
    #print(ss2)
    mask = np.logical_and(ss1, ss2)

    alnAB = np.stack((idxA[(np.cumsum(ss1)-1)[mask]],
                      idxB[(np.cumsum(ss2)-1)[mask]]))

    alnBA = np.stack((alnAB[1],alnAB[0]))

    # clean up
    fA.close()
    fB.close()
    mtx.close()
    
    resAB = {'rmsd':rmsd, 'seqid':seqid, 'tm':tm1, 'aln':alnAB, 't':t, 'u':u}
    resBA = {'rmsd':rmsd, 'seqid':seqid, 'tm':tm2, 'aln':alnBA, 't':-u.T@t, 'u':u.T}
    
    return resAB,resBA


def get_tm_pairs(chains):
    """run TM-align for all pairs of chains"""

    tm_pairs = {}
    for A,B in combinations(chains.keys(),r=2):
        resAB,resBA = TMalign(chains[A],chains[B])
        #if resAB is None:
        #    continue
        tm_pairs.update({(A,B):resAB})
        tm_pairs.update({(B,A):resBA})
        
    # add self-alignments
    for A in chains.keys():
        L = chains[A]['xyz'].shape[0]
        aln = np.arange(L)[chains[A]['mask'][:,1]]
        aln = np.stack((aln,aln))
        tm_pairs.update({(A,A):{'rmsd':0.0, 'seqid':1.0, 'tm':1.0, 'aln':aln}})
        
    return tm_pairs
        


def parseOperationExpression(expression) :

    expression = expression.strip('() ')
    operations = []
    for e in expression.split(','):
        e = e.strip()
        pos = e.find('-')
        if pos>0:
            start = int(e[0:pos])
            stop = int(e[pos+1:])
            operations.extend([str(i) for i in range(start,stop+1)])
        else:
            operations.append(e)
            
    return operations


def parseAssemblies(data,chids):

    xforms =  {'asmb_chains'  : None, 
               'asmb_details' : None, 
               'asmb_method'  : None,
               'asmb_ids'     : None}

    assembly_data = data.getObj("pdbx_struct_assembly")
    assembly_gen = data.getObj("pdbx_struct_assembly_gen")
    oper_list = data.getObj("pdbx_struct_oper_list")

    if (assembly_data is None) or (assembly_gen is None) or (oper_list is None):
        return xforms

    # save all basic transformations in a dictionary
    opers = {}
    for k in range(oper_list.getRowCount()):
        key = oper_list.getValue("id", k)
        val = np.eye(4)
        for i in range(3):
            val[i,3] = float(oper_list.getValue("vector[%d]"%(i+1), k))
            for j in range(3):
                val[i,j] = float(oper_list.getValue("matrix[%d][%d]"%(i+1,j+1), k))
        opers.update({key:val})
    
    
    chains,details,method,ids = [],[],[],[]

    for index in range(assembly_gen.getRowCount()):
        
        # Retrieve the assembly_id attribute value for this assembly
        assemblyId = assembly_gen.getValue("assembly_id", index)
        ids.append(assemblyId)

        # Retrieve the operation expression for this assembly from the oper_expression attribute	
        oper_expression = assembly_gen.getValue("oper_expression", index)

        oper_list = [parseOperationExpression(expression) 
                     for expression in re.split('\(|\)', oper_expression) if expression]
        
        # chain IDs which the transform should be applied to
        chains.append(assembly_gen.getValue("asym_id_list", index))

        index_asmb = min(index,assembly_data.getRowCount()-1)
        details.append(assembly_data.getValue("details", index_asmb))
        method.append(assembly_data.getValue("method_details", index_asmb))
    
        # 
        if len(oper_list)==1:
            xform = np.stack([opers[o] for o in oper_list[0]])
        elif len(oper_list)==2:
            xform = np.stack([opers[o1]@opers[o2] 
                              for o1 in oper_list[0] 
                              for o2 in oper_list[1]])

        else:
            print('Error in processing assembly')           
            return xforms
        
        xforms.update({'asmb_xform%d'%(index):xform})
    
    xforms['asmb_chains'] = chains
    xforms['asmb_details'] = details
    xforms['asmb_method'] = method
    xforms['asmb_ids'] = ids

    return xforms


def parse_mmcif(filename):

    #print(filename)
    
    chains = {}   # 'chain_id' -> chain_strucure

    # read a gzipped .cif file
    data = []
    with gzip.open(filename,'rt') as cif:
        reader = PdbxReader(cif)
        reader.read(data)
    data = data[0]

    #
    # get sequences
    #
    
    # map chain entity to chain ID
    entity_poly = data.getObj('entity_poly')
    if entity_poly is None:
        return {},{}

    pdbx_poly_seq_scheme = data.getObj('pdbx_poly_seq_scheme')
    pdb2asym = dict({
        (r[pdbx_poly_seq_scheme.getIndex('pdb_strand_id')],
         r[pdbx_poly_seq_scheme.getIndex('asym_id')]) 
        for r in data.getObj('pdbx_poly_seq_scheme').getRowList()
    })

    chs2num = {pdb2asym[ch]:r[entity_poly.getIndex('entity_id')] 
               for r in entity_poly.getRowList() 
               for ch in r[entity_poly.getIndex('pdbx_strand_id')].split(',')
               if r[entity_poly.getIndex('type')]=='polypeptide(L)'}

    # get canonical sequences for polypeptide chains
    num2seq = {r[entity_poly.getIndex('entity_id')]:r[entity_poly.getIndex('pdbx_seq_one_letter_code_can')].replace('\n','') 
               for r in entity_poly.getRowList() 
               if r[entity_poly.getIndex('type')]=='polypeptide(L)'}
    
    # map chain entity to amino acid sequence 
    #entity_poly_seq = data.getObj('entity_poly_seq')
    #num2seq = dict.fromkeys(set(chs2num.values()), "")
    #for row in entity_poly_seq.getRowList():
    #    num = row[entity_poly_seq.getIndex('entity_id')]
    #    res = row[entity_poly_seq.getIndex('mon_id')]
    #    if num not in num2seq.keys():
    #        continue
    #    num2seq[num] += (to1letter[res] if res in to1letter.keys() else 'X')
    
    # modified residues
    pdbx_struct_mod_residue = data.getObj('pdbx_struct_mod_residue')
    if pdbx_struct_mod_residue is None:
        modres = {}
    else:
        modres = dict({(r[pdbx_struct_mod_residue.getIndex('label_comp_id')],
                        r[pdbx_struct_mod_residue.getIndex('parent_comp_id')])
                       for r in pdbx_struct_mod_residue.getRowList()})
        for k,v in modres.items():
            print("# non-standard residue: %s %s"%(k,v))

    # initialize dict of chains
    for c,n in chs2num.items():
        seq = num2seq[n]
        L = len(seq)
        chains.update({c : {'seq'  : seq,
                            'xyz'  : np.full((L,14,3),np.nan,dtype=np.float32),
                            'mask' : np.zeros((L,14),dtype=bool),
                            'bfac' : np.full((L,14),np.nan,dtype=np.float32),
                            'occ'  : np.zeros((L,14),dtype=np.float32) }})


    #
    # populate structures
    #

    # get indices of fields of interest
    atom_site = data.getObj('atom_site')
    i = {k:atom_site.getIndex(val) for k,val in [('atm', 'label_atom_id'), # atom name
                                                 ('atype', 'type_symbol'), # atom chemical type
                                                 ('res', 'label_comp_id'), # residue name (3-letter)
                                                 #('chid', 'auth_asym_id'), # chain ID
                                                 ('chid', 'label_asym_id'), # chain ID
                                                 ('num', 'label_seq_id'), # sequence number
                                                 ('alt', 'label_alt_id'), # alternative location ID
                                                 ('x', 'Cartn_x'), # xyz coords
                                                 ('y', 'Cartn_y'),
                                                 ('z', 'Cartn_z'),
                                                 ('occ', 'occupancy'), # occupancy
                                                 ('bfac', 'B_iso_or_equiv'), # B-factors 
                                                 ('model', 'pdbx_PDB_model_num') # model number (for multi-model PDBs, e.g. NMR)
                                                ]}
    
    for a in atom_site.getRowList():
        
        # skip HETATM
        #if a[0] != 'ATOM':
        #    continue

        # skip hydrogens
        if a[i['atype']] == 'H':
            continue
        
        # skip if not a polypeptide
        if a[i['chid']] not in chains.keys():
            continue
        
        # parse atom
        atm, res, chid, num, alt, x, y, z, occ, Bfac, model = \
                (t(a[i[k]]) for k,t in (('atm',str), ('res',str), ('chid',str), 
                ('num',int), ('alt',str),
                ('x',float), ('y',float), ('z',float), 
                ('occ',float), ('bfac',float), ('model',int)))


        #print(atm, res, chid, num, alt, x, y, z, occ, Bfac, model)
        c = chains[chid]

        # remap residue to canonical
        a = c['seq'][num-1]
        if a in to3letter.keys():
            res = to3letter[a]
        else:
            if res in modres.keys() and modres[res] in to1letter.keys():
                res = modres[res]
                c['seq'] = c['seq'][:num-1] + to1letter[res] + c['seq'][num:]
            else:
                res = 'GLY'
            
        # skip if not a standard residue/atom
        if (res,atm) not in aa2idx.keys():
            continue

        # skip everything except model #1
        if model > 1:
            continue

        # populate chians using max occup atoms
        idx = (num-1, aa2idx[(res,atm)])
        if occ > c['occ'][idx]:
            c['xyz'][idx] = [x,y,z]
            c['mask'][idx] = True
            c['occ'][idx] = occ
            c['bfac'][idx] = Bfac

    # 
    # metadata
    #
    #if data.getObj('reflns') is not None:
    #    res = data.getObj('reflns').getValue('d_resolution_high',0)
    res = None
    if data.getObj('refine') is not None:
        try:
            res = float(data.getObj('refine').getValue('ls_d_res_high',0))
        except:
            res = None
        
    if (data.getObj('em_3d_reconstruction') is not None) and (res is None):
        try:
            res = float(data.getObj('em_3d_reconstruction').getValue('resolution',0))
        except:
            res = None
    
    chids = list(chains.keys())
    seq = []
    for ch in chids:
        mask = chains[ch]['mask'][:,:3].sum(1)==3
        ref_seq = chains[ch]['seq']
        atom_seq = ''.join([a if m else '-' for a,m in zip(ref_seq,mask)])
        seq.append([ref_seq,atom_seq])

    metadata = {
        'method'     : data.getObj('exptl').getValue('method',0).replace(' ','_'),
        'date'       : data.getObj('pdbx_database_status').getValue('recvd_initial_deposition_date',0),
        'resolution' : res,
        'chains'     : chids,
        'seq'        : seq,
        'id'         : data.getObj('entry').getValue('id',0)
    }
    

    #
    # assemblies
    #

    asmbs = parseAssemblies(data,chains)
    metadata.update(asmbs)

    return chains, metadata


IN = sys.argv[1]
OUT = sys.argv[2]

chains,metadata = parse_mmcif(IN)
ID = metadata['id']

tm_pairs = get_tm_pairs(chains)
if 'chains' in metadata.keys() and len(metadata['chains'])>0:
    chids = metadata['chains']
    tm = []
    for a in chids:
        tm_a = []
        for b in chids:
            tm_ab = tm_pairs[(a,b)]
            if tm_ab is None:
                tm_a.append([0.0,0.0,999.9])
            else:
                tm_a.append([tm_ab[k] for k in ['tm','seqid','rmsd']])
        tm.append(tm_a)
    metadata.update({'tm':tm})

for k,v in chains.items():
    nres = (v['mask'][:,:3].sum(1)==3).sum()
    print(">%s_%s %s %s %s %d %d\n%s"%(ID,k,metadata['date'],metadata['method'],
                                       metadata['resolution'],len(v['seq']),nres,v['seq']))
    
    torch.save({kc:torch.Tensor(vc) if kc!='seq' else str(vc)
                for kc,vc in v.items()}, f"{OUT}_{k}.pt")

meta_pt = {}
for k,v in metadata.items():
    if "asmb_xform" in k or k=="tm":
        v = torch.Tensor(v)
    meta_pt.update({k:v})
torch.save(meta_pt, f"{OUT}.pt")
