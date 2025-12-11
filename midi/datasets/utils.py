
from rdkit import Chem
# from rdkit.Chem import AllChem,Draw
import torch
from torch_geometric.data import Data
import os
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import copy
import sys
from datasets import dataset_utils
import random



def get_atom_phar_idx(mol):
    fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    feats = factory.GetFeaturesForMol(mol)

    pharmacophore_dict = {}
    pharmacophore_atoms = []
    atoms_label_dict = {}

    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        if atom.GetAtomicNum() != 1:
            atoms_label_dict[atom_idx] = ['Linker']
        else:
            atoms_label_dict[atom_idx] = ['H_atom']

    for f in feats:
        pharmacophore_name = f.GetFamily()

        for i in list(f.GetAtomIds()):
            if atoms_label_dict[i] == ['Linker']:
                atoms_label_dict[i] = [pharmacophore_name]
            else:
                if pharmacophore_name not in atoms_label_dict[i]:
                    atoms_label_dict[i] += [pharmacophore_name]

        if pharmacophore_name not in pharmacophore_dict:
            pharmacophore_dict[pharmacophore_name] = list(f.GetAtomIds())
        else:
            pharmacophore_dict[pharmacophore_name] += list(f.GetAtomIds())
        pharmacophore_atoms += list(f.GetAtomIds())
    pharmacophore_atoms = list(set(pharmacophore_atoms))
    linker_atoms,H_atoms = [],[]
    for i in atoms_label_dict:
        if i not in pharmacophore_atoms:
            if atoms_label_dict[i] != ['H_atom']:
                linker_atoms.append(i)
            else:
                H_atoms.append(i)
    if len(linker_atoms) != 0:
        pharmacophore_dict['Linker'] = linker_atoms
    pharmacophore_dict['H_atom'] = H_atoms
        
    return atoms_label_dict, pharmacophore_dict, H_atoms

def separate_phar(mol, pharmacophore_dict):
    sssr = Chem.GetSymmSSSR(mol)
    aromatic_rings,other_rings = [],[]

    for ring in sssr:
        if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
            aromatic_rings.append(list(ring))
        else:
            other_rings.append(list(ring))
    rings = aromatic_rings + other_rings
    # rings_info = list(set([num for sublist in rings for num in sublist])) 
    if 'Aromatic' in pharmacophore_dict:
        del pharmacophore_dict['Aromatic']
    if rings:
        pharmacophore_dict['Ring'] = rings
    return pharmacophore_dict

def deduplicate_phar_atoms(atoms_label_dict, pharmacophore_dict):
    exist_atoms = [atom for block in pharmacophore_dict['Ring'] for atom in block] if 'Ring' in pharmacophore_dict else []
    exist_atoms = list(set(exist_atoms))
    for i in exist_atoms:
        v = copy.deepcopy(atoms_label_dict[i])
        if 'Aromatic' in atoms_label_dict[i]:
            v[v.index('Aromatic')] = 'Ring'
        else:
            v.append('Ring')
        atoms_label_dict[i] = v

    check_phar = ['PosIonizable', 'NegIonizable', 'LumpedHydrophobe', 'Hydrophobe', 'Donor', 'Acceptor', 'ZnBinder', 'Linker']
    for check_block in check_phar:
        if check_block in pharmacophore_dict:
            l = copy.deepcopy(pharmacophore_dict[check_block])
            for atom in pharmacophore_dict[check_block]:
                if atom in exist_atoms:
                    l.remove(atom)
                else:
                    exist_atoms.append(atom)
            pharmacophore_dict[check_block] = l
    filtered_pharmacophore_dict = {key: value for key, value in pharmacophore_dict.items() if value}
    return atoms_label_dict, filtered_pharmacophore_dict


def mol_to_phar_data(mol, atom_phar_encoder, rprint = False):
    '''一个group里的atom的phar label随即选一个
        Hide atom and bond type, bond保留药效团(连接原子)之间的, 药效团内部的全部mask,
       conversion_ratio > 0 :  随机把一部分药效团mask掉(标成linker), 抹掉这部分药效团本来内部的连接
       conversion_ratio = 0 或 None 不进行mask操作
    '''
    adj = torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True))
    edge_index = adj.nonzero().contiguous().T
    

    pos = torch.tensor(mol.GetConformers()[0].GetPositions()).float()
    pos = pos - torch.mean(pos, dim=0, keepdim=True)#将原子坐标拉到质心：原子坐标-质心坐标
    
    
    if len(atom_phar_encoder) == 10:
        atom_phar_types = [atom_phar_encoder['H_atom']] * mol.GetNumAtoms()
        atoms_label_dict, pharmacophore_dict, H_atoms = get_atom_phar_idx(mol)
        pharmacophore_dict = separate_phar(mol, pharmacophore_dict)
        atoms_label_dict, pharmacophore_dict = deduplicate_phar_atoms(atoms_label_dict, pharmacophore_dict)
        control_atoms_label_dict = copy.deepcopy(atoms_label_dict)
        


        new_cliques, control_edge_index, control_edge_attr, control_pos = dataset_utils.get_phar_link(mol, pharmacophore_dict, H_atoms, pos, edge_index)

        for clique_name in new_cliques:
            for clique in new_cliques[clique_name]:
                p = set()
                for idx in clique:
                    p.update(atom_phar_encoder[label] for label in control_atoms_label_dict[idx])
                p = list(p)
                p = p if len(p) > 1 else p[0]
                for idx in clique:
                    atom_phar_types[idx] = p


    elif len(atom_phar_encoder) == 9:
        atom_phar_types = []
        atoms_label_dict, pharmacophore_dict, H_atoms = dataset_utils.get_atom_phar_idx(mol)
        pharmacophore_dict = dataset_utils.separate_phar(mol, pharmacophore_dict)
        pharmacophore_dict = dataset_utils.deduplicate_phar_atoms(pharmacophore_dict)
        atom_single_phar = {}
        for phar in pharmacophore_dict:
            for atom_idx in pharmacophore_dict[phar]:
                if isinstance(atom_idx, int):
                    if atom_idx in atom_single_phar and atom_single_phar[atom_idx] == phar:
                        continue
                    else:
                        atom_single_phar[atom_idx] = phar
                elif isinstance(atom_idx, list):
                    for i in atom_idx:
                        if i in atom_single_phar and atom_single_phar[i] == phar:
                            continue
                        else:
                            atom_single_phar[i] = phar
                else:
                    print('There are some elements of unknown type in atom_idx:{}-{}'.format(atom_idx,type(atom_idx)))
        for atom in mol.GetAtoms():
            atom_phar_types.append(atom_phar_encoder[atom_single_phar[atom.GetIdx()]])
        new_cliques, control_edge_index, control_edge_attr, control_pos = dataset_utils.get_phar_link(mol, pharmacophore_dict, H_atoms, pos, edge_index)

    control_pos = control_pos - torch.mean(control_pos, dim = 0)
        
    if not rprint:
        return atom_phar_types, control_pos
    else:
        return atoms_label_dict, atom_phar_types, control_pos