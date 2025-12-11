import pickle
import random
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
import os
from rdkit.Chem import ChemicalFeatures, SanitizeFlags
from rdkit import Chem, RDConfig
import networkx as nx
import numpy as np
import copy
from collections import defaultdict
import sys

from datasets import utils
from diffusion.distributions import DistributionNodes
from utils import PlaceHolder
import torch.nn.functional as F
import itertools


# 药效团定义加载
fdef_path = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
factory = ChemicalFeatures.BuildFeatureFactory(fdef_path)

FAMILY_MAP = {
    'ZnBinder': 'NegIonizable'
}

def map_family(fam: str) -> str:
    return FAMILY_MAP.get(fam, fam)

def move_centroid_to_origin(mol: Chem.Mol):
    conf = mol.GetConformer()
    num_atoms = mol.GetNumAtoms()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(num_atoms)])
    centroid = coords.mean(axis=0)
    for i in range(num_atoms):
        pos = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, pos - centroid)
    return num_atoms, mol

def extract_pharmacophores_from_mol(mol: Chem.Mol) -> list:
    feats = factory.GetFeaturesForMol(mol)
    result = []
    for feat in feats:
        fam = map_family(feat.GetFamily())
        atom_ids = feat.GetAtomIds()
        pos = feat.GetPos()
        result.append({
            'family': fam,
            'atom_indices': atom_ids,
            'num_atoms': len(atom_ids),
            'pos': (pos.x, pos.y, pos.z)
        })
    return result

def break_bonds_between_sets(mol, atom_set_to_break):
    edit_mol = Chem.RWMol(mol)
    atom_set = set(atom_set_to_break)
    bonds_to_remove = []
    for bond in edit_mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (a1 in atom_set) != (a2 in atom_set):
            bonds_to_remove.append(bond.GetIdx())
    for bidx in sorted(bonds_to_remove, reverse=True):
        b = edit_mol.GetBondWithIdx(bidx)
        edit_mol.RemoveBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
    return edit_mol.GetMol()

def generate_linker_fragments(mol, used_atoms):
    used_atoms = set(used_atoms)
    mol_fragments = break_bonds_between_sets(mol, used_atoms)
    fragments = Chem.GetMolFrags(mol_fragments, asMols=False, sanitizeFrags=False)
    pos = np.array([mol.GetConformer().GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
    out = []
    for frag in fragments:
        atom_indices = list(frag)
        if all(idx in used_atoms for idx in atom_indices):
            continue
        centroid = np.mean(pos[atom_indices], axis=0).tolist()
        out.append(['Linker', atom_indices, len(atom_indices), centroid])
    return out

def weighted_pick(feat_list, weight_dict):
    weights = [weight_dict.get(f['family'], 0.01) for f in feat_list]
    return random.choices(feat_list, weights=weights, k=1)[0]

def weighted_sample_no_replacement(items, weights, k):
    if k > len(items):
        raise ValueError(f"Sample size k={k} is larger than population size {len(items)}")

    weights = np.array(weights, dtype=np.float64)
    total = weights.sum()

    if total == 0:
        raise ValueError("Sum of weights is zero, cannot normalize.")
    
    probs = weights / total

    if not np.isclose(probs.sum(), 1.0):
        raise ValueError(f"Probabilities do not sum to 1: {probs.sum()}")

    indices = np.random.choice(len(items), size=k, replace=False, p=probs)
    return [items[i] for i in indices]


def select_pharmacophore(mol, feats, pharmacophore_num, total_atoms, linker_strategy):
    unique = defaultdict(list)
    for f in feats:
        unique[f['pos']].append(f)

    distinct_weights = {
        'PosIonizable': 1.0, 'NegIonizable': 1.0, 'Donor': 0.8,
        'Acceptor': 0.7, 'Aromatic': 0.5, 'LumpedHydrophobe': 0.3, 'Hydrophobe': 0.1
    }
    chosen_weights = {
        'PosIonizable': 1.0, 'NegIonizable': 1.0, 'Donor': 0.9,
        'Acceptor': 0.9, 'Aromatic': 0.6, 'LumpedHydrophobe': 0.4, 'Hydrophobe': 0.2
    }

    distinct_feats = [weighted_pick(lst, distinct_weights) for lst in unique.values()]
    pharmacophore_num = min(len(distinct_feats), pharmacophore_num)
    feat_weights = [chosen_weights.get(f['family'], 0.01) for f in distinct_feats]
    chosen = weighted_sample_no_replacement(distinct_feats, feat_weights, pharmacophore_num)

    out, used_atoms = [], set()
    for f in chosen:
        out.append([f['family'], f['atom_indices'], f['num_atoms'], list(f['pos'])])
        used_atoms.update(f['atom_indices'])

    if linker_strategy == 'fragment':
        out.extend(generate_linker_fragments(mol, used_atoms))
    if linker_strategy == 'cluster':
        unused = sorted(set(range(total_atoms)) - used_atoms)
        out.append(['Linker', unused, len(unused), [0.0, 0.0, 0.0]])
    return out

def sample_probability(elment_array, plist, N):
    Psample = []
    index = int(random.random() * len(plist))
    beta, mw = 0.0, max(plist)
    for _ in range(N):
        beta += random.random() * 2.0 * mw
        while beta > plist[index]:
            beta -= plist[index]
            index = (index + 1) % len(plist)
        Psample.append(elment_array[index])
    return Psample

def keep_largest_fragment(mol):
    """
    返回最大连接片段和对应的 SMILES
    """
    if mol is None:
        return None, None

    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False) 
    if len(frags) == 1:
        Chem.SanitizeMol(mol)
        return mol, Chem.MolToSmiles(mol)

    # 找出最大片段（按非氢原子数）
    largest = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    Chem.SanitizeMol(largest)
    return largest, Chem.MolToSmiles(largest)

def mol_to_pharmacophore(mol, smiles, full_phar_encoder, pharmacophore_statistics=None, linker_strategy='fragment', remove_h=True):
    assert remove_h or linker_strategy == 'fragment'
    assert not remove_h or linker_strategy == 'cluster'

    atom_encoder = {'B': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'Al': 5, 'Si': 6,
                    'P': 7, 'S': 8, 'Cl': 9, 'As': 10, 'Br': 11, 'I': 12, 'Hg': 13, 'Bi': 14}
    if not remove_h:
        atom_encoder = {'H': 0, **{k: v + 1 for k, v in atom_encoder.items()}}
    atom_nums, mol = move_centroid_to_origin(mol)
    pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float)
    atom_types = torch.tensor([atom_encoder[a.GetSymbol()] for a in mol.GetAtoms()]).long()
    charges = torch.tensor([a.GetFormalCharge() for a in mol.GetAtoms()]).long()

    adj = torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True))
    edge_index = adj.nonzero().T
    bond_types = adj[edge_index[0], edge_index[1]]
    bond_types[bond_types == 1.5] = 4

    feats = extract_pharmacophores_from_mol(mol)
    num = sample_probability([3, 4, 5, 6, 7, 8], [0.08, 0.08, 0.36, 0.40, 0.05, 0.03], 1)[0]
    mol_noh = Chem.RemoveHs(mol) if not remove_h else mol
    pharm_list = select_pharmacophore(mol_noh, feats, num, mol.GetNumHeavyAtoms(), linker_strategy)

    if not remove_h:
        for feat in pharm_list:
            atom_ids = set(feat[1]) 
            for idx in feat[1]:
                atom = mol.GetAtomWithIdx(idx)
                atom_ids.update(n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1)
            feat[1] = sorted(atom_ids)
            feat[2] = len(feat[1])

    priority_map = {'PosIonizable': 6, 'NegIonizable': 6, 'Donor': 5, 'Acceptor': 4,
                    'Aromatic': 3, 'LumpedHydrophobe': 2, 'Hydrophobe': 1, 'Linker': 0}

    atom_phar_types = torch.zeros(atom_nums, dtype=torch.long)
    control_pos = torch.zeros((atom_nums, 3))
    atom_priority_score = torch.full((atom_nums,), -1)


    for feat in pharm_list:
        label, atom_indices, count, centroid = feat
        label_id, priority = full_phar_encoder[label], priority_map[label]
        if pharmacophore_statistics is not None and label in pharmacophore_statistics:
            pharmacophore_statistics[label][count] += 1
        for idx in atom_indices:
            if priority > atom_priority_score[idx]:
                atom_phar_types[idx] = label_id
                atom_priority_score[idx] = priority
                control_pos[idx] = torch.tensor(centroid)

    return Data(
        x=atom_types, edge_index=edge_index, edge_attr=bond_types.long(),
        pos=pos, charges=charges, smiles=smiles, cx=atom_phar_types,
        ccharges=torch.zeros_like(charges, dtype=torch.long),
        cedge_index=torch.empty((2, 0), dtype=torch.long),
        cedge_attr=torch.empty((0,), dtype=torch.long),
        cpos=control_pos, id=mol.GetProp('_Name')
    ), pharmacophore_statistics
    
def mol_to_torch_geometric(mol, atom_encoder, smiles):
    adj = torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True))
    edge_index = adj.nonzero().contiguous().T
    bond_types = adj[edge_index[0], edge_index[1]]
    bond_types[bond_types == 1.5] = 4
    edge_attr = bond_types.long()

    pos = torch.tensor(mol.GetConformers()[0].GetPositions()).float()
    pos = pos - torch.mean(pos, dim=0, keepdim=True)#将原子坐标拉到质心：原子坐标-质心坐标
    atom_types = []
    all_charges = []
    for atom in mol.GetAtoms():
        atom_types.append(atom_encoder[atom.GetSymbol()])
        all_charges.append(atom.GetFormalCharge())        # TODO: check if implicit Hs should be kept

    atom_types = torch.Tensor(atom_types).long()
    all_charges = torch.Tensor(all_charges).long()

    #control data
    # control_atom_types = (atom_types != 0).long()
    control_atom_types = torch.full_like(atom_types, atom_encoder['C'])
    control_charges = torch.zeros_like(all_charges)
    control_edge_attr = (edge_attr != 0).long()
    ##add noise
    # control_pos = pos+torch.normal(mean=0, std=0.01, size=(5,))

    data = Data(x=atom_types, edge_index=edge_index, edge_attr=edge_attr, pos=pos, charges=all_charges, smiles=smiles,
                cx=control_atom_types, ccharges=control_charges, cedge_index=edge_index, cedge_attr=control_edge_attr, cpos=pos, id=mol.GetProp('_Name')) 

    return data


def mol_to_control_data(geometric_data):
    """
    geometric_data: result of the mol_to_torch_geometric
    """

    data = geometric_data.clone()
    data.x = (data.x != 0).long() #atom type is changed to C
    data.charges = torch.zeros_like(data.charges) #charge is changed to 0

    return data


def remove_nonpolar_hydrogens(mol):
    """
    Remove non-polar hydrogen atoms from the molecule.
    Returns a new molecule with only polar hydrogens retained.
    """
    # Create an editable mol object
    edit_mol = Chem.RWMol(mol)
    
    # Identify polar hydrogens
    polar_h_indices = set()
    for atom in edit_mol.GetAtoms():
        if atom.GetAtomicNum() == 1:  # It's a hydrogen
            neighbor = atom.GetNeighbors()[0]  # Get the atom it's bonded to
            if neighbor.GetAtomicNum() in [7, 8, 16]:  # N, O, or S
                polar_h_indices.add(atom.GetIdx())
    
    # Remove non-polar hydrogens
    atoms_to_remove = []
    for atom in edit_mol.GetAtoms():
        if atom.GetAtomicNum() == 1 and atom.GetIdx() not in polar_h_indices:
            atoms_to_remove.append(atom.GetIdx())
    
    # Remove atoms in reverse order to avoid index issues
    for idx in sorted(atoms_to_remove, reverse=True):
        edit_mol.RemoveAtom(idx)
    
    # Convert back to a regular mol object and return
    mol = edit_mol.GetMol()
    try:
        Chem.SanitizeMol(mol, sanitizeOps=SanitizeFlags.SANITIZE_ALL)
    except Exception as e:
        print(f"Sanitization failed: {e}")
    
    return mol



def remove_hydrogens(data: Data):
    to_keep = data.x > 0
    new_edge_index, new_edge_attr = subgraph(to_keep, data.edge_index, data.edge_attr, relabel_nodes=True,
                                             num_nodes=len(to_keep))
    new_pos = data.pos[to_keep] - torch.mean(data.pos[to_keep], dim=0)
    
    new_cedge_index, new_cedge_attr = subgraph(to_keep, data.cedge_index, data.cedge_attr, relabel_nodes=True,
                                             num_nodes=len(to_keep))
    new_cedge_attr = (new_cedge_attr != 0).long()
    new_cpos = data.cpos[to_keep] - torch.mean(data.cpos[to_keep], dim=0)

    return Data(x=data.x[to_keep] - 1,
                edge_index=new_edge_index,# Shift onehot encoding to match atom decoder
                edge_attr=new_edge_attr,
                pos=new_pos,
                charges=data.charges[to_keep],
                smiles=data.smiles,
                cx=data.cx[to_keep],
                ccharges=data.ccharges[to_keep], 
                cedge_index=new_cedge_index,
                cedge_attr=new_cedge_attr,
                cpos = new_cpos,
                id=data.id
                )


def save_pickle(array, path):
    with open(path, 'wb') as f:
        pickle.dump(array, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


class Statistics:
    def __init__(self, num_nodes, atom_types, bond_types, charge_types, valencies, bond_lengths, bond_angles, phar_types=None):
        self.num_nodes = num_nodes
        print("NUM NODES IN STATISTICS", num_nodes)
        self.atom_types = atom_types
        self.phar_types = phar_types
        self.bond_types = bond_types
        self.charge_types = charge_types
        self.valencies = valencies
        self.bond_lengths = bond_lengths
        self.bond_angles = bond_angles


class MolInfos():
    def __init__(self, statistics_path, remove_h=True):
        self.atom_encoder = {'H': 0, 'B': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'Al': 6, 'Si': 7,
                     'P': 8, 'S': 9, 'Cl': 10, 'As': 11, 'Br': 12, 'I': 13, 'Hg': 14, 'Bi': 15}
        
        key = 'h'
        if remove_h:
            self.atom_encoder = {k: v - 1 for k, v in self.atom_encoder.items() if k != 'H'}
            key = 'noh'
        self.atom_decoder = [key for key in self.atom_encoder.keys()]
        self.num_atom_types = len(self.atom_decoder)

        statistics = Statistics(num_nodes=load_pickle(os.path.join(statistics_path, f'train_n_{key}.pickle')),
                                                atom_types=torch.from_numpy(np.load(os.path.join(statistics_path, f'train_atom_types_{key}.npy'))),
                                                bond_types=torch.from_numpy(np.load(os.path.join(statistics_path, f'train_bond_types_{key}.npy'))),
                                                charge_types=torch.from_numpy(np.load(os.path.join(statistics_path, f'train_charges_{key}.npy'))),
                                                valencies=load_pickle(os.path.join(statistics_path, f'train_valency_{key}.pickle')),
                                                bond_lengths=load_pickle(os.path.join(statistics_path, f'train_bond_lengths_{key}.pickle')),
                                                bond_angles=torch.from_numpy(np.load(os.path.join(statistics_path, f'train_angles_{key}.npy'))))
        
        train_n_nodes = load_pickle(os.path.join(statistics_path, f'train_n_{key}.pickle'))
        val_n_nodes = load_pickle(os.path.join(statistics_path, f'val_n_{key}.pickle'))
        test_n_nodes = load_pickle(os.path.join(statistics_path, f'test_n_{key}.pickle'))
        max_n_nodes = max(max(train_n_nodes.keys()), max(val_n_nodes.keys()), max(test_n_nodes.keys()))
        n_nodes = torch.zeros(max_n_nodes + 1, dtype=torch.long)
        for c in [train_n_nodes, val_n_nodes, test_n_nodes]:
            for key, value in c.items():
                n_nodes[key] += value


        self.statistics = statistics
        self.n_nodes = n_nodes / n_nodes.sum()
        self.atom_types = statistics.atom_types
        self.edge_types = statistics.bond_types
        self.charges_types = statistics.charge_types
        self.charges_marginals = (self.charges_types * self.atom_types[:, None]).sum(dim=0)
        self.valency_distribution = statistics.valencies
        self.max_n_nodes = len(n_nodes) - 1
        self.nodes_dist = DistributionNodes(n_nodes)

        self.input_dims = PlaceHolder(X=self.num_atom_types, charges=6, E=5, y=1, pos=3)
        self.output_dims = PlaceHolder(X=self.num_atom_types, charges=6, E=5, y=0, pos=3)
        self.collapse_charges = torch.Tensor([-2, -1, 0, 1, 2, 3]).int()

    def to_one_hot(self, X, charges, E, node_mask, X_sup = None, just_control=False):
        x = X.clone()
        X = F.one_hot(X, num_classes=self.num_atom_types).float()
        E = F.one_hot(E, num_classes=5).float()
        charges = F.one_hot(charges + 2, num_classes=6).float()
        placeholder = PlaceHolder(X=X, charges=charges, E=E,  y=None, pos=None)
        pl = placeholder.mask(node_mask, just_control)
        return pl.X, pl.charges, pl.E

    def one_hot_charges(self, charges):
        return F.one_hot((charges + 2).long(), num_classes=6).float()

def get_ring_systems(mol):
    ring_info = mol.GetRingInfo()
    ring_atom_indices = ring_info.AtomRings()
    num_rings = ring_info.NumRings()

    # 构建图，节点为环，边为两个环共享原子
    ring_graph = {i: set() for i in range(num_rings)}
    
    for i in range(num_rings):
        for j in range(i + 1, num_rings):
            if set(ring_atom_indices[i]).intersection(ring_atom_indices[j]):
                ring_graph[i].add(j)
                ring_graph[j].add(i)
    
    # 找到连通分量
    def dfs(node, visited):
        stack = [node]
        component = []
        while stack:
            n = stack.pop()
            if n not in visited:
                visited.add(n)
                component.append(n)
                stack.extend(ring_graph[n] - visited)
        return component

    visited = set()
    connected_components = []

    for node in range(num_rings):
        if node not in visited:
            component = dfs(node, visited)
            if len(component) > 1:  # 连通分量中有多个环
                connected_components.append(len(component))
    if connected_components:
        return max(connected_components)
    else:
        return connected_components

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
    rings_info = list(set([num for sublist in rings for num in sublist])) 

    del_list = ['LumpedHydrophobe', 'Aromatic']
    for del_key in del_list:
        if del_key in pharmacophore_dict:
            if del_key == 'Aromatic':
                del pharmacophore_dict[del_key]
            if del_key == 'LumpedHydrophobe':
                intersection = set(pharmacophore_dict[del_key]).intersection(rings_info)
                if intersection == set(pharmacophore_dict[del_key]):
                    del pharmacophore_dict[del_key]
                else:
                    remaining_list = list(set(pharmacophore_dict[del_key]) - intersection)
                    # pharmacophore_dict[del_key] = remaining_list
                    if 'Hydrophobe' in pharmacophore_dict:
                        pharmacophore_dict['Hydrophobe'].extend(remaining_list) 
                    else:
                        pharmacophore_dict['Hydrophobe'] = remaining_list
                    del pharmacophore_dict[del_key]
    pharmacophore_dict['Ring'] = rings
    return pharmacophore_dict

def deduplicate_phar_atoms(pharmacophore_dict):
    exist_atoms = [atom for block in pharmacophore_dict['Ring'] for atom in block]
    exist_atoms = list(set(exist_atoms))
    check_phar = ['PosIonizable', 'NegIonizable', 'Hydrophobe', 'Donor', 'Acceptor', 'ZnBinder', 'Linker']
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
    return filtered_pharmacophore_dict

def get_atom_adjacency(mol):
    num_atoms = mol.GetNumAtoms()
    atom_adjacency = np.zeros((4,num_atoms,num_atoms)) # 用于存储原子之间的连接关系
    for bond in mol.GetBonds():
        bond_s = bond.GetBeginAtom().GetIdx()
        bond_e = bond.GetEndAtom().GetIdx()
        bond_type = bond.GetBondType() 
        type_idx = [Chem.BondType.SINGLE,Chem.BondType.DOUBLE,Chem.BondType.TRIPLE,Chem.BondType.AROMATIC].index(bond_type)
        atom_adjacency[type_idx,bond_s,bond_e] = 1
        atom_adjacency[type_idx,bond_e,bond_s] = 1
    return atom_adjacency

def get_phar_in_adjacency(pharmacophore_dict, atom_adjacency):
    phar_in_adjacency = {}
    for phar_name in pharmacophore_dict:
        if phar_name != 'H_atom':
            phar_block_len = len(pharmacophore_dict[phar_name])
            phar_adjacency = np.zeros((phar_block_len,phar_block_len))
            if phar_name == 'Ring':
                for i,ring_i in enumerate(pharmacophore_dict[phar_name]):
                    for j,ring_j in enumerate(pharmacophore_dict[phar_name]):
                        for atom_i in ring_i:
                            for atom_j in ring_j:
                                if atom_i == atom_j or np.sum(atom_adjacency[1:,atom_i,atom_j]) == 1:
                                    phar_adjacency[i,j] = 1
                                    phar_adjacency[j,i] = 1                    
            else:
                for i,atom_i in enumerate(pharmacophore_dict[phar_name]):
                    for j,atom_j in enumerate(pharmacophore_dict[phar_name]):
                        if np.sum(atom_adjacency[ :,atom_i,atom_j]) == 1:
                            phar_adjacency[i,j] = 1
                            phar_adjacency[j,i] = 1

            phar_in_adjacency[phar_name] = phar_adjacency
    return phar_in_adjacency

def get_connected_graph(pharmacophore_dict, phar_in_adjacency):
    connected_graphs = {}
    for block in pharmacophore_dict:
        if block != 'H_atom':
            part_graph = []
            block_num = len(pharmacophore_dict[block])
            clique_adjacency = phar_in_adjacency[block]
            for i in range(block_num):
                for j in range(i + 1 ,block_num):
                    if clique_adjacency[i,j] == 1:
                        part_graph.append((i,j))
            part_graph = nx.Graph(part_graph)
            connected_graphs[block] = part_graph
    return connected_graphs

def get_connected_cliques(connected_graphs,pharmacophore_dict):
    all_new_cliques = {}
    for block in connected_graphs:
        new_cliques=[]
        connected_cliques=[]
        part_graph = connected_graphs[block]
        for subplot in nx.connected_components(part_graph):
            nodeSet = part_graph.subgraph(subplot).nodes()
            connected_cliques += nodeSet
            new_cliques.append([])
            for i in nodeSet:
                
                new_cliques[-1] += pharmacophore_dict[block][i] if isinstance(pharmacophore_dict[block][i], list) else [pharmacophore_dict[block][i]]
        
        for i in range(len(part_graph)):
            if i not in connected_cliques:
                new_cliques.append(pharmacophore_dict[block][i])
        new_cliques = [np.sort(list(set(clique))) if isinstance(clique, (list, np.ndarray)) else np.array([clique]) for clique in new_cliques]
        all_new_cliques[block] = new_cliques
    return all_new_cliques

def get_new_cliques(connected_cliques,pharmacophore_dict):
    new_cliques = {}
    total_cliques = []
    for block in connected_cliques:
        cliques=[]
        exist_list = []
        if connected_cliques[block]:
            cliques_items = [list(item) for item in connected_cliques[block]]
            exist_list += list([item for sublist in connected_cliques[block] for item in sublist])
            cliques += cliques_items
            total_cliques += cliques_items
        if block == 'Ring':
            for ring in pharmacophore_dict[block]:
                for atom in ring:
                    if atom not in exist_list:
                        cliques.append(ring)
                        total_cliques.append(ring)
                        exist_list += ring
        else:
            for atom in pharmacophore_dict[block]:
                if atom not in exist_list:
                    cliques.append([atom])
                    total_cliques.append([atom])
                    exist_list.append(atom)
        new_cliques[block] = cliques
    return new_cliques,total_cliques

# def get_phar_out_adjacency(total_cliques,atom_adjacency):
#     '''药效团之间连接关系'''
#     cliques_num = len(total_cliques)
#     phar_out_adjacency = np.zeros((cliques_num,cliques_num))
    
#     for i, clique_i in enumerate(total_cliques):
#         for j, clique_j in enumerate(total_cliques):
#             if any(np.sum(atom_adjacency[:,atom_i, atom_j]) == 1 or atom_i == atom_j for atom_i in clique_i for atom_j in clique_j):
#                 phar_out_adjacency[i, j] = 1
#                 phar_out_adjacency[j, i] = 1
#     return phar_out_adjacency

def get_phar_out_adjacency(total_cliques,atom_adjacency):
    '''药效团之间连接, 团和团之间连接的原子的连接关系'''
    phar_out_adjacency = np.zeros((atom_adjacency.shape[1], atom_adjacency.shape[1]))
    
    for i, clique_i in enumerate(total_cliques):
        for j, clique_j in enumerate(total_cliques):
            if clique_i != clique_j:
                for atom_j in clique_j:
                    for atom_i in clique_i:
                        if np.sum(atom_adjacency[:,atom_i, atom_j]) == 1:
                            phar_out_adjacency[atom_i, atom_j] = 1
                            phar_out_adjacency[atom_j, atom_i] = 1

    phar_out_adjacency = torch.from_numpy(phar_out_adjacency)
    return phar_out_adjacency


def get_phar_link(mol, pharmacophore_dict, H_atoms, pos, edge_index):
    atom_adjacency = get_atom_adjacency(mol)
    phar_in_adjacency = get_phar_in_adjacency(pharmacophore_dict, atom_adjacency)
    connected_graphs = get_connected_graph(pharmacophore_dict, phar_in_adjacency)
    connected_cliques = get_connected_cliques(connected_graphs,pharmacophore_dict)
    new_cliques,total_cliques = get_new_cliques(connected_cliques,pharmacophore_dict)
    phar_out_adjacency = get_phar_out_adjacency(total_cliques,atom_adjacency)

    phar_edge_index = phar_out_adjacency.nonzero().contiguous().T

    bond_types = phar_out_adjacency[phar_edge_index[0], phar_edge_index[1]]
    phar_edge_attr = bond_types.long()

    
    centroid_pos_list = {}
    for c in new_cliques:
        for sublist in new_cliques[c]:
            if len(sublist) == 1:
                centroid_pos_list[sublist[0]] = pos[sublist[0]]
            else:
                record_pos = pos[torch.tensor(sublist)]
                cen_pos = torch.mean(record_pos, dim=0, keepdim=True)
                for itx in sublist:
                    centroid_pos_list[itx] = cen_pos
    or_pos = pos.clone()
    for key, value in centroid_pos_list.items():
        or_pos[key] = value
    for h in H_atoms:
        try:
            or_pos[h] = centroid_pos_list[edge_index[1][torch.where(edge_index[0] == h)].tolist()[0]]
        except:
            print(h, atom_adjacency[:,h,:], '\n', edge_index, '\n', centroid_pos_list)
            if h in centroid_pos_list:
                continue
            else:
                arrow = edge_index[1][torch.where(edge_index[0] == h)].tolist()
                if len(arrow) != 0 :
                    or_pos[h] = centroid_pos_list[arrow[0]]
                else:
                    ce_tensors = {}
                    for key, value in centroid_pos_list.items():
                        while len(value.size()) != 1:
                            value = value.squeeze(0)
                        ce_tensors[key] = value
                    points_tensor = torch.stack(list(ce_tensors.values()))
                    distances = torch.norm(points_tensor.float() - pos[h].float(), dim=1)
                    nearest_point_index = torch.argmin(distances)
                    nearest_point_label = list(centroid_pos_list.keys())[nearest_point_index]
                    or_pos[h] = centroid_pos_list[nearest_point_label]

    return new_cliques, phar_edge_index, phar_edge_attr, or_pos

def find_atom_from_cliques(atom_idx, cliques):
    ex = False
    index_of_sublist = None
    for i, sublist in enumerate(cliques):
        if atom_idx in sublist:
            index_of_sublist = i
            ex = True
            break
    return ex, index_of_sublist

def merge_connected_cliques(cliques, connectivity_matrix):
    n = len(cliques)
    
    # 构建图并找到连通分量
    def find_connected_components(matrix):
        visited = [False] * n
        components = []

        def dfs(node, component):
            stack = [node]
            while stack:
                v = stack.pop()
                if not visited[v]:
                    visited[v] = True
                    component.append(v)
                    for neighbor in range(n):
                        if matrix[v, neighbor] == 1 and not visited[neighbor]:
                            stack.append(neighbor)

        for i in range(n):
            if not visited[i]:
                component = []
                dfs(i, component)
                components.append(component)

        return components

    connected_components = find_connected_components(connectivity_matrix)

    # 合并连通分量中的基团
    new_cliques = []
    for component in connected_components:
        new_clique = []
        for index in component:
            new_clique.extend(cliques[index])
        new_cliques.append(new_clique)

    return new_cliques

def get_phar_link_after_conversion(mol, pharmacophore_dict, H_atoms, pos, edge_index, conversion_ratio = 0.4):
    atom_adjacency = get_atom_adjacency(mol)
    phar_in_adjacency = get_phar_in_adjacency(pharmacophore_dict, atom_adjacency)
    connected_graphs = get_connected_graph(pharmacophore_dict, phar_in_adjacency)
    connected_cliques = get_connected_cliques(connected_graphs,pharmacophore_dict)
    new_cliques,total_cliques = get_new_cliques(connected_cliques,pharmacophore_dict)
    phar_out_adjacency = get_phar_out_adjacency(total_cliques,atom_adjacency)


    old_linker = new_cliques['Linker'] if 'Linker' in new_cliques else []
    phar_cliques = [clique for clique in total_cliques if clique not in old_linker]

    new_linker = []
    random.seed(42)
    num_linkers = round(conversion_ratio * (len(phar_cliques)))
    linker_indices = random.sample(range(len(phar_cliques)), num_linkers)
    new_linker = [clique for i, clique in enumerate(phar_cliques) if i in linker_indices]

    total_linker = old_linker + new_linker

    linker_adj = np.zeros((len(total_linker),len(total_linker)))
    for clique in new_linker:
        for idx in clique:
            idx_to = phar_out_adjacency[idx].nonzero().contiguous().T
            if idx_to.numel() != 0:
                for atom in idx_to.tolist()[0]:
                    
                    ex,index = find_atom_from_cliques(atom, total_linker)
                    if ex:
                        index_to = total_linker.index(clique)
                        linker_adj[index, index_to] = 1
                        linker_adj[index_to, index] = 1

                        phar_out_adjacency[idx, atom] = 0
                        phar_out_adjacency[atom, idx] = 0
                        
    new_total_linker = merge_connected_cliques(total_linker, linker_adj)
    if 'Linker' in new_cliques:
        for clique_name in new_cliques:
            if clique_name == 'Linker':
                new_cliques[clique_name] = new_total_linker
            else:
                new_cliques[clique_name] = [sublist for sublist in new_cliques[clique_name] if sublist not in new_linker]
    else:
        for clique_name in new_cliques:
            new_cliques[clique_name] = [sublist for sublist in new_cliques[clique_name] if sublist not in new_linker]
        new_cliques['Linker'] = new_total_linker
    new_cliques = {key: value for key, value in new_cliques.items() if value}
                     
    phar_edge_index = phar_out_adjacency.nonzero().contiguous().T
    bond_types = phar_out_adjacency[phar_edge_index[0], phar_edge_index[1]]
    phar_edge_attr = bond_types.long()

    centroid_pos_list = {}
    for c in new_cliques:
        for sublist in new_cliques[c]:
            if len(sublist) == 1:
                centroid_pos_list[sublist[0]] = pos[sublist[0]]
            else:
                record_pos = pos[torch.tensor(sublist)]
                cen_pos = torch.mean(record_pos, dim=0, keepdim=True)
                for itx in sublist:
                    centroid_pos_list[itx] = cen_pos
    or_pos = pos.clone()
    for key, value in centroid_pos_list.items():
        or_pos[key] = value
    for h in H_atoms:
        or_pos[h] = centroid_pos_list[edge_index[1][torch.where(edge_index[0] == h)].tolist()[0]]

    return new_cliques, phar_edge_index, phar_edge_attr, or_pos

def reconstruct_data(data, slices, idx) -> Data:
    """
    从批处理的 `data` 和 `slices` 中恢复原始数据对象。

    参数:
    - data: 批处理后的 Data 对象。
    - slices: 包含切片信息的字典，用于指示每个数据的起始和终止位置。
    - idx: 要恢复的数据对象的索引。

    返回值:
    - 原始的 Data 对象。
    """
    reconstructed_data = Data()
    for key in slices.keys():
        start, end = slices[key][idx].item(), slices[key][idx + 1].item()
        if key != 'edge_index' and key != 'cedge_index':
            reconstructed_data[key] = data[key][start:end]
        else:
            reconstructed_data[key] = data[key][:, start:end]
    
    return reconstructed_data

def reconstruct_data_list(data, slices):
    """
    从批处理的 `data` 和 `slices` 中恢复原始的 `data_list`。

    参数:
    - data: 批处理后的 Data 对象。
    - slices: 包含切片信息的字典，用于指示每个数据的起始和终止位置。

    返回值:
    - 原始的 data_list，其中每个元素都是一个 Data 对象。
    """
    # 计算原始数据的数量
    num_data = len(next(iter(slices.values()))) - 1  # slices 的每个列表比数据多一个长度
    data_list = []

    # 遍历所有数据索引，逐个还原
    from tqdm import trange
    print('start restore Data')
    for idx in trange(num_data):
        data_item = reconstruct_data(data, slices, idx)  # 使用你的函数逐一还原数据
        data_list.append(data_item)  # 将还原后的数据加入列表

    return data_list


