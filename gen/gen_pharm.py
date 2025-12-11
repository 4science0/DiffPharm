import random
import torch
from torch_geometric.data import Data
import os
from rdkit.Chem import ChemicalFeatures
from rdkit import Chem, RDConfig
import numpy as np
from collections import defaultdict
import json
import argparse
import ast

# 药效团定义加载
# fdef_path = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
fdef_path = 'DiffPharm/gen/BaseFeatures.fdef'
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


def select_pharmacophore(mol, feats, pharmacophore_num, total_atoms, linker_strategy, chosen_pharm_list=None):
    
    if chosen_pharm_list is None:
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
    else:
        chosen = [feats[i] for i in chosen_pharm_list]


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

        
def save_all_features(feats: list, total_atoms:int, out_json: str) -> None:
    """
    将所有药效团特征保存为 JSON（不限制位置）并添加 Linker 特征
    """
    out_list = []
    for i,f in enumerate(feats):
        out_list.append([i,
            [
            f['family'],
            f['num_atoms'],
            list(f['pos'])
        ]])

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    write_json_lines(out_list, out_json, total_atoms)
    print(f"已保存所有特征（含 Linker）到 {out_json}")

def write_json_lines(data: list, out_path: str, final_int: int | None = None, mode='all'):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('[\n')
        for i,item in enumerate(data):
            if mode != 'all':
                item = [x for i, x in enumerate(item) if i != 1]
            line = json.dumps(item, ensure_ascii=False)
            f.write(f"  {line},\n")

        # 如果指定了 final_int，写入最后一行（不带逗号）
        if final_int is not None:
            f.write(f"  {json.dumps(final_int, ensure_ascii=False)}\n")
        else:
            # 移除最后一项的逗号
            f.seek(f.tell() - 2)  # 回退3个字符：`,\n`
            f.write('\n')

        f.write(']\n')
        
def mol_to_pharmacophore(mol, smiles, pharm_num=None,  chosen_pharm_list=None, linker_strategy='fragment', remove_h=True, output_path=None):
    assert remove_h or linker_strategy == 'fragment'
    assert not remove_h or linker_strategy == 'cluster'
    
    atom_encoder = {'B': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'Al': 5, 'Si': 6,
                    'P': 7, 'S': 8, 'Cl': 9, 'As': 10, 'Br': 11, 'I': 12, 'Hg': 13, 'Bi': 14}
    phar_encoder = {'Linker': 0, 'Donor': 1, 'Acceptor': 2, 'NegIonizable': 3, 'PosIonizable': 4, 'Aromatic': 5, 'Hydrophobe': 6, 'LumpedHydrophobe':7}
    h = 'h' if not remove_h else 'noh'
    suffix = f'{linker_strategy}-L-{h}'
    
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
    save_all_features(feats, atom_nums, os.path.join(output_path, f'all_pharmacophores_{h}.json'))


    pharm_num = sample_probability([3, 4, 5, 6, 7, 8], [0.08, 0.08, 0.36, 0.40, 0.05, 0.03], 1)[0] if pharm_num is None else pharm_num
    mol_noh = Chem.RemoveHs(mol) if not remove_h else mol
    pharm_list = select_pharmacophore(mol_noh, feats, pharm_num, mol.GetNumHeavyAtoms(), linker_strategy, chosen_pharm_list)

    if not remove_h:
        for feat in pharm_list:
            atom_ids = set(feat[1]) 
            for idx in feat[1]:
                atom = mol.GetAtomWithIdx(idx)
                atom_ids.update(n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1)
            feat[1] = sorted(atom_ids)
            feat[2] = len(feat[1])

    write_json_lines(pharm_list, os.path.join(output_path, f'pharm_feature_{suffix}.json'), mode='part')
    priority_map = {'PosIonizable': 6, 'NegIonizable': 6, 'Donor': 5, 'Acceptor': 4,
                    'Aromatic': 3, 'LumpedHydrophobe': 2, 'Hydrophobe': 1, 'Linker': 0}

    atom_phar_types = torch.zeros(atom_nums, dtype=torch.long)
    control_pos = torch.zeros((atom_nums, 3))
    atom_priority_score = torch.full((atom_nums,), -1)


    for feat in pharm_list:
        label, atom_indices, count, centroid = feat
        label_id, priority = phar_encoder[label], priority_map[label]
        for idx in atom_indices:
            if priority > atom_priority_score[idx]:
                atom_phar_types[idx] = label_id
                atom_priority_score[idx] = priority
                control_pos[idx] = torch.tensor(centroid)

    data=Data(
        x=atom_types, edge_index=edge_index, edge_attr=bond_types.long(),
        pos=pos, charges=charges, smiles=smiles, cx=atom_phar_types,
        ccharges=torch.zeros_like(charges, dtype=torch.long),
        cedge_index=torch.empty((2, 0), dtype=torch.long),
        cedge_attr=torch.empty((0,), dtype=torch.long),
        cpos=control_pos, id=mol.GetProp('_Name'),
        idx=0
    )
    torch.save([data], os.path.join(output_path, f'template_{suffix}.pt'))

def main():
    parser = argparse.ArgumentParser(description='Generate pharmacophore features for molecules in an SDF file')
    parser.add_argument('--sdf', required=True, help='Input SDF file with molecules')
    parser.add_argument('--pharm_num', type=int, default=None, help='Number of pharmacophores to select')
    parser.add_argument('--chosen', type=ast.literal_eval, default=None,
                    help='List of pharmacophore indices, e.g. "[1, 3, 5]"')
    parser.add_argument('--linker_strategy', choices=['fragment', 'cluster'], default='fragment',
                        help='Linker strategy to use')
    parser.add_argument('--remove_h', action='store_true', default=False,
                    help='Remove hydrogens before processing (default: False)')
    parser.add_argument('--output_path', default='Molecular pharmacophore results', help='Output path')
    args = parser.parse_args()
    suppl = Chem.SDMolSupplier(args.sdf, removeHs=args.remove_h)
    
    for i,mol in enumerate(suppl): 
        if mol is None:
            continue
        mol_name = mol.GetProp('_Name') if mol.HasProp('_Name') else f'Mol_{i}'
        args.output_path = os.path.join(args.output_path, mol_name)
        mol_to_pharmacophore(mol,
                             Chem.MolToSmiles(mol), 
                             pharm_num=args.pharm_num,
                             chosen_pharm_list=args.chosen,
                             linker_strategy=args.linker_strategy,
                             remove_h=args.remove_h,
                             output_path=args.output_path)

        print(f'Results of molecule {i} were written to {args.output_path}')

if __name__ == '__main__':
    main()