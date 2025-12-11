import torch
from rdkit import Chem
from torch_geometric.data.batch import Batch
from torch_geometric.loader import DataLoader
import itertools
from rdkit.Chem import QED
import utils as utils
import json
from diffusion_model import FullDenoisingDiffusion
import hydra
import os
import omegaconf
from collections import Counter
from torch_geometric.data import Data
from datasets.dataset_utils import MolInfos
from collections import defaultdict
import numpy as np

def sample_linker(mean, mode = 'normal'): 
    if mode == 'normal':
        std = 4
        lower_bound = max(0, mean - 15)
        upper_bound = mean + 15
        samples = np.random.normal(loc=mean, scale=std, size=1)
        samples = np.clip(samples, lower_bound, upper_bound)
        int_samples = np.round(samples).astype(int)
        return int_samples[0]
    if mode == 'skewness':
        left_tail = np.arange(max(0, mean-15), mean)
        left_prob = np.exp(-0.5 * (mean - left_tail))  # 指数衰减
        left_prob /= left_prob.sum()  # 归一化

    
        right_range_1 = np.arange(mean, mean + 5)  # mean到mean+10之间的均匀分布
        right_range_2 = np.arange(mean + 5, mean + 20)  # 大于mean+10的右偏部分


        right_prob_1 = np.ones_like(right_range_1, dtype=float)  # 均匀分布
        right_prob_1 /= right_prob_1.sum()  # 归一化

        right_prob_2 = np.exp(-0.1 * (right_range_2 - mean - 10))
        right_prob_2 /= right_prob_2.sum()  # 归一化

        values = np.concatenate([left_tail, right_range_1, right_range_2])
        probabilities = np.concatenate([left_prob, right_prob_1, right_prob_2])
        probabilities /= probabilities.sum()  # 归一化所有的概率

        samples = np.random.choice(values, size=1, p=probabilities)
        return samples[0]
        
        
        
phar_encoder = {'Linker': 0, 'Donor': 1, 'Acceptor': 2, 'NegIonizable': 3, 'PosIonizable': 4, 'Aromatic': 5, 'Hydrophobe': 6, 'LumpedHydrophobe':7}
def get_control_data(pharmacophore_info, id=None, samples_to_generate=1, based_on_template='copy'):
    '''pharmacophore_info: [(pharmacophore label, pharmacophore pos)]'''
    assert samples_to_generate == 1

    list1 = []
    phar_list,pos_list = [],[]
    with open('DiffPharm/statistics_info/train_pharmacophore_statistics_noh.json', 'r')as f1:
        num_dict = json.load(f1)

    for phar_type,nums,phar_pos in pharmacophore_info:     
        if isinstance(phar_pos, torch.Tensor) and phar_pos.size() == torch.Size([3]):
            pos_list.append(phar_pos)
        elif isinstance(phar_pos, list) and len(phar_pos) == 3:
            pos_list.append(torch.tensor(phar_pos))
        else:
            raise ValueError('The type of location is incorrect')
        
        phar_label = phar_encoder[phar_type]
        phar_list.append(phar_label)

        if based_on_template == 'copy':
            sample_num = nums
        else:
            # first sample
            if phar_type in num_dict.keys():
                data = num_dict[phar_type]
                keys = torch.tensor(list(map(int, data.keys())))
                values = list(data.values())

                weights = torch.tensor(values)
                probabilities = weights / weights.sum()

                distribution = torch.distributions.Categorical(probabilities)

                sample_index = distribution.sample()
                sample_num = keys[sample_index].item()
            else:
                sample_num = sample_linker(mode=based_on_template, mean=nums)
        
        list1.append(sample_num)


    atom_nums = sum(list1)
    control_atom_phar_types = torch.zeros(atom_nums).long()
    control_atom_sup_phar_types = torch.zeros(atom_nums).long()
    control_pos = torch.zeros(atom_nums, 3)
    control_charges = torch.zeros(atom_nums).long()
    interval = 0
    for i,j in enumerate(list1):
        control_atom_phar_types[interval: interval + list1[i]] = phar_list[i]
        # if len(sup_phar_list) != 0:
        #     control_atom_sup_phar_types[interval: interval + list1[i]] = sup_phar_list[i]
        control_pos[interval: interval + j] = pos_list[i]
        interval += j

    # control_pos = control_pos - torch.mean(control_pos, dim = 0)

    all_connections = list(itertools.combinations(range(atom_nums), 2))
    all_connections += [(j, i) for i, j in all_connections]
    control_edge_index = torch.tensor(all_connections).T
    control_edge_attr = torch.zeros(int(len(control_edge_index[0]))).long()


    return Data(x=control_atom_phar_types, cx=control_atom_phar_types, cx_sup=control_atom_sup_phar_types, ccharges=control_charges,\
            cedge_index = control_edge_index, cedge_attr=control_edge_attr, cpos=control_pos, idx=id)

def write_sdf_file(out_path, samples, filter_substructure=None):

    error_message = Counter()
    all_valid_mols = defaultdict(list)
    # all_valid_mols = list()
    if filter_substructure:
        filter_smarts = [Chem.MolFromSmarts(subst) for subst in filter_substructure if Chem.MolFromSmarts(subst)]
    for mol in samples:
        rdmol = mol.rdkit_mol
        if rdmol is not None:

            try:
                mol_frags = Chem.rdmolops.GetMolFrags(rdmol, asMols=True, sanitizeFrags=False)
                largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                Chem.SanitizeMol(largest_mol)
                if largest_mol is not None:
                    try:
                        smiles = Chem.MolToSmiles(largest_mol)
                        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
                    except:
                        continue
                    largest_mol.SetProp('smiles', smiles)
                    largest_mol.SetProp('qed', str(QED.qed(largest_mol)))
                    largest_mol.SetProp('template_idx', rdmol.GetProp('template_idx'))

                    match = any([largest_mol.HasSubstructMatch(subst) for subst in filter_smarts])

                    if not match:
                        # all_valid_mols.append(largest_mol)
                        all_valid_mols[rdmol.GetProp('template_idx')].append(largest_mol)
                        error_message[-1] += 1
                    # all_valid_mols[rdmol.GetProp('template_idx')].append(largest_mol)
                    # error_message[-1] += 1


            except Chem.rdchem.AtomValenceException:
                error_message[1] += 1
                # print("Valence error in GetmolFrags")
            except Chem.rdchem.KekulizeException:
                error_message[2] += 1
                # print("Can't kekulize molecule")
            except Chem.rdchem.AtomKekulizeException or ValueError: 
                error_message[3] += 1

    print(f"Error messages: AtomValence {error_message[1]}, Kekulize {error_message[2]}, other {error_message[3]}, "
              f" -- No error {error_message[-1]}")

    
    if len(all_valid_mols.keys())>0:
        for idx in all_valid_mols.keys():
            # template = template_dict.get(idx)
            with Chem.SDWriter(f"{out_path}sample_template{idx}.sdf")as f:
                # f.write(template)
                for i, mol in enumerate(all_valid_mols[idx]):
                    mol.SetProp('_Name', f"sample{i+1}")
                    f.write(mol) 


def generate_mols(samples_to_generate, model, potential_ebs, device, based_on_template, pharmacophore_info=None, dataset_info=None):
    assert pharmacophore_info is None

    if samples_to_generate <= 0:
        return []

    samples = []


    template = Batch.from_data_list([get_control_data(pharmacophore_info=pharmacophore_info, id=0, based_on_template=based_on_template) \
        for _ in range(samples_to_generate)])
    

    template_loader = DataLoader(template, potential_ebs, shuffle=True)
    for i, template_batch in enumerate(template_loader):
        template_batch = template_batch.to(device)
        try:
            dense_data = utils.control_to_dense(template_batch, dataset_info=dataset_info, device=device)
        except:
            continue
        dense_data.idx = template_batch.idx

        current_n_list = torch.unique(template_batch.batch, return_counts=True)[1]
        
        samples.extend(model.sample_batch(n_nodes=current_n_list, template=dense_data))

    return samples

@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: omegaconf.DictConfig):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load model
    dataset_infos = MolInfos(statistics_path=cfg.sample.statistics_path, remove_h=cfg.dataset.remove_h)
    model = FullDenoisingDiffusion.load_from_checkpoint(checkpoint_path=cfg.sample.loading_model, map_location={'cuda:1': 'cuda:0'}, dataset_infos=dataset_infos, train_smiles=None)
    model.T = cfg.model.diffusion_steps
    model = model.to(device)

    phar_model_file = cfg.sample.phar_model_file


    with open(phar_model_file, 'r')as f:
        pharmacophore_info = json.load(f)
    molecules = generate_mols(pharmacophore_info=pharmacophore_info, samples_to_generate=cfg.sample.samples_to_generate,
                            model=model, potential_ebs=cfg.sample.potential_ebs, device=device, based_on_template=cfg.sample.based_on_template, dataset_info=dataset_infos)
    # Make SDF files
    with open(cfg.sample.filter_substructure, 'r')as f:
        filter_substructure = json.load(f)
    current_path = os.getcwd()
    result_dir = 'sample'
    result_path = os.path.join(current_path, f"{result_dir}/")
    os.makedirs(result_path, exist_ok=True)
    # out_path = os.path.join(result_path, 'molecules.sdf')
    write_sdf_file(result_path, molecules, filter_substructure)





    

if __name__ == "__main__":
    main()
