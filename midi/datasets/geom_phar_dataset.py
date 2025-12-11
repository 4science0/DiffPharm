import collections
import os
import pathlib
import pickle

from rdkit import Chem, RDLogger
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import BatchSampler
import argparse
from torch_geometric.data import InMemoryDataset, DataLoader
from hydra.utils import get_original_cwd

from utils import PlaceHolder
import datasets.dataset_utils as dataset_utils
from datasets.dataset_utils import load_pickle, save_pickle
from datasets.abstract_dataset import AbstractDatasetInfos, AbstractAdaptiveDataModule
from metrics.metrics_utils import compute_all_statistics
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from collections import defaultdict

full_atom_encoder = {'H': 0, 'B': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'Al': 6, 'Si': 7,
                     'P': 8, 'S': 9, 'Cl': 10, 'As': 11, 'Br': 12, 'I': 13, 'Hg': 14, 'Bi': 15}
full_phar_encoder = {'H_atom': 0, 'Donor': 1, 'Acceptor': 2, 'NegIonizable': 3, 'PosIonizable': 4, 'ZnBinder': 5, 'Ring': 6, 'Hydrophobe': 7, 'LumpedHydrophobe':8, 'Linker': 9}
# full_phar_encoder = {'H_atom': 0, 'Donor': 1, 'Acceptor': 2, 'NegIonizable': 3, 'PosIonizable': 4, 'ZnBinder': 5, 'Ring': 6, 'Hydrophobe': 7, 'Linker': 8}


class GeomDrugsDataset(InMemoryDataset):
    def __init__(self, split, root, dataset_cfg, transform=None, pre_transform=None, pre_filter=None, val_template_num=200, test_template_num=200):
        assert split in ['train', 'val', 'test']
        self.split = split
        self.dataset_cfg = dataset_cfg
        self.remove_h = self.dataset_cfg.remove_h
        self.atom_encoder = full_atom_encoder
        self.phar_encoder = full_phar_encoder
        self.template_name = self.dataset_cfg.template_name if hasattr(self.dataset_cfg, "template_name") else None
        if self.remove_h:
            self.atom_encoder = {k: v - 1 for k, v in self.atom_encoder.items() if k != 'H'}
            self.phar_encoder = {k: v - 1 for k, v in self.phar_encoder.items() if k != 'H_atom'}
        self.val_template_num = val_template_num
        self.test_template_num = test_template_num 
        self.control_data_dict = self.dataset_cfg.control_data_dict
        self.control_add_noise_dict = self.dataset_cfg.control_add_noise_dict

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.statistics = dataset_utils.Statistics(num_nodes=load_pickle(self.processed_paths[1]),
                                                   atom_types=torch.from_numpy(np.load(self.processed_paths[2])),
                                                   phar_types=torch.from_numpy(np.load(self.processed_paths[3])),
                                                   bond_types=torch.from_numpy(np.load(self.processed_paths[4])),
                                                   charge_types=torch.from_numpy(np.load(self.processed_paths[5])),
                                                   valencies=load_pickle(self.processed_paths[6]),
                                                   bond_lengths=load_pickle(self.processed_paths[7]),
                                                   bond_angles=torch.from_numpy(np.load(self.processed_paths[8])))
        self.smiles = load_pickle(self.processed_paths[9])
        if len(self.processed_paths) > 10:
            self.template_data = torch.load(self.processed_paths[10])

        self.data.cx = self.data.cx if self.control_data_dict['cX'] == 'cX' else self.data.x
        self.data.ccharges = self.data.ccharges if self.control_data_dict['cX'] == 'cX' else self.data.charges
        try:
            self.data.cx_sup = self.data.cx_sup
        except:
            pass

        if self.control_data_dict['cE'] == 'cE':
            self.data.cedge_attr = self.data.cedge_attr
        elif self.control_data_dict['cE'] == 'None':
            print(f"control_data_dict['cE']:{self.control_data_dict['cE']}")
            self.data.cedge_attr = torch.zeros_like(self.data.cedge_attr)
        elif self.control_data_dict['cE'] == 'E':
            self.data.cedge_attr = self.data.edge_attr
        elif self.control_data_dict['cE'] == 'single_mask_None':
            mask_tensor = torch.rand(self.data.cedge_attr.shape[0])>0.5
            self.data.cedge_attr[mask_tensor] = 0


        for i, _ in enumerate(self.template_data):
            self.template_data[i].cx = self.template_data[i].cx if self.control_data_dict['cX'] == 'cX' else self.template_data[i].x
            self.template_data[i].ccharges = self.template_data[i].ccharges if self.control_data_dict['cX'] == 'cX' else self.template_data[i].charges
            try:
                self.template_data[i].cx_sup = self.template_data[i].cx_sup
            except:
                pass
            
            if self.control_data_dict['cE'] == 'cE':
                self.template_data[i].cedge_attr = self.template_data[i].cedge_attr
            elif self.control_data_dict['cE'] == 'None':
                self.template_data[i].cedge_attr = torch.zeros_like(self.template_data[i].cedge_attr)
            elif self.control_data_dict['cE'] == 'E':
                self.template_data[i].cedge_attr = self.template_data[i].edge_attr
            elif self.control_data_dict['cE'] == 'single_mask_None':
                mask_tensor = torch.rand(self.template_data[i].cedge_attr.shape[0]) > 0.5
                self.template_data[i].cedge_attr[mask_tensor] = 0


    @property
    def raw_file_names(self):
        if self.split == 'train':
            return ['train_data.txt']
        elif self.split == 'val':
            return ['val_data.txt']
        else:
            if self.template_name is None:
                return ['test_data.txt']
            else:
                return [f'test_data_{self.template_name}.txt']
    @property
    def processed_file_names(self):
        h = 'noh' if self.remove_h else 'h'
        if self.split == 'train':
            return [f'train_{h}.pt', f'train_n_{h}.pickle', f'train_atom_types_{h}.npy', f'train_phar_types_{h}.npy', f'train_bond_types_{h}.npy',
                    f'train_charges_{h}.npy', f'train_valency_{h}.pickle', f'train_bond_lengths_{h}.pickle',
                    f'train_angles_{h}.npy', f'train_smiles.pickle', f'train_template_{h}.pt']
        elif self.split == 'val':
            return [f'val_{h}.pt', f'val_n_{h}.pickle', f'val_atom_types_{h}.npy', f'val_phar_types_{h}.npy', f'val_bond_types_{h}.npy',
                    f'val_charges_{h}.npy', f'val_valency_{h}.pickle', f'val_bond_lengths_{h}.pickle',
                    f'val_angles_{h}.npy', f'val_smiles.pickle', f'val_template_{h}.pt']
        else:
            if self.template_name is None:
                return [f'test_{h}.pt', f'test_n_{h}.pickle', f'test_atom_types_{h}.npy', f'test_phar_types_{h}.npy',f'test_bond_types_{h}.npy',
                        f'test_charges_{h}.npy', f'test_valency_{h}.pickle', f'test_bond_lengths_{h}.pickle',
                        f'test_angles_{h}.npy', f'test_smiles.pickle', f'test_template_{h}.pt']
            else:
                return [f'test_{h}_{self.template_name}.pt', f'test_n_{h}_{self.template_name}.pickle',
                        f'test_atom_types_{h}_{self.template_name}.npy', f'test_phar_types_{h}_{self.template_name}.npy', f'test_bond_types_{h}_{self.template_name}.npy',
                        f'test_charges_{h}_{self.template_name}.npy', f'test_valency_{h}_{self.template_name}.pickle',
                        f'test_bond_lengths_{h}_{self.template_name}.pickle',f'test_angles_{h}_{self.template_name}.npy',
                        f'test_smiles_{self.template_name}.pickle', f'test_template_{h}_{self.template_name}.pt']

    def download(self):
        raise ValueError('Download and preprocessing is manual. If the data is already downloaded, '
                         f'check that the paths are correct. Root dir = {self.root} -- raw files {self.raw_paths}')

    def process(self):
        RDLogger.DisableLog('rdApp.*')
        print(self.split, self.raw_paths[0]) 
        with open(self.raw_paths[0], 'r')as f:
            all_files = f.read().splitlines()
        # all_files = all_files[:200]
        all_smiles = []
        data_list = []
        IA_list = defaultdict(list)
        # all_files = all_files[:200]
        for i, smiles in enumerate(tqdm(all_files)):
            all_smiles.append(smiles)
            # all_conformers = Chem.SDMolSupplier(f"{self.root}/raw/geom_drug_sdf/{smiles}.sdf", removeHs=False)
            all_conformers = Chem.SDMolSupplier(f"/data/lbh/Dataset/geom_drugs/geom_drug_sdf/{smiles}.sdf", removeHs=False)

            for j, conformer in enumerate(all_conformers):
                if j >= 5:
                    break
                
                connected_loop_length = dataset_utils.get_ring_systems(conformer)
                IA_existence = dataset_utils.find_isolated_atom(conformer)
                condition = connected_loop_length < 4 if isinstance(connected_loop_length, int) else not connected_loop_length
                if not IA_existence and condition:
                    data = dataset_utils.mol_to_phar_graph_data4(conformer, full_atom_encoder, full_phar_encoder, smiles, onehot_up = self.dataset_cfg.update_onehot, conversion_ratio=self.dataset_cfg.conversion_ratio)
                    if self.remove_h:
                        data = dataset_utils.remove_hydrogens(data)

                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    data_list.append(data)


        torch.save(self.collate(data_list), self.processed_paths[0])
        # ###########
        template_num = self.test_template_num if self.split=='test' else self.val_template_num
        # random.seed(0)
        # template_data = random.sample(data_list, template_num)
        template_data = data_list[::5][:template_num]
        # template_data = data_list[:template_num]
        for i in range(len(template_data)):
            template_data[i].idx = i
        torch.save(template_data, self.processed_paths[10])

        statistics = compute_all_statistics(data_list, self.atom_encoder, phar_encoder=self.phar_encoder, charges_dic={-2: 0, -1: 1, 0: 2,
                                                                                       1: 3, 2: 4, 3: 5})
        save_pickle(statistics.num_nodes, self.processed_paths[1])
        np.save(self.processed_paths[2], statistics.atom_types)
        np.save(self.processed_paths[3], statistics.phar_types)
        np.save(self.processed_paths[4], statistics.bond_types)
        np.save(self.processed_paths[5], statistics.charge_types)
        save_pickle(statistics.valencies, self.processed_paths[6])
        save_pickle(statistics.bond_lengths, self.processed_paths[7])
        np.save(self.processed_paths[8], statistics.bond_angles)
        save_pickle(set(all_smiles), self.processed_paths[9])


class GeomDataModule(AbstractAdaptiveDataModule):
    def __init__(self, cfg):
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(get_original_cwd()).parents[0]
        root_path = os.path.join(base_path, self.datadir)

        train_dataset = GeomDrugsDataset(split='train', root=root_path, dataset_cfg=cfg.dataset, val_template_num=cfg.general.val_template_num)
        val_dataset = GeomDrugsDataset(split='val', root=root_path, dataset_cfg=cfg.dataset, val_template_num=cfg.general.val_template_num)
        test_dataset = GeomDrugsDataset(split='test', root=root_path, dataset_cfg=cfg.dataset, test_template_num=cfg.general.test_template_num)
        self.remove_h = cfg.dataset.remove_h
        self.statistics = {'train': train_dataset.statistics, 'val': val_dataset.statistics,
                           'test': test_dataset.statistics}
        super().__init__(cfg, train_dataset, val_dataset, test_dataset)


class GeomInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg):
        self.remove_h = cfg.dataset.remove_h
        self.need_to_strip = False        # to indicate whether we need to ignore one output from the model
        self.statistics = datamodule.statistics
        self.name = 'geom'
        self.atom_encoder = full_atom_encoder
        self.phar_encoder = full_phar_encoder
        self.collapse_charges = torch.Tensor([-2, -1, 0, 1, 2, 3]).int()
        if self.remove_h:
            self.atom_encoder = {k: v - 1 for k, v in self.atom_encoder.items() if k != 'H'}
            self.phar_encoder = {k: v - 1 for k, v in self.phar_encoder.items() if k != 'H_atom'}

        super().complete_infos(datamodule.statistics, self.atom_encoder, self.phar_encoder)
        self.input_dims = PlaceHolder(X=self.num_atom_types, charges=6, E=5, y=1, pos=3)
        self.output_dims = PlaceHolder(X=self.num_atom_types, charges=6, E=5, y=0, pos=3)

    def to_one_hot(self, X, charges, E, node_mask, X_sup = None, just_control=False):
        x = X.clone()
        X = F.one_hot(X, num_classes=self.num_atom_types).float()
        if X_sup is not None:
            assert X_sup.shape == x.shape
            X_sup = F.one_hot(X_sup, num_classes=self.num_atom_types).float()
            X_sup[:,:,0] = torch.zeros(X_sup[:,:,0].shape).float()
            X += X_sup
        E = F.one_hot(E, num_classes=5).float()
        charges = F.one_hot(charges + 2, num_classes=6).float()
        placeholder = PlaceHolder(X=X, charges=charges, E=E,  y=None, pos=None)
        pl = placeholder.mask(node_mask, just_control)
        return pl.X, pl.charges, pl.E

    def one_hot_charges(self, charges):
        return F.one_hot((charges + 2).long(), num_classes=6).float()

# def to_one_hot(X, charges, E, node_mask, X_sup = None, just_control=False):
#     num_atom_types = 16
#     x = X.clone()
#     X = F.one_hot(X, num_classes=num_atom_types).float()
#     if X_sup is not None:
#         assert X_sup.shape == x.shape
#         X_sup = F.one_hot(X_sup, num_classes=num_atom_types).float()
#         X_sup[:,:,0] = torch.zeros(X_sup[:,:,0].shape).float()
#         X += X_sup
#     E = F.one_hot(E, num_classes=5).float()
#     charges = F.one_hot(charges + 2, num_classes=6).float()
#     placeholder = PlaceHolder(X=X, charges=charges, E=E,  y=None, pos=None)
#     pl = placeholder.mask(node_mask, just_control)
#     return pl.X, pl.charges, pl.E