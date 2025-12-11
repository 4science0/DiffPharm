import os
import pathlib
import pickle

from rdkit import Chem, RDLogger
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

from itertools import islice
from torch_geometric.data import InMemoryDataset, DataLoader
from hydra.utils import get_original_cwd
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from collections import defaultdict
from utils import PlaceHolder
import datasets.dataset_utils as dataset_utils
from datasets.dataset_utils import load_pickle, save_pickle
from datasets.abstract_dataset import AbstractDatasetInfos, AbstractAdaptiveDataModule
from datasets.adaptive_loader import AdaptiveDataLoader
from metrics.metrics_utils import compute_all_statistics


full_atom_encoder = {'H': 0, 'B': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'Al': 6, 'Si': 7,
                     'P': 8, 'S': 9, 'Cl': 10, 'As': 11, 'Br': 12, 'I': 13, 'Hg': 14, 'Bi': 15}
# full_atom_encoder = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'P': 5, 'S': 6, 'Cl': 7, 'Br': 8, 'I': 9}
full_phar_encoder = {'Linker': 0, 'Donor': 1, 'Acceptor': 2, 'NegIonizable': 3, 'PosIonizable': 4, 'Aromatic': 5, 'Hydrophobe': 6, 'LumpedHydrophobe':7}

class LargeZincDrugsDataset(InMemoryDataset):
    def __init__(self, split, root, dataset_cfg, batch_id=None, transform=None, pre_transform=None, pre_filter=None, val_template_num=200, test_template_num=200):
        assert split in ['train', 'val', 'test']
        self.split = split
        self.dataset_cfg = dataset_cfg
        self.remove_h = self.dataset_cfg.remove_h
        self.batch_id = batch_id  # which batch to load
        self.atom_encoder = full_atom_encoder
        self.template_name = self.dataset_cfg.template_name if hasattr(self.dataset_cfg, "template_name") else None
        if self.remove_h:
            self.atom_encoder = {k: v - 1 for k, v in self.atom_encoder.items() if k != 'H'}
        self.linker_strategy = self.dataset_cfg.linker_strategy
        self.val_template_num = val_template_num
        self.test_template_num = test_template_num
        self.control_add_noise_dict = self.dataset_cfg.control_add_noise_dict
        
        super().__init__(root, transform, pre_transform, pre_filter)

        self.template_data = None
        if split == 'train':
            path = self.processed_paths[batch_id]  # load single batch
        else:
            path = self.processed_paths[0]  # val/test only has 1 batch

            self.template_data = torch.load(self.processed_paths[-1], weights_only=False)

        self.data, self.slices = torch.load(path, weights_only=False)

        # Load smiles and statistics

        self.smiles = load_pickle(self.processed_paths[-2])
        self.statistics = dataset_utils.Statistics(
            num_nodes=load_pickle(self.processed_paths[-9]),
            atom_types=torch.from_numpy(np.load(self.processed_paths[-8])),
            bond_types=torch.from_numpy(np.load(self.processed_paths[-7])),
            charge_types=torch.from_numpy(np.load(self.processed_paths[-6])),
            valencies=load_pickle(self.processed_paths[-5]),
            bond_lengths=load_pickle(self.processed_paths[-4]),
            bond_angles=torch.from_numpy(np.load(self.processed_paths[-3])),
        )
        


    @property
    def processed_dir(self):
        if self.linker_strategy == 'fragment':
            return os.path.join(self.root, 'processed_pharm_fragment_linker')
        if self.linker_strategy == 'cluster':
            length = '_10' if len(self.atom_encoder.keys()) in [9,10] else ''
            return os.path.join(self.root, 'processed_pharm_cluster_linker' + length)

    @property
    def raw_file_names(self):
        expected_file = f'zinc_{self.split}.sdf'
        expected_path = os.path.join(self.raw_dir, expected_file)

        if os.path.exists(expected_path):
            return [expected_file]
        else:
            dummy_path = os.path.join(self.raw_dir, 'skip.txt')
            if not os.path.exists(dummy_path):
                with open(dummy_path, 'w') as f:
                    f.write("Dummy file to bypass raw_file_names requirement.")
            return ['skip.txt']

    @property
    def processed_file_names(self):
        h= 'noh' if self.remove_h else 'h'
        data_paths = [f'train_batch_{h}_{i}.pt' for i in range(10)] if self.split == 'train' else [f'{self.split}_{h}.pt']
        
        statistics_paths = [f'{self.split}_n_{h}.pickle', f'{self.split}_atom_types_{h}.npy', f'{self.split}_bond_types_{h}.npy',
                            f'{self.split}_charges_{h}.npy', f'{self.split}_valency_{h}.pickle', f'{self.split}_bond_lengths_{h}.pickle',
                            f'{self.split}_angles_{h}.npy', f'{self.split}_smiles.pickle', f'{self.split}_template_{h}.pt']
        return data_paths + statistics_paths

    def download(self):
        raise ValueError("Download manually.")

    def process(self):
        RDLogger.DisableLog('rdApp.*')
        path = self.raw_paths[0]
        h= 'noh' if self.remove_h else 'h'
        pharmacophore_statistics = {'Donor': defaultdict(int), 'Acceptor': defaultdict(int), 'NegIonizable': defaultdict(int), 'PosIonizable': defaultdict(int), 'Aromatic': defaultdict(int), 'Hydrophobe': defaultdict(int)}
        mol_supplier = Chem.SDMolSupplier(path, removeHs=False)

        batch_size = 500_000
        data_list = []
        all_smiles = []
        batch_id = 0

        for i, mol in enumerate(tqdm(mol_supplier, desc=f"Processing {self.split}")):
            if mol is None:
                continue
            mol,smiles = dataset_utils.keep_largest_fragment(mol)
            all_smiles.append(smiles)
            if self.remove_h:
                mol = Chem.RemoveHs(mol)
                if any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()):
                    continue
            try:
                data, pharmacophore_statistics = dataset_utils.mol_to_pharmacophore(mol, smiles, full_phar_encoder, pharmacophore_statistics, self.linker_strategy, self.remove_h)
            except:
                continue
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

            # 保存一个 batch
            if len(data_list) >= batch_size:
                torch.save(self.collate(data_list), self.processed_paths[batch_id])
                save_pickle(set(all_smiles), self.processed_paths[-2])
                print(f"✅ Saved batch {batch_id} with {len(data_list):,} molecules")

                if batch_id == 0:
                    statistics = compute_all_statistics(data_list, self.atom_encoder, charges_dic={-2: 0, -1: 1, 0: 2, 1: 3, 2: 4, 3: 5})
                    save_pickle(statistics.num_nodes, self.processed_paths[-9])
                    np.save(self.processed_paths[-8], statistics.atom_types)
                    np.save(self.processed_paths[-7], statistics.bond_types)
                    np.save(self.processed_paths[-6], statistics.charge_types)
                    save_pickle(statistics.valencies, self.processed_paths[-5])
                    save_pickle(statistics.bond_lengths, self.processed_paths[-4])
                    np.save(self.processed_paths[-3], statistics.bond_angles)

                data_list, all_smiles = [], []
                batch_id += 1

        if len(data_list) > 0:
            torch.save(self.collate(data_list), self.processed_paths[batch_id])
            save_pickle(set(all_smiles), self.processed_paths[-2])
            print(f"✅ Saved last batch {batch_id} with {len(data_list):,} molecules")

            if self.split in ['val', 'test', 'train'] and batch_id == 0:
                template_num = self.test_template_num if self.split=='test' else self.val_template_num
                template_data = data_list[::5][:template_num]
                for i in range(len(template_data)):
                    template_data[i].idx = str(i)
                torch.save(template_data, self.processed_paths[-1])
                
                statistics = compute_all_statistics(data_list, self.atom_encoder, charges_dic={-2: 0, -1: 1, 0: 2, 1: 3, 2: 4, 3: 5})
                save_pickle(statistics.num_nodes, self.processed_paths[-9])
                np.save(self.processed_paths[-8], statistics.atom_types)
                np.save(self.processed_paths[-7], statistics.bond_types)
                np.save(self.processed_paths[-6], statistics.charge_types)
                save_pickle(statistics.valencies, self.processed_paths[-5])
                save_pickle(statistics.bond_lengths, self.processed_paths[-4])
                np.save(self.processed_paths[-3], statistics.bond_angles)


class LargeZincDataModule(AbstractAdaptiveDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(get_original_cwd()).parents[0]
        root_path = os.path.join(base_path, self.datadir)
        self.root_path = root_path
        self.remove_h = cfg.dataset.remove_h
        self.epoch_counter = 0

        # For val/test, only one batch (batch_id=None)
        val_dataset = LargeZincDrugsDataset(split='val', root=root_path, dataset_cfg=cfg.dataset, val_template_num=cfg.general.val_template_num)
        test_dataset = LargeZincDrugsDataset(split='test', root=root_path, dataset_cfg=cfg.dataset, test_template_num=cfg.general.test_template_num)

        # For train, will be switched per epoch
        train_dataset = LargeZincDrugsDataset(split='train', root=root_path, dataset_cfg=cfg.dataset, val_template_num=cfg.general.val_template_num, batch_id=0)
        self.statistics = {
            'train': train_dataset.statistics,
            'val': val_dataset.statistics,
            'test': test_dataset.statistics
        }
        
        self.reload_dataloaders_every_n_epochs = 1  # 每一轮都重新加载 dataloader
        # self.train_dataset = LargeZincDrugsDataset(split='train', root=root_path, remove_h=self.remove_h, batch_id=self.epoch_counter)

        super().__init__(cfg, train_dataset, val_dataset, test_dataset)
    
    
    def train_dataloader(self):
        # 每轮使用一个分片作为训练数据
        idx = self.trainer.current_epoch % 10  # 取 Trainer 实际 epoch
        rank_zero_info(f"[DataModule] Epoch {self.trainer.current_epoch}: Loading train_batch_{idx}.pt")
        # print(f"[DataModule] Epoch {self.trainer.current_epoch}: Loading training data from train_batch_{idx}.pt")
        self.train_dataset = LargeZincDrugsDataset(
            root=self.root_path, split='train', batch_id=idx, dataset_cfg=self.cfg.dataset
        )
        self.epoch_counter += 1

        return self.dataloader(self.train_dataset)





class ZincInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg):
        self.remove_h = cfg.dataset.remove_h
        self.need_to_strip = False        # to indicate whether we need to ignore one output from the model
        self.statistics = datamodule.statistics
        self.name = 'zinc'
        self.atom_encoder = full_atom_encoder
        self.collapse_charges = torch.Tensor([-2, -1, 0, 1, 2, 3]).int()
        if self.remove_h:
            self.atom_encoder = {k: v - 1 for k, v in self.atom_encoder.items() if k != 'H'}

        super().complete_infos(datamodule.statistics, self.atom_encoder)
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
