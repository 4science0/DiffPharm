import torch
from rdkit import Chem
from torch_geometric.data.batch import Batch
from torch_geometric.loader import DataLoader
import itertools
from rdkit.Chem import QED
import os
import utils
from metrics.molecular_metrics import filter_substructure
from datasets import dataset_utils
from datasets import chembl_geom_dataset,geom_phar_dataset
# from datasets.geom_phar_dataset import full_atom_encoder,full_phar_encoder
from diffusion_model import FullDenoisingDiffusion
import hydra
import omegaconf
from collections import Counter
import copy
from analysis.rdkit_functions import Molecule
import json
from datasets.dataset_utils import MolInfos

def data_from_sdf(sdf_path, pt_path, fix_atoms):
    mol = torch.load(pt_path,weights_only=False)[0]
    orign_mol = copy.deepcopy(mol)
    orign_mol.idx = 0

    remove_atoms = torch.tensor(fix_atoms) # 4kd1

    total_num = len(mol.x)
    all_atoms = torch.arange(0, total_num)
    mask = torch.zeros_like(all_atoms, dtype=bool)
    mask[remove_atoms] = True
    fixed_atoms = all_atoms[mask]
    

    rdmol = Chem.SDMolSupplier(sdf_path, removeHs=False)
    rdmol = next(rdmol)
    rdmol.SetProp('_Name', f"template")

    return mol, rdmol, fixed_atoms

def write_sdf_file(out_path, sample_template_mol, samples):
    all_valid_mols = list()
    all_invalid_mols = list()
    error_message = Counter()
    filter_smarts = [Chem.MolFromSmarts(subst) for subst in filter_substructure if Chem.MolFromSmarts(subst)]
    for mol in samples:
        rdmol = mol.rdkit_mol
        if rdmol is not None:
            try:
                mol_frags = Chem.rdmolops.GetMolFrags(rdmol, asMols=True, sanitizeFrags=False)
                largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                Chem.SanitizeMol(largest_mol)
                smiles = Chem.MolToSmiles(largest_mol)
                mol_from_smiles = Chem.MolFromSmiles(smiles)
                if mol_from_smiles is not None:
                    smiles = Chem.MolToSmiles(mol_from_smiles)
                    largest_mol.SetProp('smiles', smiles)
                    largest_mol.SetProp('qed', str(QED.qed(largest_mol)))

                    match = any([largest_mol.HasSubstructMatch(subst) for subst in filter_smarts])

                    if not match:
                        all_valid_mols.append(largest_mol)
                        error_message[-1] += 1
                    else:
                        all_invalid_mols.append(largest_mol)
                    # all_valid_mols.append(largest_mol)
    
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
    if len(all_valid_mols) > 0:
        with Chem.SDWriter(out_path)as f:
            f.write(sample_template_mol)
            for i, mol in enumerate(all_valid_mols):
                mol.SetProp('_Name', f'gen_{i}')
                f.write(mol)
    out_path = out_path.replace('.sdf', '_invalid.sdf')
    if len(all_invalid_mols) > 0:
        with Chem.SDWriter(out_path)as f:
            f.write(sample_template_mol)
            for i, mol in enumerate(all_invalid_mols):
                mol.SetProp('_Name', f'gen_invalid_{i}')
                f.write(mol)



def inpaint_mol(model, sdf_path, pt_path, fix_atoms, samples_to_generate, potential_ebs, device, dataset_infos=None, saturated_atoms=None, resamplings=1):
    if samples_to_generate <= 0:
        return []
    
    # Load SDF
    inpainting_template, template_mol, fixed_atoms = data_from_sdf(sdf_path,pt_path, fix_atoms)
    fixed_atoms.to(device)

    samples = []
    template = Batch.from_data_list(list(itertools.repeat(inpainting_template, samples_to_generate)))
    template_loader = DataLoader(template, potential_ebs, shuffle=True)
    total_batches = len(template_loader)
    for i, template_batch in enumerate(template_loader):
        print(f"🔄 Sampling of batch {i+1} / {total_batches} is in progress")
        template_batch = template_batch.to(device)
        template_batch.idx = i
        dense_data = utils.to_dense(template_batch, dataset_infos)
        current_n_list = torch.unique(template_batch.batch, return_counts=True)[1]

        # saturated_atoms = torch.tensor([9,10,36,37]) #keras
        if saturated_atoms is not None:
            saturated_atoms = torch.tensor(saturated_atoms)
        # Run sampling
        samples.extend(model.inpainting_sample_batch(n_nodes = current_n_list, fixed_data=dense_data, 
                                                     fixed_atoms=fixed_atoms, resamplings=resamplings, initial_data=None, saturated_atoms=saturated_atoms))
    return template_mol, samples
    
    

@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: omegaconf.DictConfig):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset_infos = MolInfos(statistics_path=cfg.sample.statistics_path, remove_h=cfg.dataset.remove_h)
    model = FullDenoisingDiffusion.load_from_checkpoint(checkpoint_path=cfg.sample.loading_model, map_location={'cuda:1': 'cuda:0'}, dataset_infos=dataset_infos, train_smiles=None)
    model.T = cfg.model.diffusion_steps
    model = model.to(device)
    template_mol, molecules = inpaint_mol(model,
                                          cfg.sample.sdf_path,
                                          cfg.sample.pt_path,
                                          cfg.sample.fix_atoms,
                                          samples_to_generate=cfg.sample.samples_to_generate,
                                          potential_ebs=cfg.sample.potential_ebs, 
                                          device=device,
                                          saturated_atoms=cfg.sample.saturated_atoms,
                                          resamplings=cfg.sample.resamplings,
                                          dataset_infos=dataset_infos)
    
    # Make SDF files
    current_path = os.getcwd()
    result_dir = 'sample'
    result_path = os.path.join(current_path, f"{result_dir}/")
    os.makedirs(result_path, exist_ok=True)
    out_path = os.path.join(result_path, 'molecules.sdf')
    write_sdf_file(out_path, template_mol, molecules)
    

if __name__ == "__main__": 
    main()

