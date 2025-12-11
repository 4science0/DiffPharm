# Pharmacophore Generation Workflow

This document provides a step-by-step guide for generating and preparing pharmacophore features from a reference molecule (`.sdf` file). The process involves four main steps: generating pharmacophores, selecting relevant features, refining the files, and configuring the sampling parameters in a YAML file.

⚠️ The `linker-free pharmacophore` mentioned in the main manuscript corresponds to what is referred to later as `cluster`,” while the `linker-aligned pharmacophore` corresponds to  `fragment`.

---

## Step 1. Generate Pharmacophores

Suppose the input molecule file is `/mols/ref_mol.sdf` with `_Name = Mol`.

### Case A: With Hydrogen Atoms (all-atom molecule)

* Use the **fragment strategy** (default):

  ```bash
  python gen_pharm.py --sdf /mols/ref_mol.sdf
  ```
* This will create a folder named **Molecular pharmacophore results** in the current directory.
  Inside, the `Mol` folder will contain all results for `ref.sdf`:

  * `all_pharmacophores_h.json`: All pharmacophore information.

    * Columns: \[[Pharmacophore Index, [Pharmacophore Name, Number of Atoms in Pharmacophore, Pharmacophore Centroid Position],……, Total Atom Count]]
  * `pharm_feature_fragment-L-h.json`: Randomly selected pharmacophore information.
  * `template_fragment-L-h.pt`: Torch tensor file corresponding to the random pharmacophore selection.

⚠️ Note: The pharmacophore model at this stage is **randomly chosen** and cannot be directly used for generation as your mind.

### Case B: Without Hydrogen Atoms

* Use the **cluster strategy** with `--remove_h`:

  ```bash
  python gen_pharm.py --sdf /mols/ref_mol.sdf --remove_h --linker_strategy cluster
  ```
* Outputs:

  * `all_pharmacophores_noh.json`: All pharmacophore information (different from the H-included case due to centroid changes after removing hydrogens).
  * `pharm_feature_cluster-L-noh.json`: Random pharmacophore selection.
  * `template_cluster-L-noh.pt`: Torch tensor file corresponding to the random selection.

---

## Step 2. Select Pharmacophore Features

Review `all_pharmacophores_h.json` (for H-included) or `all_pharmacophores_noh.json` (for H-removed). Choose the pharmacophore indices to be used for molecule generation.

For example:

```python
chosen_indices = [2, 7, 8, 4, 5]
```

---

## Step 3. Regenerate JSON and PT Files with Chosen Features

Re-run `gen_pharm.py` with the chosen pharmacophore indices:

* With hydrogens:

  ```bash
  python gen_pharm.py --sdf /mols/ref_mol.sdf --chosen [2,7,8,4,5]
  ```

* Without hydrogens:

  ```bash
  python gen_pharm.py --sdf /mols/ref_mol.sdf --remove_h --linker_strategy cluster --chosen [2,7,8,4,5]
  ```

This regenerates the `.json` and `.pt` files for model use.

---

## Step 4. Fix Atoms in the Fragment

Identify the atom indices of the fragment that must remain fixed during generation. For example:

```python
fix_atoms = [0, 1, 2, 13, 20, 21, 25, 28, 29]
```

Fill these indices into the corresponding YAML configuration file under `sample.fix_atoms`.

---

## YAML Configuration Parameters

```yaml
general:
  name: Project name (user-defined)

sample:
  loading_model: # Choose model checkpoint according to strategy
    # Fragment strategy with hydrogens:
    #   LinkerFragment_withh_nstd0.3_16t.ckpt
    # Cluster strategy without hydrogens:
    #   LinkerCluster_noh_nstd0.3_16t.ckpt

  sdf_path: Path to the input .sdf molecule file
  pt_path: Path to the generated .pt file (from Steps 1–3)
  fix_atoms: [0, 1, 2, 13, 20, 21, 25, 28, 29]

  # For cluster strategy only
  saturated_atoms: [ ] # Subset of fix_atoms where no new bonds are allowed
                       # If not required, set to null

  statistics_path: Path to statistics folder (default recommended according to the strategy)
    # Fragment strategy with hydrogens:
    #   processed_pharm_fragment_linker
    # Cluster strategy without hydrogens:
    #   processed_pharm_cluster_linker
  samples_to_generate: Number of molecules to generate (may differ from final valid count)
  potential_ebs: Batch size
  resamplings: Number of resamplings in inpainting (default = 1)

dataset:
  remove_h: # True for cluster strategy, False otherwise
```

---

## Notes

* The choice of strategy (`fragment` vs. `cluster`) depends on whether hydrogen atoms are included.
* `saturated_atoms` only applies in **cluster mode**.
* `samples_to_generate` specifies how many molecules to attempt; actual valid outputs may be fewer.
* Always ensure that `.pt` and `.json` files are regenerated with the correct `--chosen` pharmacophore indices before sampling.

---

Assumed YAML name `inpainting_cluster` or `inpainting_fragment`, generation instruction is:

```bash
CUDA_VISIBLE_DEVICES=0 python diffpharm_inpainting.py +experiment=inpainting_cluster
or
CUDA_VISIBLE_DEVICES=0 python diffpharm_inpainting.py +experiment=inpainting_fragment
```

