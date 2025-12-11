# 药效团生成流程

本文档提供了一个逐步指南，用于从参考分子 (`.sdf` 文件) 中生成和准备药效团特征。整个过程分为四个主要步骤：生成药效团、选择相关特征、更新文件、以及在 YAML 文件中配置采样参数。

---

## Step 1. 生成药效团

假设输入的分子文件为 `/mols/ref_mol.sdf`，其 `_Name = Mol`。

### 情况 A: 保留氢原子（全原子分子）

* 使用 **fragment 策略**（默认）：

  ```bash
  python gen_pharm.py --sdf /mols/ref_mol.sdf
  ```
* 该命令会在当前目录下生成名为 **Molecular pharmacophore results** 的文件夹。
  在其中的 `Mol` 文件夹下包含该分子 `ref.sdf` 的所有结果：

  * `all_pharmacophores_h.json`: 所有药效团信息。

    * 列说明:

      * \[[药效团索引, [药效团名称, 药效团原子数量, 药效团质心位置]],……, 分子总原子数]
  * `pharm_feature_fragment-L-h.json`: 随机选择的药效团信息。
  * `template_fragment-L-h.pt`: 对应随机选择的药效团的 Torch tensor 文件。

⚠️ 注意: 此阶段的药效团模型为 **随机选择**，不能直接用于分子生成。

### 情况 B: 去除氢原子

* 使用 **cluster 策略** 并加上 `--remove_h`：

  ```bash
  python gen_pharm.py --sdf /mols/ref_mol.sdf --remove_h --linker_strategy cluster
  ```
* 输出结果：

  * `all_pharmacophores_noh.json`: 所有药效团信息（由于去氢，药效团质心会与保留氢时不同）。
  * `pharm_feature_cluster-L-noh.json`: 随机选择的药效团信息。
  * `template_cluster-L-noh.pt`: 对应随机选择的药效团 Torch tensor 文件。

---

## Step 2. 选择药效团特征

查看 `all_pharmacophores_h.json`（保留氢原子）或 `all_pharmacophores_noh.json`（去氢），选择将用于分子生成的药效团索引。

例如：

```python
chosen_indices = [2, 7, 8, 4, 5]
```

---

## Step 3. 使用选定特征重新生成 JSON 和 PT 文件

使用选定的药效团索引重新运行 `gen_pharm.py`：

* 保留氢：

  ```bash
  python gen_pharm.py --sdf /mols/ref_mol.sdf --chosen [2,7,8,4,5]
  ```

* 去除氢：

  ```bash
  python gen_pharm.py --sdf /mols/ref_mol.sdf --remove_h --linker_strategy cluster --chosen [2,7,8,4,5]
  ```

这样会生成新的 `.json` 和 `.pt` 文件，可供模型使用。

---

## Step 4. 固定片段中的原子

确定需要在生成过程中保持固定的 fragment 原子索引。例如：

```python
fix_atoms = [0, 1, 2, 13, 20, 21, 25, 28, 29]
```

将该索引列表填入对应 YAML 配置文件中的 `sample.fix_atoms`。

---

## YAML 配置参数

```yaml
general:
  name: 项目名称（可自定义）

sample:
  loading_model: # 根据策略选择模型 checkpoint
    # fragment 策略 + 保留氢:
    #   LinkerFragment_withh_nstd0.3_16t.ckpt
    # cluster 策略 + 去氢:
    #   LinkerCluster_noh_nstd0.3_16t.ckpt

  sdf_path: 输入 .sdf 分子文件路径
  pt_path: 前 3 步生成的 .pt 文件路径
  fix_atoms: [0, 1, 2, 13, 20, 21, 25, 28, 29]

  # 仅在 cluster 策略下使用
  saturated_atoms: [ ] # fix_atoms 的子集，这些原子不允许再长出新结构
                       # 如果无需限制，设为 null

  statistics_path: 统计信息文件夹路径（根据策略，选择对应推荐默认）
  # fragment 策略 + 保留氢:
    #   processed_pharm_fragment_linker
    # cluster 策略 + 去氢:
    #   processed_pharm_cluster_linker
  samples_to_generate: 需要生成的数量（实际有效数量可能少于此值）
  potential_ebs: batch size
  resamplings: inpainting 重采样次数（默认 1）

dataset:
  remove_h: # cluster 策略为 True，否则为 False
```

---

## 注意事项

* 策略选择（`fragment` vs. `cluster`）取决于是否保留氢原子。
* `saturated_atoms` 参数仅在 **cluster 策略** 下使用。
* `samples_to_generate` 表示尝试生成的分子数量，有效分子可能更少。
* 在采样前必须确保 `.pt` 和 `.json` 文件已用正确的 `--chosen` 药效团索引重新生成。

---

假定 YAML 的名字为 `inpainting`，生成命令为：

```bash
CUDA_VISIBLE_DEVICES=0 python inpainting4fragment.py +experiment=inpainting
```
