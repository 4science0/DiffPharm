import math
import numpy as np
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from midi.datasets.dataset_utils import Statistics
from torchmetrics import MeanAbsoluteError
from collections import Counter
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool

def compute_all_statistics(data_list, atom_encoder, charges_dic, phar_encoder=None): 
    num_nodes = node_counts(data_list)
    atom_types = atom_type_counts(data_list, num_classes=len(atom_encoder)) #control_data的atom_types不一样，为：array([1.0, 0, 0, 0, 0])
    print(f"Atom types: {atom_types}")

    bond_types = edge_counts(data_list) #control_data的bond_types一样
    print(f"Bond types: {bond_types}")
    charge_types = charge_counts(data_list, num_classes=len(atom_encoder), charges_dic=charges_dic)#control_data的charge_types不一样
    print(f"Charge types: {charge_types}")

    valency = valency_count(data_list, atom_encoder)#每个atom的价统计 #control_data的valency不一样
    print("Valency: ", valency)
    bond_lengths = bond_lengths_counts(data_list) #control_data的valency不一样
    print("Bond lengths: ", bond_lengths)
    angles = bond_angles(data_list, atom_encoder)#control_data的angles不一样

    if phar_encoder is not None:
        phar_types = phar_type_counts(data_list, num_classes=len(phar_encoder)) #control_data的phar_types一样
        print(f"Phar types: {phar_types}")
        return Statistics(num_nodes=num_nodes, atom_types=atom_types, phar_types=phar_types, bond_types=bond_types, charge_types=charge_types,
                        valencies=valency, bond_lengths=bond_lengths, bond_angles=angles)
    else:
        return Statistics(num_nodes=num_nodes, atom_types=atom_types, bond_types=bond_types, charge_types=charge_types,
                      valencies=valency, bond_lengths=bond_lengths, bond_angles=angles)


def worker_node_counts(data_chunk, result_queue):
    """
    子进程处理函数：计算部分数据的节点计数并将结果发送到队列
    :param data_chunk: 子任务数据
    :param result_queue: 用于通信的队列
    """
    local_counts = Counter()
    for data in data_chunk:
        num_nodes = data.num_nodes
        local_counts[num_nodes] += 1
    result_queue.put(local_counts)  # 将结果发送到队列

def merge_counters(counters):
    """合并多个 Counter 对象"""
    total_counts = Counter()
    for counter in counters:
        total_counts.update(counter)
    return total_counts

def node_counts(data_list, num_processes=8):
    """
    使用 multiprocessing.Process 手动管理的多进程节点计数
    :param data_list: 数据列表，每个元素为 torch_geometric.data.Data
    :param num_processes: 并行进程数
    :return: 所有节点计数的 Counter
    """
    print("Computing node counts in parallel...")
    
    # 分割数据为多个子任务
    chunk_size = math.ceil(len(data_list) / num_processes)
    chunks = [data_list[i:i + chunk_size] for i in range(0, len(data_list), chunk_size)]
    
    # 创建队列用于存储结果
    result_queue = multiprocessing.Queue()
    processes = []

    # 创建并启动进程
    for chunk in chunks:
        p = multiprocessing.Process(target=worker_node_counts, args=(chunk, result_queue))
        processes.append(p)
        p.start()

    # 使用 tqdm 跟踪子任务完成情况
    results = []
    with tqdm(total=len(chunks), desc="Processing Chunks") as pbar:
        for _ in range(len(chunks)):
            results.append(result_queue.get())  # 从队列中获取结果
            pbar.update(1)

    # 等待所有进程结束
    for p in processes:
        p.join()

    # 合并结果
    all_node_counts = merge_counters(results)
    print("Done.")
    return all_node_counts


def worker_atom_type_counts(data_chunk, num_classes, result_queue):
    """
    子进程函数：计算部分数据的原子类型分布
    :param data_chunk: 子任务数据
    :param num_classes: 原子类型的类别数量
    :param result_queue: 用于通信的队列
    """
    local_counts = np.zeros(num_classes)
    for data in data_chunk:
        x = torch.nn.functional.one_hot(data.x, num_classes=num_classes)
        local_counts += x.sum(dim=0).numpy()
    result_queue.put(local_counts)  # 将局部结果发送到队列

def atom_type_counts(data_list, num_classes, num_processes=8):
    """
    多进程计算原子类型分布
    :param data_list: 数据列表，每个元素为 torch_geometric.data.Data
    :param num_classes: 原子类型的类别数量
    :param num_processes: 并行进程数
    :return: 归一化的原子类型分布
    """
    print("Computing node types distribution in parallel...")
    
    # 分割数据为多个子任务
    chunk_size = math.ceil(len(data_list) / num_processes)
    chunks = [data_list[i:i + chunk_size] for i in range(0, len(data_list), chunk_size)]

    # 创建队列用于存储结果
    result_queue = multiprocessing.Queue()
    processes = []

    # 创建并启动进程
    for chunk in chunks:
        p = multiprocessing.Process(target=worker_atom_type_counts, args=(chunk, num_classes, result_queue))
        processes.append(p)
        p.start()

    # 使用 tqdm 跟踪子任务完成情况
    results = []
    with tqdm(total=len(chunks), desc="Processing Chunks") as pbar:
        for _ in range(len(chunks)):
            results.append(result_queue.get())  # 从队列中获取结果
            pbar.update(1)

    # 等待所有进程结束
    for p in processes:
        p.join()

    # 合并所有结果
    total_counts = np.zeros(num_classes)
    for result in results:
        total_counts += result

    # 归一化分布
    total_counts = total_counts / total_counts.sum()
    print("Done.")
    return total_counts

def worker_phar_type_counts(data_chunk, num_classes, result_queue):
    """
    子进程函数：计算部分数据的药效团类型分布
    :param data_chunk: 子任务数据
    :param num_classes: 药效团类型类别数量
    :param result_queue: 用于通信的队列
    """
    local_counts = np.zeros(num_classes)
    for data in data_chunk:
        x = torch.nn.functional.one_hot(data.cx, num_classes=num_classes)
        try:
            if data.cx_sup.shape == data.cx.shape:
                x_sup = torch.nn.functional.one_hot(data.cx_sup, num_classes=num_classes)
                x_sup[:, :, 0] = torch.zeros(x_sup[:, :, 0].shape).float()
                x += x_sup
        except AttributeError:  # 如果 cx_sup 不存在或无效，跳过
            pass
        local_counts += x.sum(dim=0).numpy()
    result_queue.put(local_counts)  # 将局部结果发送到队列

def phar_type_counts(data_list, num_classes, num_processes=8):
    """
    多进程计算药效团类型分布
    :param data_list: 数据列表，每个元素为 torch_geometric.data.Data
    :param num_classes: 药效团类型类别数量
    :param num_processes: 并行进程数
    :return: 归一化的药效团类型分布
    """
    print("Computing node pharmacophore types distribution in parallel...")
    
    # 分割数据为多个子任务
    chunk_size = math.ceil(len(data_list) / num_processes)
    chunks = [data_list[i:i + chunk_size] for i in range(0, len(data_list), chunk_size)]

    # 创建队列用于存储结果
    result_queue = multiprocessing.Queue()
    processes = []

    # 创建并启动进程
    for chunk in chunks:
        p = multiprocessing.Process(target=worker_phar_type_counts, args=(chunk, num_classes, result_queue))
        processes.append(p)
        p.start()

    # 使用 tqdm 跟踪子任务完成情况
    results = []
    with tqdm(total=len(chunks), desc="Processing Chunks") as pbar:
        for _ in range(len(chunks)):
            results.append(result_queue.get())  # 从队列中获取结果
            pbar.update(1)

    # 等待所有进程结束
    for p in processes:
        p.join()

    # 合并所有结果
    total_counts = np.zeros(num_classes)
    for result in results:
        total_counts += result

    # 归一化分布
    total_counts = total_counts / total_counts.sum()
    print("Done.")
    return total_counts

def worker_edge_counts(data_chunk, num_bond_types, result_queue):
    """
    子进程函数：计算部分数据的边类型分布
    :param data_chunk: 子任务数据
    :param num_bond_types: 边的类型数量
    :param result_queue: 用于通信的队列
    """
    local_counts = np.zeros(num_bond_types)
    for data in data_chunk:
        total_pairs = data.num_nodes * (data.num_nodes - 1)

        num_edges = data.edge_attr.shape[0]
        num_non_edges = total_pairs - num_edges
        assert num_non_edges >= 0

        edge_types = torch.nn.functional.one_hot(data.edge_attr - 1, num_classes=num_bond_types - 1).sum(dim=0).numpy()
        local_counts[0] += num_non_edges  # 非边计数
        local_counts[1:] += edge_types  # 边类型计数

    result_queue.put(local_counts)  # 将局部结果发送到队列

def edge_counts(data_list, num_bond_types=5, num_processes=8):
    """
    多进程计算边类型分布
    :param data_list: 数据列表，每个元素为 torch_geometric.data.Data
    :param num_bond_types: 边的类型数量
    :param num_processes: 并行进程数
    :return: 归一化的边类型分布
    """
    print("Computing edge counts in parallel...")
    
    # 分割数据为多个子任务
    chunk_size = math.ceil(len(data_list) / num_processes)
    chunks = [data_list[i:i + chunk_size] for i in range(0, len(data_list), chunk_size)]

    # 创建队列用于存储结果
    result_queue = multiprocessing.Queue()
    processes = []

    # 创建并启动进程
    for chunk in chunks:
        p = multiprocessing.Process(target=worker_edge_counts, args=(chunk, num_bond_types, result_queue))
        processes.append(p)
        p.start()

    # 使用 tqdm 跟踪子任务完成情况
    results = []
    with tqdm(total=len(chunks), desc="Processing Chunks") as pbar:
        for _ in range(len(chunks)):
            results.append(result_queue.get())  # 从队列中获取结果
            pbar.update(1)

    # 等待所有进程结束
    for p in processes:
        p.join()

    # 合并所有结果
    total_counts = np.zeros(num_bond_types)
    for result in results:
        total_counts += result

    # 归一化分布
    total_counts = total_counts / total_counts.sum()
    print("Done.")
    return total_counts

def worker_charge_counts(data_chunk, num_classes, charges_dic, result_queue):
    """
    子进程函数：计算部分数据的电荷分布
    :param data_chunk: 子任务数据
    :param num_classes: 原子类型类别数量
    :param charges_dic: 电荷值字典
    :param result_queue: 用于通信的队列
    """
    local_counts = np.zeros((num_classes, len(charges_dic)))
    for data in data_chunk:
        for atom, charge in zip(data.x, data.charges):
            assert charge.item() in charges_dic, f"Unexpected charge value: {charge.item()}"
            local_counts[atom.item(), charges_dic[charge.item()]] += 1
    result_queue.put(local_counts)  # 将局部结果发送到队列

def charge_counts(data_list, num_classes, charges_dic, num_processes=8):
    """
    多进程计算电荷分布
    :param data_list: 数据列表，每个元素为 torch_geometric.data.Data
    :param num_classes: 原子类型类别数量
    :param charges_dic: 电荷值字典
    :param num_processes: 并行进程数
    :return: 归一化的电荷分布矩阵
    """
    print("Computing charge counts in parallel...")

    # 分割数据为多个子任务
    chunk_size = math.ceil(len(data_list) / num_processes)
    chunks = [data_list[i:i + chunk_size] for i in range(0, len(data_list), chunk_size)]

    # 创建队列用于存储结果
    result_queue = multiprocessing.Queue()
    processes = []

    # 创建并启动进程
    for chunk in chunks:
        p = multiprocessing.Process(target=worker_charge_counts, args=(chunk, num_classes, charges_dic, result_queue))
        processes.append(p)
        p.start()

    # 使用 tqdm 跟踪子任务完成情况
    results = []
    with tqdm(total=len(chunks), desc="Processing Chunks") as pbar:
        for _ in range(len(chunks)):
            results.append(result_queue.get())  # 从队列中获取结果
            pbar.update(1)

    # 等待所有进程结束
    for p in processes:
        p.join()

    # 合并所有结果
    total_counts = np.zeros((num_classes, len(charges_dic)))
    for result in results:
        total_counts += result

    # 归一化分布
    sums = np.sum(total_counts, axis=1, keepdims=True)
    sums[sums == 0] = 1  # 防止除以 0
    total_counts = total_counts / sums
    print("Done.")
    return total_counts

def worker_valency_count(data_chunk, atom_encoder, result_queue):
    """
    子进程函数：计算部分数据的原子价电子分布
    :param data_chunk: 子任务数据
    :param atom_encoder: 原子编码字典
    :param result_queue: 用于通信的队列
    """
    atom_decoder = {v: k for k, v in atom_encoder.items()}
    local_valencies = {atom_type: Counter() for atom_type in atom_encoder.keys()}

    for data in data_chunk:
        edge_attr = data.edge_attr.clone()
        edge_attr[edge_attr == 4] = 1.5
        bond_orders = edge_attr

        for atom in range(data.num_nodes):
            edges = bond_orders[data.edge_index[0] == atom]
            valency = edges.sum(dim=0)
            local_valencies[atom_decoder[data.x[atom].item()]][valency.item()] += 1

    result_queue.put(local_valencies)  # 将局部结果发送到队列

def merge_valencies(results, atom_encoder):
    """
    合并所有子任务的价电子计数结果
    :param results: 所有子任务的局部结果
    :param atom_encoder: 原子编码字典
    :return: 合并后的价电子分布
    """
    merged_valencies = {atom_type: Counter() for atom_type in atom_encoder.keys()}
    for local_valencies in results:
        for atom_type, counter in local_valencies.items():
            merged_valencies[atom_type].update(counter)
    return merged_valencies

def normalize_valencies(valencies):
    """
    归一化价电子分布
    :param valencies: 合并后的价电子分布
    :return: 归一化的价电子分布
    """
    for atom_type in valencies.keys():
        s = sum(valencies[atom_type].values())
        for valency, count in valencies[atom_type].items():
            valencies[atom_type][valency] = count / s if s > 0 else 0
    return valencies

def valency_count(data_list, atom_encoder, num_processes=8):
    """
    多进程计算原子价电子分布
    :param data_list: 数据列表，每个元素为 torch_geometric.data.Data
    :param atom_encoder: 原子编码字典
    :param num_processes: 并行进程数
    :return: 归一化的价电子分布
    """
    print("Computing valency counts in parallel...")

    # 分割数据为多个子任务
    chunk_size = (len(data_list) + num_processes - 1) // num_processes
    chunks = [data_list[i:i + chunk_size] for i in range(0, len(data_list), chunk_size)]

    # 创建队列用于存储结果
    result_queue = multiprocessing.Queue()
    processes = []

    # 创建并启动进程
    for chunk in chunks:
        p = multiprocessing.Process(target=worker_valency_count, args=(chunk, atom_encoder, result_queue))
        processes.append(p)
        p.start()

    # 使用 tqdm 跟踪子任务完成情况
    results = []
    with tqdm(total=len(chunks), desc="Processing Chunks") as pbar:
        for _ in range(len(chunks)):
            results.append(result_queue.get())  # 从队列中获取结果
            pbar.update(1)

    # 等待所有进程结束
    for p in processes:
        p.join()

    # 合并所有子任务结果
    merged_valencies = merge_valencies(results, atom_encoder)

    # 归一化分布
    normalized_valencies = normalize_valencies(merged_valencies)
    print("Done.")
    return normalized_valencies

def worker_bond_lengths_counts(data_chunk, num_bond_types, result_queue):
    """
    子进程函数：计算部分数据的键长分布
    :param data_chunk: 子任务数据
    :param num_bond_types: 键类型的总数
    :param result_queue: 用于通信的队列
    """
    local_bond_lengths = {bond_type: Counter() for bond_type in range(1, num_bond_types)}
    for data in tqdm(data_chunk):
        cdists = torch.cdist(data.pos.unsqueeze(0), data.pos.unsqueeze(0)).squeeze(0)
        bond_distances = cdists[data.edge_index[0], data.edge_index[1]]
        for bond_type in range(1, num_bond_types):
            bond_type_mask = data.edge_attr == bond_type
            distances_to_consider = bond_distances[bond_type_mask]
            distances_to_consider = torch.round(distances_to_consider, decimals=2)
            for d in distances_to_consider:
                local_bond_lengths[bond_type][d.item()] += 1
    result_queue.put(local_bond_lengths)  # 将局部结果发送到队列

def merge_bond_lengths(results, num_bond_types):
    """
    合并所有子任务的键长分布结果
    :param results: 所有子任务的局部结果
    :param num_bond_types: 键类型的总数
    :return: 合并后的键长分布
    """
    merged_bond_lengths = {bond_type: Counter() for bond_type in range(1, num_bond_types)}
    for local_bond_lengths in results:
        for bond_type in range(1, num_bond_types):
            merged_bond_lengths[bond_type].update(local_bond_lengths[bond_type])
    return merged_bond_lengths

def normalize_bond_lengths(bond_lengths, num_bond_types):
    """
    归一化键长分布
    :param bond_lengths: 合并后的键长分布
    :param num_bond_types: 键类型的总数
    :return: 归一化的键长分布
    """
    for bond_type in range(1, num_bond_types):
        s = sum(bond_lengths[bond_type].values())
        for d, count in bond_lengths[bond_type].items():
            bond_lengths[bond_type][d] = count / s if s > 0 else 0
    return bond_lengths

def bond_lengths_counts(data_list, num_bond_types=5, num_processes=36):
    """
    多进程计算键长分布
    :param data_list: 数据列表，每个元素为 torch_geometric.data.Data
    :param num_bond_types: 键类型的总数
    :param num_processes: 并行进程数
    :return: 归一化的键长分布
    """
    print("Computing bond lengths in parallel...")

    # 分割数据为多个子任务
    chunk_size = math.ceil(len(data_list) / num_processes)
    chunks = [data_list[i:i + chunk_size] for i in range(0, len(data_list), chunk_size)]

    # 创建队列用于存储结果
    result_queue = multiprocessing.Queue()
    processes = []

    # 创建并启动进程
    for chunk in chunks:
        p = multiprocessing.Process(target=worker_bond_lengths_counts, args=(chunk, num_bond_types, result_queue))
        processes.append(p)
        p.start()

    # 使用 tqdm 跟踪子任务完成情况
    results = []
    with tqdm(total=len(chunks), desc="Processing Chunks") as pbar:
        for _ in range(len(chunks)):
            results.append(result_queue.get())  # 从队列中获取结果
            pbar.update(1)

    # 等待所有进程结束
    for p in processes:
        p.join()

    # 合并所有子任务结果
    merged_bond_lengths = merge_bond_lengths(results, num_bond_types)

    # 归一化分布
    normalized_bond_lengths = normalize_bond_lengths(merged_bond_lengths, num_bond_types)
    print("Done.")
    return normalized_bond_lengths

def worker_bond_angles(data_chunk, atom_encoder, result_queue):
    """
    子进程函数：计算部分数据的键角分布
    :param data_chunk: 子任务数据
    :param atom_encoder: 原子编码字典
    :param result_queue: 用于通信的队列
    """
    num_atoms = len(atom_encoder.keys())
    local_bond_angles = np.zeros((num_atoms, 180 * 10 + 1))  # 初始化局部键角分布

    for data in tqdm(data_chunk):
        assert not torch.isnan(data.pos).any(), "Position tensor contains NaN values."
        for i in range(data.num_nodes):
            neighbors = data.edge_index[1][data.edge_index[0] == i]
            for j in neighbors:
                for k in neighbors:
                    if j == k:
                        continue
                    a = data.pos[j] - data.pos[i]
                    b = data.pos[k] - data.pos[i]
                    angle = torch.acos(torch.dot(a, b) / (torch.norm(a) * torch.norm(b) + 1e-6))
                    angle = angle * 180 / math.pi
                    bin_index = int(torch.round(angle, decimals=1) * 10)
                    local_bond_angles[data.x[i].item(), bin_index] += 1

    result_queue.put(local_bond_angles)  # 将局部结果发送到队列

def merge_bond_angles(results, num_atoms, bins):
    """
    合并所有子任务的键角分布结果
    :param results: 所有子任务的局部结果
    :param num_atoms: 原子类型总数
    :param bins: 每个键角分布的分箱数
    :return: 合并后的键角分布
    """
    merged_bond_angles = np.zeros((num_atoms, bins))
    for local_bond_angles in results:
        merged_bond_angles += local_bond_angles
    return merged_bond_angles

def normalize_bond_angles(bond_angles):
    """
    归一化键角分布
    :param bond_angles: 合并后的键角分布
    :return: 归一化的键角分布
    """
    s = bond_angles.sum(axis=1, keepdims=True)
    s[s == 0] = 1  # 防止除以 0
    bond_angles = bond_angles / s
    return bond_angles

def bond_angles(data_list, atom_encoder, num_processes=36):
    """
    多进程计算键角分布
    :param data_list: 数据列表，每个元素为 torch_geometric.data.Data
    :param atom_encoder: 原子编码字典
    :param num_processes: 并行进程数
    :return: 归一化的键角分布
    """
    print("Computing bond angles in parallel...")

    # 分割数据为多个子任务
    chunk_size = (len(data_list) + num_processes - 1) // num_processes
    chunks = [data_list[i:i + chunk_size] for i in range(0, len(data_list), chunk_size)]

    # 创建队列用于存储结果
    result_queue = multiprocessing.Queue()
    processes = []

    # 创建并启动进程
    for chunk in chunks:
        p = multiprocessing.Process(target=worker_bond_angles, args=(chunk, atom_encoder, result_queue))
        processes.append(p)
        p.start()

    # 使用 tqdm 跟踪子任务完成情况
    results = []
    num_bins = 180 * 10 + 1
    num_atoms = len(atom_encoder.keys())
    with tqdm(total=len(chunks), desc="Processing Chunks") as pbar:
        for _ in range(len(chunks)):
            results.append(result_queue.get())  # 从队列中获取结果
            pbar.update(1)

    # 等待所有进程结束
    for p in processes:
        p.join()

    # 合并所有子任务结果
    merged_bond_angles = merge_bond_angles(results, num_atoms, num_bins)

    # 归一化分布
    normalized_bond_angles = normalize_bond_angles(merged_bond_angles)
    print("Done.")
    return normalized_bond_angles