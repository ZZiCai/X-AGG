
import random
import torch
import scipy.sparse as sp
import numpy as np
import logging
import os
from datetime import datetime
import yaml
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from torch_geometric.data import Data
from os.path import join, dirname, abspath

def case_insensitive(data):
    dir_names = os.listdir(join(dirname(__file__), 'data'))
    data = next(name for name in dir_names if data.lower() == name.lower()) # case insensitive
    return data
    
def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--update', nargs="*", help='# usage: --update <key1>=<value1> <key2>=<value2> ...')
    cmd_args = parser.parse_args()

    yaml_file = "config/config.yaml"
    
    with open(yaml_file) as file:
        config = yaml.safe_load(file)
    
    if cmd_args.update: # update config
        for item in cmd_args.update:
            key, value = item.split('=')
            val_type = type(config[key])
            if val_type == bool:
                value = True if value.lower() in ['y', 'yes', 'true'] else False
            config[key] = val_type(value)

    log_folder()
    
    args = {}
    
    # DATA_PATH = '/home/zhouzicai/GraphGeneration/GraphGenerator/data'
    DATA_PATH = './data'
    config['data'] = case_insensitive(config['data'])
    data = config['data']
    log_name = logging_conf(data, config['experiment_name'])
    args['model_save_path'] = os.path.join('models', log_name + '.pth')
    args['adj_save_path'] = os.path.join('graphs', log_name + '_adj.pkl')
    args['gen_adj_save_path'] = os.path.join('graphs', log_name + '_gen_adj.pkl')
    args['gen_attr_save_path'] = os.path.join('graphs', log_name + '_gen_attr.pkl')
    
    args['data_path'] = os.path.join(DATA_PATH, data)
    
    args.update(config) # update args
    
    if not os.path.exists('graphs'):
        os.makedirs('graphs')
    if not os.path.exists('models'):
        os.makedirs('models')
    
    log(log_name)
    
    return args

def random_seed(seed=2024):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def sp_to_tensor(sp_mat):
    if not sp.isspmatrix_coo(sp_mat):
        sp_mat = sp_mat.tocoo()
    
    sp_tensor = torch.sparse_coo_tensor(torch.LongTensor(np.stack([sp_mat.row, sp_mat.col])),
                                         torch.tensor(sp_mat.data),
                                         torch.Size([sp_mat.shape[0], sp_mat.shape[1]]))
    sp_tensor = sp_tensor.to_dense()
    return sp_tensor

########## log settings ##########
def log_folder():
    log_folder_name = 'logs'
    if not os.path.exists(log_folder_name):
        os.makedirs(log_folder_name)
        print(f"folder '{log_folder_name}' is created.")

def logging_conf(data, experiment_name):
    log_name = data + '-' + experiment_name + '-' + datetime.now().strftime("%m-%d %H:%M")
    log_file = os.path.join('logs/', log_name + '.log')
    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s:%(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename=log_file,
                    filemode='w')
    return log_name

def log(msg):
    """For uniform printing in the repository.

    Args:
        msg (str): message to be printed
    """    
    logging.info(msg)
    print(msg)

def logPeakGPUMem(device='cuda:0'):
    max_allocated = torch.cuda.max_memory_allocated(device=device)
    max_reserved = torch.cuda.max_memory_reserved(device=device)
    
    log(f"Peak GPU Memory Cached    : {max_reserved / (1024 ** 3):.2f} GB")
    log(f"Peak GPU Memory Allocated : {max_allocated / (1024 ** 3):.2f} GB")
    log(f"Peak GPU Memory Reserved  : {max_reserved / (1024 ** 3):.2f} GB")

def jaccard_at_k(list1, list2, k):
    """
    计算两个有序列表的 Jaccard Index @K。

    Args:
        list1 (list): 第一个升序列表。
        list2 (list): 第二个升序列表。
        k (int): 指定要比较的前 K 个元素。

    Returns:
        float: Jaccard Index @K 的值，范围在 0 到 1 之间。
    """
    # 确保 k 不超过列表长度
    k = min(k, len(list1), len(list2))
    if k == 0:
        return 0.0

    # 1. 从列表中取出前 K 个元素，并转换为集合
    set1 = set(list1[:k])
    set2 = set(list2[:k])
    
    # 2. 计算交集和并集
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    # 3. 计算 Jaccard Index
    if not union:
        return 0.0
    
    return len(intersection) / len(union)

def from_numpy_build_data(row, col, attr, y_tensor):
    x_tensor = torch.from_numpy(attr).float()
    edge_index_numpy = np.vstack((row, col))
    edge_index_tensor = torch.from_numpy(edge_index_numpy).long()
    data = Data(x=x_tensor, edge_index=edge_index_tensor, y=y_tensor)
    
    return data
    
    