import os
import shutil

import torch
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np
from tqdm import tqdm

# Auxiliary model
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_size, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_size) 
        self.conv2 = GCNConv(hidden_size, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1) #! Currently, it is a multi-classification task. Different tasks need to be modified here

class Preprocessor:
    def __init__(self, data: Data, num_features, hidden_size, num_classes, data_path, lr=0.01, weight_decay=5e-4, epochs=200):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.data = data.to(self.device)
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.data_path = data_path
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.num_nodes = self.data.num_nodes
        self.temp_dir = os.path.join(self.data_path, "temp_matrices")
        
    def get_explainer(self):
        data = self.data
        # 初始化模型以及优化器
        
        model = GCN(self.num_features, self.hidden_size, self.num_classes).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        # 训练模型
        for _ in range(self.epochs):
            #! If the dataset is too large, it may be necessary to train in batches
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
        
        # 生成解释
        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(),
            explanation_type='model',
            node_mask_type='common_attributes', # attributes
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification', # binary_classification, multiclass_classification, regression
                task_level='node', # node, edge, graph
                return_type='log_probs', # raw, probs, log_probs
            ),
        )
        
        return explainer
    
    def combine_tmp_file(self, temp_file_index):
        # 读取所有临时文件并合并
        all_feat = []
        all_struct = []
        for i in range(temp_file_index):
            # 按顺序读取临时文件
            feat_temp = np.load(os.path.join(self.temp_dir, f"feat_temp_{i}.npy"), allow_pickle=True)
            struct_temp = np.load(os.path.join(self.temp_dir, f"struct_temp_{i}.npy"), allow_pickle=True)
            
            all_feat.extend(feat_temp)
            all_struct.extend(struct_temp)
        
        # 转换为numpy数组
        feat_matrix = np.array(all_feat)
        struct_matrix = np.array(all_struct)
        
        # 保存矩阵
        
        np.save(os.path.join(self.data_path, 'cond_feat.npy'), feat_matrix)
        np.save(os.path.join(self.data_path, 'cond_struct.npy'), struct_matrix)
    
    def preprocess(self):
        if os.path.exists(os.path.join(self.data_path, 'cond_feat.npy')) and \
            os.path.exists(os.path.join(self.data_path, 'cond_struct.npy')):
            return
        
        data = self.data
        explainer = self.get_explainer()
        
        #! 逐个节点生成解释
        # 创建临时文件夹
        os.makedirs(self.temp_dir, exist_ok=True)
        temp_file_index = 0  # 临时文件计数器
        
        feat_matrix = []
        struct_matrix = []
        for node_index in tqdm(range(self.num_nodes)): 
            explanation = explainer(data.x, data.edge_index, index=node_index)
            
            # feat_matrix
            node_mask = explanation.node_mask
            score = node_mask.sum(dim=0) #  num_nodes
            feat_matrix.append(score.cpu().numpy())
            
            # struct_matrix
            edge_weight = explanation.edge_mask.cpu()
            edge_index = explanation.edge_index.cpu()
            if edge_weight is not None:  # Normalize edge weights.
                edge_weight = edge_weight - edge_weight.min()
                edge_weight = edge_weight / edge_weight.max()

            # Discard any edges with zero edge weight:
                mask = edge_weight > 1e-7
                edge_index = edge_index[:, mask]
            # 把index里的节点unique化，然后对应成矩阵的一行，赋值为1
            edge_index = torch.unique(edge_index, sorted=True)
            matrix = torch.zeros(self.num_nodes, dtype=torch.float)
            matrix[edge_index] = 1.0
            struct_matrix.append(matrix.cpu().numpy())
            
            if (node_index + 1) % 500 == 0 or node_index == self.num_nodes - 1:
                # 保存临时文件
                np.save(os.path.join(self.temp_dir, f"feat_temp_{temp_file_index}.npy"), feat_matrix)
                np.save(os.path.join(self.temp_dir, f"struct_temp_{temp_file_index}.npy"), struct_matrix)
                
                # 清空列表释放内存
                feat_matrix = []
                struct_matrix = []
                temp_file_index += 1
        self.combine_tmp_file(temp_file_index)
        
        # 清理临时文件和文件夹
        shutil.rmtree(self.temp_dir)
    
if __name__ == '__main__':
    dataset = Planetoid(root='data', name='Cora')
    data = dataset[0]
    preprocessor = Preprocessor(data, dataset.num_features, 16, dataset.num_classes, '../data/Cora')
    preprocessor.preprocess()