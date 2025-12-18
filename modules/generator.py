import torch
from torch import nn
from torch_geometric.data import Data
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

from .graphEncoder import GraphEncoder

class ZEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ZEncoder, self).__init__()
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            # nn.LayerNorm(hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            # nn.LayerNorm(hidden_size),
            nn.LeakyReLU()
        )

        # Encoder heads: Separate layers for mu and logvar
        self.encode_mu = nn.Linear(hidden_size, hidden_size)
        self.encode_logvar = nn.Linear(hidden_size, hidden_size)

   
        # initialize
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):    
        x = self.shared_encoder(x)
        # Get mu and logvar
        z_mu = self.encode_mu(x)
        z_logvar = self.encode_logvar(x)
        
        z_logvar = torch.clamp(z_logvar, min=-10, max=10)  # limit the range of z_logvar
        z_sgm = torch.exp(z_logvar * 0.5)
           
        # Reparameterization trick
        eps = torch.randn_like(z_sgm)
        z = eps * z_sgm + z_mu
        
        return z, z_mu, z_sgm

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, drop=0):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

        self.dropout1 = nn.Dropout(drop)
        self.dropout2 = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.dropout1(F.leaky_relu(self.layer1(x)))
        x = self.layer2(x)
        return x

class CVGAE(nn.Module):
    def __init__(self, num_feat, hidden_size, cond_feat_size, cond_struct_size, nodes_num, device):
        super(CVGAE, self).__init__()
        self.hidden_size = hidden_size
        self.nodes_num = nodes_num
        self.device = device
        
        self.graph_encoder = GraphEncoder(num_feat, hidden_size)
        self.zEncoder = ZEncoder(hidden_size + cond_feat_size, hidden_size) 
        
        self.decoder = MLP(hidden_size + cond_struct_size, hidden_size, nodes_num)
        self.attrDecoder = MLP(hidden_size + cond_feat_size, hidden_size, num_feat)
        self.struc_loss = nn.BCEWithLogitsLoss()
        self.attr_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, data: Data, cond_feat, cond_struct, y_adj): # batch_size
        num_seed = len(data.input_id)
        # y = data.y[:num_seed] # 如果标签编码过来，会导致dataloader过大，所幸先稀疏编码，训练哪些节点就转成dense
        graph_emb = self.graph_encoder(data.x, data.edge_index)
        graph_emb = graph_emb[:num_seed]
        
        if cond_feat is not None:
            z, z_mu, z_sgm = self.zEncoder(torch.cat([graph_emb, cond_feat], -1))
            attr_logits = self.attrDecoder(torch.cat([z, cond_feat], -1))
        else:
            z, z_mu, z_sgm = self.zEncoder(graph_emb)
            attr_logits = self.attrDecoder(z)
        
        if cond_struct is not None:
            struct_logits = self.decoder(torch.cat([z, cond_struct], -1))
        else:
            struct_logits = self.decoder(z)
        
        # loss
        recon_loss = self.struc_loss(struct_logits, y_adj)
        
        attr_loss = self.attr_loss(attr_logits, data.x[:num_seed])
    
        kld_loss = self._kld_gauss(
            p_mu=z_mu, 
            p_std=z_sgm, 
            q_mu=torch.zeros_like(z_mu),  # 先验分布的mu
            q_std=torch.ones_like(z_sgm)  # 先验分布的std
        )
        
        total_loss = recon_loss + attr_loss + kld_loss
        
        return total_loss, recon_loss, kld_loss, attr_loss
    
    def _kld_gauss(self, p_mu, p_std, q_mu, q_std):
        
        p_std = torch.clamp(p_std, min=1e-7)
        q_std = torch.clamp(q_std, min=1e-7)
        kld = (torch.log(q_std / p_std) + 
            (p_std**2 + (p_mu - q_mu)**2) / (2 * q_std**2) - 0.5)
        return kld.mean()
    
    def generate(self, data: Data, cond_feat, cond_struct, y_adj):
        
        # self.eval()  # 切换到评估模式
        with torch.no_grad():  # 关闭梯度计算
            num_seed = len(data.input_id)
            # 获取图嵌入
            graph_emb = self.graph_encoder(data.x, data.edge_index)
            graph_emb = graph_emb[:num_seed]
            
            # 编码获取分布参数
            if cond_feat is not None:
                z_samples, _, _ = self.zEncoder(torch.cat([graph_emb, cond_feat], -1))
                attr_logits = self.attrDecoder(torch.cat([z_samples, cond_feat], -1))
            else:
                z_samples, _, _ = self.zEncoder(graph_emb)
                attr_logits = self.attrDecoder(z_samples)
                
            if cond_struct is not None:
                struct_logits = self.decoder(torch.cat([z_samples, cond_struct], -1))
            else:
                struct_logits = self.decoder(z_samples)
            generated = torch.sigmoid(struct_logits)  # 转换为概率
            
            attr_gen = torch.sigmoid(attr_logits)
            
        # self.train()
        return self.parallel_logits_to_adj(generated, y_adj), attr_gen.cpu().numpy()
    
    def _parallel_logits_to_adj(self, gen_p, y_adj, SIZE):
        y_adj = y_adj.reshape(-1, self.nodes_num * SIZE)
        gen_p = gen_p.reshape(-1, self.nodes_num * SIZE)
        
        block_edges = y_adj.sum(dim=-1)
        
        gen_adj = torch.zeros_like(gen_p, device=self.device)
        
        _, indices = torch.topk(gen_p, int(torch.max(block_edges).item()), dim=1)
        for i, n in enumerate(block_edges):
            gen_adj[i, indices[i, :n]] = 1
        
        return gen_adj.reshape(-1, self.nodes_num)
    
    def parallel_logits_to_adj(self, gen_p, y_adj): # fast
        SIZE = 8
        y_adj = y_adj.long()
        
        extra_num = y_adj.shape[0] % SIZE
        if extra_num != 0:
            if y_adj.shape[0] // SIZE > 0:
                gen_adj_1 = self._parallel_logits_to_adj(gen_p[:extra_num, :], y_adj[:extra_num, :], extra_num)
                gen_adj_2 = self._parallel_logits_to_adj(gen_p[extra_num:, :], y_adj[extra_num:, :], SIZE)
                gen_adj = torch.cat((gen_adj_1, gen_adj_2), 0)
            else:
                gen_adj = self._parallel_logits_to_adj(gen_p, y_adj, extra_num)
        else:
            gen_adj = self._parallel_logits_to_adj(gen_p, y_adj, SIZE)
        
        non_zero_indices = np.nonzero(gen_adj.cpu().numpy())
        
        return non_zero_indices