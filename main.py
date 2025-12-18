import torch
from torch.nn.utils import clip_grad_norm_
import yaml
import time
import json
import pickle as pkl
import pandas as pd
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, AttributedGraphDataset
from modules.generator import CVGAE
from modules.preprocessor import Preprocessor
from torch_geometric.loader import NeighborLoader
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from utils import *
import sys
from tqdm import tqdm
from graph_metrics import compute_statistics, CompEvaluator, eval_attr

class NetTrainer:
    def __init__(self, args):
        self.args = args
        self.dataloader = None
        self.y_adj = None
        
        device = args['device']
        
        log(json.dumps(args, indent=4))
        
        log('Loading data...')
        
        # start = time.time()
        self.build_dataloader()
        # end = time.time()
        # log('build_dataLoader time cost: {:.4f}s'.format(end-start))
        ########## ablation study ##########
        ablation = args['ablation']
        self.ablation = ablation
        
        # 'cond_feat_size' 只有在 'no-M_X' 或 'no-M' 时才为 0
        cond_feat_size = 0 if ablation in ('no-M_X', 'no-M') else self.cond_feat.shape[1]
        # 'cond_struct_size' 只有在 'no-M_A' 或 'no-M' 时才为 0
        cond_struct_size = 0 if ablation in ('no-M_A', 'no-M') else self.cond_struct.shape[1]
            
        self.model = CVGAE(self.num_feat, args['hidden_size'], cond_feat_size, cond_struct_size, self.num_nodes, device).to(device)
        self.evaluator = CompEvaluator(mmd_beta=1.0)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args['lr'])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args['num_epoch'], eta_min=args['lr']/50.0)
                    
        log('Data loaded.')
        
    def build_dataloader(self):
        data_name = self.args['data']
        if data_name == "Cora":
            dataset = Planetoid(root='data', name='Cora')
        elif data_name == "Pubmed":
            dataset = Planetoid(root='data', name='Pubmed')
        elif data_name == "flickr":
            dataset = AttributedGraphDataset(root='data', name='Flickr')
        elif data_name == "CS":
            dataset = Coauthor(root='data', name='CS')
        elif data_name == "Physics":
            dataset = Coauthor(root='data', name='Physics')
        elif data_name == "Computers":
            dataset = Amazon(root='data', name='Computers')
        elif data_name == "Photo":
            dataset = Amazon(root='data', name='Photo')
        else:
            pass #TODO
        
        data = dataset[0]
        if data_name == "flickr":
            data.x = data.x.to_dense()
            data.x = data.x.to(torch.float32)
        self.original_data = data # To test explainability alignment.
        
        self.num_feat = dataset.num_features
        self.num_classes = dataset.num_classes
        
        preprocessor = Preprocessor(data, self.num_feat, 64, self.num_classes, self.args['data_path'])
        preprocessor.preprocess() # 预处理，得到cond_matrix，并存储为文件
        
        # get y_adj
        edge_index = data.edge_index.cpu().numpy()  # [2, E]
    
        if hasattr(data, 'num_nodes') and data.num_nodes is not None:
            num_nodes = data.num_nodes
        else:
            num_nodes = data.x.size(0)
        
        self.num_nodes = num_nodes
        
        edge_weight = np.ones(edge_index.shape[1], dtype=np.float32)
        adj_coo = sp.coo_matrix(
            (edge_weight, (edge_index[0], edge_index[1])),
            shape=(num_nodes, num_nodes)
        )
        adj_csr = adj_coo.tocsr()
        self.y_adj = adj_csr
        
        self.src_x = data.x.cpu().numpy()
        
        # cond_feat 和 cond_struct
        data_path = self.args['data_path']
        self.cond_feat = np.load(os.path.join(data_path, "cond_feat.npy"))
        self.cond_struct = np.load(os.path.join(data_path, "cond_struct.npy"))
        
        self.dataloader = NeighborLoader(
            data,
            num_neighbors=self.args['num_neighbors'],
            batch_size=self.args['batch_size'],
        )
        
        with open(self.args['adj_save_path'], 'wb') as f:
            pkl.dump(self.y_adj, f)
        
    def train(self):
        log('Start training...')
        best_res = {}
        best_align_res = {}
        
        total_train_time = 0
        for epoch in range(self.args['num_epoch']):
            
            avg_struc_loss, avg_kld_loss, avg_attr_loss = 0, 0, 0
            cnt = 0
            
            time_start = time.time()
            for data in self.dataloader:
                self.optimizer.zero_grad()
                
                num_seed = len(data.input_id)
                global_index = data.n_id[:num_seed].cpu().numpy()
                y_adj = sp_to_tensor(self.y_adj[global_index, :]).to(self.args['device'])
                
                cond_feat = None if self.ablation in ('no-M_X', 'no-M') else torch.from_numpy(self.cond_feat[global_index, :]).to(self.args['device'])
                
                cond_struct = None if self.ablation in ('no-M_A', 'no-M') else torch.from_numpy(self.cond_struct[global_index, :]).to(self.args['device'])
                
                loss_step, struc_loss, kld_loss, attr_loss = self.model(data.to(self.args['device']), cond_feat, cond_struct, y_adj)
                loss_step.backward()
                self.optimizer.step()
                
                avg_struc_loss += struc_loss.item()
                avg_kld_loss += kld_loss.item()
                avg_attr_loss += attr_loss.item()
                
                cnt += 1
                if cnt % 10 == 0:
                    print('Epoch: {}, Batch: {}, Struc Loss: {:.4f}, KLD Loss: {:.4f}, Attr_loss: {:.4f}'.format(
                        epoch+1, cnt, struc_loss.item(), kld_loss.item(), attr_loss.item()
                    ))
            one_epoch_time = time.time() - time_start
            total_train_time += one_epoch_time
            
            self.scheduler.step()
            
            avg_struc_loss /= cnt
            avg_kld_loss /= cnt
            avg_attr_loss /=cnt
            
            loss_all = avg_struc_loss + avg_kld_loss + avg_attr_loss
            
            log('Epoch: {}, Struc Loss: {:.4f}, KLD Loss: {:.4f}, Attr_loss: {:.4f}, Loss: {:.4f}, training time: {:.4f}'.format(
                epoch+1, avg_struc_loss, avg_kld_loss, avg_attr_loss, loss_all, one_epoch_time
            ))
            if (epoch + 1) % self.args['save_model_intervals'] == 0:
                torch.save(self.model.state_dict(), self.args['model_save_path'])
            if (epoch + 1) % self.args['eval_per_epochs'] == 0:
                log('Start testing...')
                res, align_res = self.evaluate()
                for k, v in res.items():
                    if k not in best_res or v < best_res[k]:
                        best_res[k] = v
                for k, v in align_res.items():
                    if k not in best_align_res or v > best_align_res[k]:
                        best_align_res[k] = v
                log('Test result, Epoch: {}, {}, {}'.format(epoch+1, json.dumps(res, indent=4), json.dumps(align_res, indent=4)))
        
        log('Total training time: {:.4f} s'.format(total_train_time))
        
        best_res.update(best_align_res)
        df_avg = pd.DataFrame([best_res])
        log("Best result for copy:\n" + df_avg.to_csv(sep='\t', index=False, float_format='%.4f'))
        log('Best result: {}'.format(json.dumps(best_res, indent=4)))
    
    def eval_explainability_alignment(self, gen_data):
        data = self.original_data
        data = data.to(self.args['device'])
        gen_data = gen_data.to(self.args['device'])
        preprocessor = Preprocessor(data, self.num_feat, 64, self.num_classes, self.args['data_path'])
        explainer = preprocessor.get_explainer() # get explainer
        
        node_index = np.random.choice(self.num_nodes, 20, replace=False)
        
        spearman_corr_list = []
        Jat5_list = []
        Jat10_list = []
        Jat20_list = []
        
        edge_Jat5_list = []
        edge_Jat10_list = []
        edge_Jat20_list = []
        for index in tqdm(node_index):
            # attr-origin_data
            ori_explanation = explainer(data.x, data.edge_index, index=index)
            ori_node_mask = ori_explanation.node_mask
            ori_attr_importance = ori_node_mask.sum(dim=0)
            ori_attr_importance = ori_attr_importance.cpu().numpy()
            
            # attr-gen_data
            gen_explanation = explainer(gen_data.x, gen_data.edge_index, index=index)
            gen_node_mask = gen_explanation.node_mask
            gen_attr_importance = gen_node_mask.sum(dim=0)
            gen_attr_importance = gen_attr_importance.cpu().numpy()
            
            spearman_corr, _ = spearmanr(ori_attr_importance, gen_attr_importance)
            spearman_corr_list.append(spearman_corr)
            ori_sorted_indices = np.argsort(ori_attr_importance)[::-1]
            gen_sorted_indices = np.argsort(gen_attr_importance)[::-1]
            Jat5_list.append(jaccard_at_k(ori_sorted_indices, gen_sorted_indices, 5))
            Jat10_list.append(jaccard_at_k(ori_sorted_indices, gen_sorted_indices, 10))
            Jat20_list.append(jaccard_at_k(ori_sorted_indices, gen_sorted_indices, 20))
            
            # edge-origin_data
            edge_weight = ori_explanation.edge_mask.cpu().numpy()
            edge_index = ori_explanation.edge_index.cpu().numpy()
            ori_edge_list = []
            ori_sorted_indices = np.argsort(edge_weight)[::-1]
            for idx in ori_sorted_indices[:20]:
                ori_edge_list.append((edge_index[0,idx], edge_index[1, idx]))
            # edge-gen_data
            edge_weight = gen_explanation.edge_mask.cpu().numpy()
            edge_index = gen_explanation.edge_index.cpu().numpy()
            gen_edge_list = []
            gen_sorted_indices = np.argsort(edge_weight)[::-1]
            for idx in gen_sorted_indices[:20]:
                gen_edge_list.append((edge_index[0,idx], edge_index[1, idx]))
            
            edge_Jat5_list.append(jaccard_at_k(ori_edge_list, gen_edge_list, 5))
            edge_Jat10_list.append(jaccard_at_k(ori_edge_list, gen_edge_list, 10))
            edge_Jat20_list.append(jaccard_at_k(ori_edge_list, gen_edge_list, 20))
            
        
        spearman_corr = np.mean(spearman_corr_list)
        Jat5 = np.mean(Jat5_list)
        Jat10 = np.mean(Jat10_list)
        Jat20 = np.mean(Jat20_list)
        
        e_Jat5 = np.mean(edge_Jat5_list)
        e_Jat10 = np.mean(edge_Jat10_list)
        e_Jat20 = np.mean(edge_Jat20_list)
        # log('Attribute Importance Spearman Correlation: {:.4f}'.format(spearman_corr))
        # log('Attribute Importance Jaccard@5: {:.4f}, Jaccard@10: {:.4f}, Jaccard@20: {:.4f}'.format(Jat5, Jat10, Jat20))
        return {
            'attr_spearman_corr': spearman_corr,
            'attr_j@5': Jat5,
            'attr_j@10': Jat10,
            'attr_j@20': Jat20,
            'edge_j@5': e_Jat5,
            'edge_j@10': e_Jat10,
            'edge_j@20': e_Jat20
        }
            
        
    def evaluate(self): 
        self.model.eval()
        
        gen_row = []
        gen_col = []
        gen_attr = []
        
        start = time.time()
        for data in self.dataloader:
            num_seed = len(data.input_id)
            global_index = data.n_id[:num_seed].cpu().numpy()
            y_adj = sp_to_tensor(self.y_adj[global_index, :]).to(self.args['device'])
            
            # 'cond_feat' 在 'no-M_X' 或 'no-M' 时被消融 (设为 None)
            cond_feat = None if self.ablation in ('no-M_X', 'no-M') else torch.from_numpy(self.cond_feat[global_index, :]).to(self.args['device'])
            # 'cond_struct' 在 'no-M_A' 或 'no-M' 时被消融 (设为 None)
            cond_struct = None if self.ablation in ('no-M_A', 'no-M') else torch.from_numpy(self.cond_struct[global_index, :]).to(self.args['device'])
            
            non_zero_indices, attr = self.model.generate(data.to(self.args['device']), cond_feat, cond_struct, y_adj)
            gen_row.extend(global_index[non_zero_indices[0]])
            gen_col.extend(non_zero_indices[1])
            
            gen_attr.append(attr)
            
            
        gen_mat = sp.coo_matrix((np.ones(len(gen_row)), (gen_row, gen_col)), shape=(self.num_nodes, self.num_nodes))
        gen_mat = gen_mat.tocsr()
        with open(self.args['gen_adj_save_path'], 'wb') as f:
            pkl.dump(gen_mat, f)
        
        gen_attr = np.concatenate(gen_attr, axis=0)
        
        log('Generation time cost: {:.4f} s'.format(time.time()-start))
        
        self.model.train()
        
        with open(self.args['gen_attr_save_path'], 'wb') as f:
            pkl.dump(gen_attr, f)
        
        res = self.evaluator.comp_graph_stats(self.y_adj, gen_mat)
        
        res_mmd, res_emd, res_js = eval_attr(self.src_x, gen_attr)
        res_attr = {
            'attr_mmd': res_mmd,
            'attr_emd': res_emd,
            'attr_js': res_js
        }
        res.update(res_attr)
        
        ## 评估可解释性对齐
        gen_data = from_numpy_build_data(gen_row, gen_col, gen_attr, y_tensor=self.original_data.y)
        res_alignment = self.eval_explainability_alignment(gen_data)
        return res, res_alignment
                
def main(args):
    random_seed(args["seed"])
    
    trainer = NetTrainer(args)
    
    trainer.train()
    logPeakGPUMem(args['device'])

if __name__ == '__main__':
    main(parse_args())