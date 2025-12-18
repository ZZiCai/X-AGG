# Temporal Graph Evaluator

import numpy as np
from concurrent.futures import ProcessPoolExecutor
import time
from collections import defaultdict
import networkx as nx
from .metrics import *
from .utils import *

class CompEvaluator:
    
    def __init__(self, mmd_beta=2.0):
        self.eval_time_len = None
        self.is_ratio = True
        
        self.mmd_beta=mmd_beta 
        self.stats_map={
                        'deg_dist': deg_dist,
                        'clus_dist': clus_dist,
                        'wedge_count': wedge_count,
                        'triangle_count': triangle_count, 
                        'claw_count': claw_count,
                        'n_components': n_component,
                        'lcc_size': LCC,
                        'power_law': power_law_exp,
                        'gini': gini,
                        # 'node_div_dist': node_div_dist, # 不要出现
                        'global_cluster_coef': clustering_coefficient,
                        # 'mean_bc': mean_betweeness_centrality,  # 不要出现
                        # 'mean_cc': mean_closeness_centrality  # 不要出现
                       }
        
        self.hists_range={'deg_dist_in':(0,100),
                          'deg_dist_out':(0,100),
                          'deg_dist':(0,100),
                          'clus_dist':(0.0,1.0),
                          'node_div_dist':(-1.0,1.0)
                         }
    
    def error_func(self, org_graph, generated_graph):
        
        if self.is_ratio==True:
            metric = np.divide(np.abs(org_graph - generated_graph), np.abs(org_graph))
        else:
            metric = np.abs(org_graph - generated_graph)

        return metric
    
    def error_func_dist_mmd(self, org_dist, gen_dist):
        
        metric=calculate_mmd(org_dist.reshape(-1,1),gen_dist.reshape(-1,1),beta=self.mmd_beta)
        return metric
    
    def cal_stat(self, A_csr, stat_func, flow=None):
        
        if flow is None:
            return stat_func(A_csr)
        else:
            return stat_func(A_csr, flow)
    
    def cal_stat_dist(self, A_csr, stat_func, hist_range, flow=None):
        
        if flow is None:
            dist = stat_func(A_csr)
        else:
            dist = stat_func(A_csr, flow)
        
        hist, _ = np.histogram(np.array(dist), bins=100, range=hist_range, density=False)
        
        return hist
    
    def cal_func(self,stat):
        flow = None
        if stat.endswith('in'):
            flow = 'in'
        elif stat.endswith('out'):
            flow = 'out'

        if stat in self.hists_range.keys():
            return self.error_func_dist_mmd(self.cal_stat_dist(self.A_src, self.stats_map[stat], self.hists_range[stat], flow),
                                                    self.cal_stat_dist(self.A_gen, self.stats_map[stat], self.hists_range[stat], flow))
        else:
            return self.error_func(self.cal_stat(self.A_src, self.stats_map[stat], flow),
                                            self.cal_stat(self.A_gen, self.stats_map[stat], flow))
    # output the evaluation results for all statistics
    def comp_graph_stats(self,A_src, A_gen, is_ratio=True, stats_list=None):
        # transform the input graph format
        A_src = trans_graph_format(A_src)
        A_gen = trans_graph_format(A_gen)

        
        # make the graph undirected
        A_src = A_src + A_src.T
        A_src[A_src > 1] = 1
        self.A_src = A_src
        
        A_gen = A_gen + A_gen.T
        A_gen[A_gen > 1] = 1
        self.A_gen = A_gen
        
        self.is_ratio=is_ratio
            
        res_dict=defaultdict(float) # result dictionary
        
        metric_list=stats_list if stats_list is not None else list(self.stats_map.keys())

        time_start = time.time()
        
        # with ProcessPoolExecutor() as executor:
        #     results = executor.map(self.cal_func, metric_list)
        #     res_dict = {stat: result for stat, result in zip(metric_list, results)}
        res_dict = {stat: self.cal_func(stat) for stat in metric_list}
        time_end = time.time()
        print("Time taken:", time_end - time_start, "seconds")
            
        return res_dict
