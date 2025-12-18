import numpy as np
from scipy.stats import wasserstein_distance
from .utils import calculate_mmd

def get_distributions(sample1, sample2, bins=100):
    """
    计算并返回两个样本集的分布
    """
    # 确保输入是numpy数组
    sample1 = np.asarray(sample1).flatten()
    sample2 = np.asarray(sample2).flatten()
    
    # 确定数据范围
    min_val = min(np.min(sample1), np.min(sample2))
    max_val = max(np.max(sample1), np.max(sample2))
    
    # 创建直方图（概率分布）
    hist1, _ = np.histogram(sample1, bins=bins, range=(min_val, max_val), density=True)
    hist2, _ = np.histogram(sample2, bins=bins, range=(min_val, max_val), density=True)
    
    hist1 = hist1/np.sum(hist1)
    hist2 = hist2/np.sum(hist2)
    return hist1, hist2

def js_divergence(src_dist, gen_dist):
    """
    从预先计算的分布中计算Jensen-Shannon Divergence (JS散度)
    """
    hist1, hist2 = src_dist, gen_dist
    
    # 添加微小值以避免log(0)
    hist1 = hist1 + 1e-10
    hist2 = hist2 + 1e-10
    
    # 归一化
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)
    
    # 计算M = (P + Q) / 2
    m = (hist1 + hist2) / 2
    
    # 计算JS散度
    js = 0.5 * np.sum(hist1 * np.log(hist1 / m)) + 0.5 * np.sum(hist2 * np.log(hist2 / m))
    
    return js

def eval_attr(src_x, gen_x, bins=10):
    
    mmd_list, emd_list, js_list = [], [], []
    
    for idx in range(src_x.shape[1]):
        hist_src, hist_gen = get_distributions(src_x[:,idx], gen_x[:,idx], bins=bins)
        mmd_list.append(calculate_mmd(hist_src.reshape(-1, 1), hist_gen.reshape(-1, 1)))
        emd_list.append(wasserstein_distance(hist_src, hist_gen))
        js_list.append(js_divergence(hist_src, hist_gen))
    
    return np.mean(mmd_list), np.mean(emd_list), np.mean(js_list)
