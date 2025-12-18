from torch_geometric.nn import GATConv, GATv2Conv
import torch.nn as nn

class GraphEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super(GraphEncoder, self).__init__()
        heads = 4
        self.conv1 = GATConv(in_channels, hidden_dim//heads, heads=heads, dropout=0, residual=False)
        
        self.conv2 = GATConv(hidden_dim, hidden_dim//heads, heads=heads, dropout=0, residual=True)
        
        # self.conv3 = GATConv(hidden_dim, hidden_dim//heads, heads=heads, dropout=0,residual=True)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        # x = self.conv3(x, edge_index)
        return x