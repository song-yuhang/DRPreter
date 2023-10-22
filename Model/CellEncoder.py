import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv, max_pool
from torch_geometric.nn import global_add_pool, global_max_pool

# 图卷积网络
class CellEncoder(torch.nn.Module):
    def __init__(self, num_feature, num_genes, layer_cell, dim_cell):
        super().__init__()
        self.num_feature = num_feature
        self.layer_cell = layer_cell
        self.dim_cell = dim_cell
        self.final_node = num_genes
        self.convs_cell = torch.nn.ModuleList()

        for i in range(self.layer_cell):
            if i:
                conv = GATConv(self.dim_cell, self.dim_cell)
            else:
                conv = GATConv(self.num_feature, self.dim_cell)

            self.convs_cell.append(conv)

  
    def forward(self, cell):
        for i in range(self.layer_cell):
            cell.x = F.relu(self.convs_cell[i](cell.x, cell.edge_index))

        node_representation = cell.x.reshape(-1, self.final_node * self.dim_cell) # val_mse on 1st epohc ~=  1.6
        # node_representation = global_add_pool(cell.x, cell.batch) # performance down: val_mse on 1st epoch  ~= 2.5

        return node_representation
    
     
    def grad_cam(self, cell):
        # 遍历细胞层的每一层，通过卷积操作更新节点的特征
        for i in range(self.layer_cell):
            # 对当前层应用图卷积，并通过ReLU激活函数进行非线性变换
            cell.x = F.relu(self.convs_cell[i](cell.x, cell.edge_index))
            # 在第0层（第一层）保存节点的特征，以便于后续的梯度访问
            if i == 0:
                cell_node = cell.x  # 保存第0层的节点特征
                cell_node.retain_grad()  # 保留节点特征的梯度，以便于后续的梯度计算

        # 将最终层的节点特征重新塑形为二维张量，其中每行代表一个节点的特征向量
        # self.final_node可能表示最终层的节点数量，self.dim_cell可能表示每个节点特征的维度
        node_representation = cell.x.reshape(-1, self.final_node * self.dim_cell)

        # 返回第0层的节点特征和最终层的节点特征表示
        return cell_node, node_representation
