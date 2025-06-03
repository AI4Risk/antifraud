import sys
sys.path.append('../')
from utils.MyUtils import color_print, save_pic_iterly

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, ClusterData, ClusterLoader
from torch_geometric.utils import add_self_loops, degree

import GCL.losses as L
import GCL.augmentors as A
from GCL.models import DualBranchContrast

import os
from tqdm import tqdm

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class HighPassFilterLayer(MessagePassing):
    def __init__(self):
        # MessagePassing base class requires an aggregation method
        super().__init__(aggr='mean')

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # Degree of nodes
        deg = degree(edge_index[0], x.size(0), dtype=x.dtype)
        # Mean aggregation
        out = self.propagate(edge_index, x=x, size=None)
        # Subtract the mean neighbor feature from the node feature to act as a high pass filter
        return x - out / deg.view(-1, 1)

class SupGCL_GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super().__init__()
        self.activation = activation()  # 激活函数实例化将在外部完成，这里直接使用
        self.layers = torch.nn.ModuleList()
        # 添加第一个线性层
        self.layers.append(nn.Linear(input_dim, hidden_dim))

        # 为每个后续层添加更多的线性层
        for _ in range(num_layers - 1):
            self.layers.append(HighPassFilterLayer())
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for layer in self.layers:
            # 检查层类型以确定是否使用边缘索引和权重
            if isinstance(layer, HighPassFilterLayer):
                # 假设HighPassFilterLayer需要边缘索引和边缘权重
                z = layer(z, edge_index)
                None
            else:
                # 对于线性层，仅应用线性变换
                z = layer(z)

            # 检查层类型后决定是否应用激活函数
            if not isinstance(layer, HighPassFilterLayer):
                z = self.activation(z)
        
        return z

class SupGCL_Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, proj_dim):
        super().__init__()
        self.encoder = encoder
        self.augmentor = augmentor
        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        z = self.encoder(x, edge_index, edge_weight)
        z1 = self.encoder(x1, edge_index1, edge_weight1)
        z2 = self.encoder(x2, edge_index2, edge_weight2)
        return z, z1, z2

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

class SupGCL():
    def __init__(self, global_args, graph_pyg_selected, nodes_per_subgraph, device, num_train_part=20, batch_size=5):
        self.nodes_per_subgraph=nodes_per_subgraph
        self.device=device
        self.data=graph_pyg_selected
        self.num_train_part=num_train_part
        self.batch_size=batch_size
        self.global_args=global_args

        color_print(f'!!!!! Start clustering for batch in SupGCL')
        self.cluster_data=ClusterData(self.data, num_parts=self.num_train_part, recursive=False)
        self.subgraph_cluster_loader = ClusterLoader(self.cluster_data, batch_size=self.batch_size, shuffle=True, num_workers=12)
        color_print(f'!!!!! Finish clustering for batch in SupGCL')

        self.aug1 = A.Compose([A.FeatureDropout(pf=0.3), A.FeatureMasking(pf=0.3)])
        self.aug2 = A.Compose([A.FeatureDropout(pf=0.3), A.FeatureMasking(pf=0.3)])
        # aug2 = A.Compose([A.FeatureDropout(pf=0.1), A.Identity()])

        self.gconv = SupGCL_GConv(input_dim=self.data.x.shape[1], hidden_dim=self.nodes_per_subgraph, activation=torch.nn.ReLU, num_layers=3).to(self.device)
        self.encoder_model = SupGCL_Encoder(encoder=self.gconv, augmentor=(self.aug1, self.aug2), hidden_dim=self.nodes_per_subgraph, proj_dim=128).to(self.device)
        self.contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=True).to(self.device)

        self.optimizer = Adam(self.encoder_model.parameters(), lr=0.001)

    def SupGCL_train(self, batch_data):
        self.encoder_model.train()
        self.optimizer.zero_grad()
        z, z1, z2 = self.encoder_model(batch_data.x, batch_data.edge_index, batch_data.edge_attr)
        h1, h2 = [self.encoder_model.project(x) for x in [z1, z2]]

        # compute extra pos and neg masks for semi-supervised learning
        extra_pos_mask = torch.eq(batch_data.y, batch_data.y.unsqueeze(dim=1))
        # construct extra supervision signals for only training samples
        extra_pos_mask[~batch_data.train_mask][:, ~batch_data.train_mask] = False
        extra_pos_mask.fill_diagonal_(False)
        # pos_mask: [N, 2N] for both inter-view and intra-view samples
        extra_pos_mask = torch.cat([extra_pos_mask, extra_pos_mask], dim=1).to(self.device)
        # fill interview positives only; pos_mask for intraview samples should have zeros in diagonal
        extra_pos_mask.fill_diagonal_(True)

        extra_neg_mask = torch.ne(batch_data.y, batch_data.y.unsqueeze(dim=1))
        extra_neg_mask[~batch_data.train_mask][:, ~batch_data.train_mask] = True
        extra_neg_mask.fill_diagonal_(False)
        extra_neg_mask = torch.cat([extra_neg_mask, extra_neg_mask], dim=1).to(self.device)

        loss = self.contrast_model(h1=h1, h2=h2, extra_pos_mask=extra_pos_mask, extra_neg_mask=extra_neg_mask)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def train(self, epochs=50):
        color_print(f'!!!!! SupGCL start training')
        time=range(1, epochs+1)
        with tqdm(time, desc='(T)') as pbar:
            for epoch in pbar:
                # 迭代数据加载器
                for batch in self.subgraph_cluster_loader:
                    # 可以将节点索引和邻接表传递给您的模型
                    batch=batch.to(self.device)
                    # print(batch.x.shape)
                    loss = self.SupGCL_train(batch_data=batch)
                    # 更新进度条和损失信息
                    pbar.set_postfix({'loss': '{:.6f}'.format(loss),'time': epoch})
                    # pbar.update()
                    batch=batch.to('cpu')
        
        color_print(f'!!!!! SupGCL finish training')

    def visualize(self):
        tmp_data=self.data.to(self.device)

        self.encoder_model.eval()
        with torch.no_grad():
            z, _, _ = self.encoder_model(tmp_data.x, tmp_data.edge_index, tmp_data.edge_attr)
        
        color_print(f'!!!!! SupGCL start visualizing')

        # 运行t-SNE算法，降维到2D
        tsne = TSNE(n_components=2, random_state=42)
        z_reduced = tsne.fit_transform(z.cpu().numpy())  # 确保数据在CPU上并转换为NumPy数组

        # 对于Cora数据集，还可以获取标签以用于颜色编码
        labels = tmp_data.y.cpu().numpy()

        # 创建一个散点图，每种类别用不同颜色表示
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(z_reduced[:, 0], z_reduced[:, 1], c=labels, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter)
        plt.title('t-SNE visualization of node embeddings')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.show()

        pic_name=f'./tmp/{self.global_args.dataset}_{self.global_args.SupGCL_epochs}epochs_SupGCLVisualization'
        info='SupGCL visualization'
        save_pic_iterly(pic_name=pic_name, postfix='png', info=info)

    def project(self, graph_pyg):
        self.encoder_model.eval()
        tmp_data=graph_pyg.to(self.device)
        with torch.no_grad():
            z, z1, z2 = self.encoder_model(tmp_data.x, tmp_data.edge_index, tmp_data.edge_attr)

        tmp_data.new_x=z

        return tmp_data.cpu()
    
    def save_model(self, path):
        torch.save({
            'gconv_state_dict': self.gconv.state_dict(),
            'encoder_model_state_dict': self.encoder_model.state_dict(),
            'contrast_model_state_dict': self.contrast_model.state_dict(),
        }, path)

        color_print(f'!!!!! SupGCL saving parameter in {path} Success')

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.gconv.load_state_dict(checkpoint['gconv_state_dict'])
        self.encoder_model.load_state_dict(checkpoint['encoder_model_state_dict'])
        self.contrast_model.load_state_dict(checkpoint['contrast_model_state_dict'])

        color_print(f'!!!!! SupGCL loading parameter from {path} Success')

        