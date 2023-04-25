import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv


class TransactionGCN(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, g, device):
        super().__init__()
        self.conv1 = GraphConv(in_feats, hidden_feats,
                               allow_zero_in_degree=True)
        self.conv2 = GraphConv(hidden_feats, out_feats,
                               allow_zero_in_degree=True)
        self.lin1 = nn.Linear(in_feats, hidden_feats)
        self.lin2 = nn.Linear(hidden_feats, out_feats)

        g.ndata['feat'] = torch.nn.init.xavier_uniform_(torch.empty(
            g.num_nodes(), g.edata['feat'].shape[1])).to(torch.float32).to(device)
        g.ndata['h'] = g.ndata['feat']
        g.edata['x'] = g.edata['feat']

    def forward(self, g, h, e):
        h1 = torch.relu(self.conv1(g, h))
        e1 = torch.relu(self.lin1(e))
        g.ndata['h'] = h1
        g.edata['x'] = e1
        g.apply_edges(
            lambda edges: {'x': edges.src['h'] + edges.dst['h'] + edges.data['x']})

        h2 = self.conv2(g, h1)
        e2 = torch.relu(self.lin2(e1))
        g.ndata['h'] = h2
        g.edata['x'] = e2
        g.apply_edges(
            lambda edges: {'x': edges.src['h'] + edges.dst['h'] + edges.data['x']})
        return g.ndata['h'], g.edata['x']


class stagn_2d_model(nn.Module):

    def __init__(
        self,
        time_windows_dim: int,
        feat_dim: int,
        num_classes: int,
        attention_hidden_dim: int,
        g: dgl.DGLGraph,
        filter_sizes: tuple = (2, 2),
        num_filters: int = 64,
        in_channels: int = 1,
        device="cpu"
    ) -> None:
        """
        Initialize the STAGN-2d model

        Args:
        :param time_windows_dim (int): length of time windows
        :param feat_dim (int): feature dimension
        :param num_classes (int): number of classes
        :param attention_hidden_dim (int): attention hidden dimenstion
        :param g (dgl.DGLGraph): dgl graph for gcn embeddings
        :param filter_sizes (tuple, optional): cnn filter size
        :param num_filters (int, optional): number of hidden channels
        :param in_channels (int, optional): number of in channels
        :param device (str, optional): where to train the model
        """
        super().__init__()
        self.time_windows_dim = time_windows_dim
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.attention_hidden_dim = attention_hidden_dim

        self.filter_sizes = filter_sizes
        self.num_filters = num_filters

        self.graph = g.to(device)

        # attention layer
        self.attention_W = nn.Parameter(torch.Tensor(
            self.feat_dim, self.attention_hidden_dim).uniform_(0., 1.))
        self.attention_U = nn.Parameter(torch.Tensor(
            self.feat_dim, self.attention_hidden_dim).uniform_(0., 1.))
        self.attention_V = nn.Parameter(torch.Tensor(
            self.attention_hidden_dim, 1).uniform_(0., 1.))

        # cnn layer
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_filters,
            kernel_size=filter_sizes,
            padding='same'
        )

        # FC layer
        self.flatten = nn.Flatten()
        self.linears1 = nn.Sequential(
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.LazyLinear(24),
            nn.ReLU())

        self.linears2 = nn.LazyLinear(self.num_classes)

        # gnn for transaction graph
        self.gcn = TransactionGCN(
            g.edata['feat'].shape[1], 128, 8, g, device)

    def attention_layer(
        self,
        X: torch.Tensor
    ):
        self.output_att = []
        # input_att = torch.split(X, self.time_windows_dim, dim=1)
        input_att = torch.split(X, 1, dim=1)  # 第二个参数是split_size!
        for index, x_i in enumerate(input_att):
            # print(f"x_i shape: {x_i.shape}")
            x_i = x_i.reshape(-1, self.feat_dim)
            c_i = self.attention(x_i, input_att, index)
            inp = torch.concat([x_i, c_i], axis=1)
            self.output_att.append(inp)

        input_conv = torch.reshape(torch.concat(self.output_att, axis=1),
                                   [-1, self.time_windows_dim, self.feat_dim*2])

        self.input_conv_expanded = torch.unsqueeze(input_conv, 1)

        return self.input_conv_expanded

    def cnn_layer(
        self,
        input: torch.Tensor
    ):
        if len(input.shape) == 3:
            self.input_conv_expanded = torch.unsqueeze(input, 1)
        elif len(input.shape) == 4:
            self.input_conv_expanded = input
        else:
            print("Wrong conv input shape!")

        self.input_conv_expanded = F.relu(self.conv(input))

        return self.input_conv_expanded

    def attention(self, x_i, x, index):
        e_i = []
        c_i = []

        for i in range(len(x)):
            output = x[i]
            output = output.reshape(-1, self.feat_dim)
            att_hidden = torch.tanh(torch.add(torch.matmul(
                x_i, self.attention_W), torch.matmul(output, self.attention_U)))
            e_i_j = torch.matmul(att_hidden, self.attention_V)
            e_i.append(e_i_j)

        e_i = torch.concat(e_i, axis=1)
        # print(f"e_i shape: {e_i.shape}")
        alpha_i = F.softmax(e_i, dim=1)
        alpha_i = torch.split(alpha_i, 1, 1)  # !!!

        for j, (alpha_i_j, output) in enumerate(zip(alpha_i, x)):
            if j == index:
                continue
            else:
                output = output.reshape(-1, self.feat_dim)
                c_i_j = torch.multiply(alpha_i_j, output)
                c_i.append(c_i_j)

        c_i = torch.reshape(torch.concat(c_i, axis=1),
                            [-1, self.time_windows_dim-1, self.feat_dim])
        c_i = torch.sum(c_i, dim=1)
        return c_i

    def forward(self, X_nume, g):
        # X shape be like: (batch_size, time_windows_dim, feat_dim)
        out = self.attention_layer(X_nume)  # all, 1, 8, 10

        out = self.cnn_layer(out)  # all, 64, 8, 10
        node_embs, edge_embs = self.gcn(g, g.ndata['feat'], g.edata['feat'])

        src_nds, dst_nds = g.edges()
        src_feat = g.ndata['h'][src_nds]
        dst_feat = g.ndata['h'][dst_nds]
        # all, 3, embedding_dim
        node_feats = torch.stack(
            [src_feat, dst_feat, edge_embs], dim=1).view(X_nume.shape[0], -1)
        out = self.flatten(out)
        out = self.linears1(out)
        out = torch.cat([out, node_feats], dim=1)
        out = self.linears2(out)

        return out
