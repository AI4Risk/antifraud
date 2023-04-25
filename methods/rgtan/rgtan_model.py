import torch
import torch.nn as nn
import torch.optim as optim
from dgl.utils import expand_as_pair
from dgl import function as fn
from dgl.base import DGLError
from dgl.nn.functional import edge_softmax
import numpy as np
import pandas as pd
from math import sqrt


class PosEncoding(nn.Module):

    def __init__(self, dim, device, base=10000, bias=0):

        super(PosEncoding, self).__init__()
        """
        Initialize the posencoding component
        :param dim: the encoding dimension 
		:param device: where to train model
		:param base: the encoding base
		:param bias: the encoding bias
        """
        p = []
        sft = []
        for i in range(dim):
            b = (i - i % 2) / dim
            p.append(base ** -b)
            if i % 2:
                sft.append(np.pi / 2.0 + bias)
            else:
                sft.append(bias)
        self.device = device
        self.sft = torch.tensor(
            sft, dtype=torch.float32).view(1, -1).to(device)
        self.base = torch.tensor(p, dtype=torch.float32).view(1, -1).to(device)

    def forward(self, pos):
        with torch.no_grad():
            if isinstance(pos, list):
                pos = torch.tensor(pos, dtype=torch.float32).to(self.device)
            pos = pos.view(-1, 1)
            x = pos / self.base + self.sft
            return torch.sin(x)


class TransformerConv(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 bias=True,
                 allow_zero_in_degree=False,
                 # feat_drop=0.6,
                 # attn_drop=0.6,
                 skip_feat=True,
                 gated=True,
                 layer_norm=True,
                 activation=nn.PReLU()):
        """
        Initialize the transformer layer.
        Attentional weights are jointly optimized in an end-to-end mechanism with graph neural networks and fraud detection networks.
            :param in_feat: the shape of input feature
            :param out_feats: the shape of output feature
            :param num_heads: the number of multi-head attention 
            :param bias: whether to use bias
            :param allow_zero_in_degree: whether to allow zero in degree
            :param skip_feat: whether to skip some feature 
            :param gated: whether to use gate
            :param layer_norm: whether to use layer regularization
            :param activation: the type of activation function   
        """

        super(TransformerConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._num_heads = num_heads

        self.lin_query = nn.Linear(
            self._in_src_feats, self._out_feats*self._num_heads, bias=bias)
        self.lin_key = nn.Linear(
            self._in_src_feats, self._out_feats*self._num_heads, bias=bias)
        self.lin_value = nn.Linear(
            self._in_src_feats, self._out_feats*self._num_heads, bias=bias)

        # self.feat_dropout = nn.Dropout(p=feat_drop)
        # self.attn_dropout = nn.Dropout(p=attn_drop)
        if skip_feat:
            self.skip_feat = nn.Linear(
                self._in_src_feats, self._out_feats*self._num_heads, bias=bias)
        else:
            self.skip_feat = None
        if gated:
            self.gate = nn.Linear(
                3*self._out_feats*self._num_heads, 1, bias=bias)
        else:
            self.gate = None
        if layer_norm:
            self.layer_norm = nn.LayerNorm(self._out_feats*self._num_heads)
        else:
            self.layer_norm = None
        self.activation = activation

    def forward(self, graph, feat, get_attention=False):
        """
        Description: Transformer Graph Convolution
        :param graph: input graph
            :param feat: input feat
            :param get_attention: whether to get attention
        """

        graph = graph.local_var()

        if not self._allow_zero_in_degree:
            if (graph.in_degrees() == 0).any():
                raise DGLError('There are 0-in-degree nodes in the graph, '
                               'output for those nodes will be invalid. '
                               'This is harmful for some applications, '
                               'causing silent performance regression. '
                               'Adding self-loop on the input graph by '
                               'calling `g = dgl.add_self_loop(g)` will resolve '
                               'the issue. Setting ``allow_zero_in_degree`` '
                               'to be `True` when constructing this module will '
                               'suppress the check and let the code run.')

        # check if feat is a tuple
        if isinstance(feat, tuple):
            h_src = feat[0]
            h_dst = feat[1]
        else:
            h_src = feat
            h_dst = h_src[:graph.number_of_dst_nodes()]

        # Step 0. q, k, v
        q_src = self.lin_query(
            h_src).view(-1, self._num_heads, self._out_feats)
        k_dst = self.lin_key(h_dst).view(-1, self._num_heads, self._out_feats)
        v_src = self.lin_value(
            h_src).view(-1, self._num_heads, self._out_feats)
        # Assign features to nodes
        graph.srcdata.update({'ft': q_src, 'ft_v': v_src})
        graph.dstdata.update({'ft': k_dst})
        # Step 1. dot product
        graph.apply_edges(fn.u_dot_v('ft', 'ft', 'a'))

        # Step 2. edge softmax to compute attention scores
        graph.edata['sa'] = edge_softmax(
            graph, graph.edata['a'] / self._out_feats**0.5)

        # Step 3. Broadcast softmax value to each edge, and aggregate dst node
        graph.update_all(fn.u_mul_e('ft_v', 'sa', 'attn'),
                         fn.sum('attn', 'agg_u'))

        # output results to the destination nodes
        rst = graph.dstdata['agg_u'].reshape(-1,
                                             self._out_feats*self._num_heads)

        if self.skip_feat is not None:
            skip_feat = self.skip_feat(feat[:graph.number_of_dst_nodes()])
            if self.gate is not None:
                gate = torch.sigmoid(
                    self.gate(
                        torch.concat([skip_feat, rst, skip_feat - rst], dim=-1)))
                rst = gate * skip_feat + (1 - gate) * rst
            else:
                rst = skip_feat + rst

        if self.layer_norm is not None:
            rst = self.layer_norm(rst)

        if self.activation is not None:
            rst = self.activation(rst)

        if get_attention:
            return rst, graph.edata['sa']
        else:
            return rst


class Tabular1DCNN2(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        K: int = 4,  # K*input_dim -> hidden dim
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hid_dim = input_dim * embed_dim * 2
        self.cha_input = self.cha_output = input_dim
        self.cha_hidden = (input_dim*K) // 2
        self.sign_size1 = 2 * embed_dim
        self.sign_size2 = embed_dim
        self.K = K

        self.bn1 = nn.BatchNorm1d(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dense1 = nn.Linear(input_dim, self.hid_dim)

        self.bn_cv1 = nn.BatchNorm1d(self.cha_input)
        self.conv1 = nn.Conv1d(
            in_channels=self.cha_input,
            out_channels=self.cha_input*self.K,
            kernel_size=5,
            padding=2,
            groups=self.cha_input,
            bias=False
        )

        self.ave_pool1 = nn.AdaptiveAvgPool1d(self.sign_size2)

        self.bn_cv2 = nn.BatchNorm1d(self.cha_input*self.K)
        self.dropout2 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            in_channels=self.cha_input*self.K,
            out_channels=self.cha_input*(self.K),
            kernel_size=3,
            padding=1,
            bias=True
        )

        self.bn_cv3 = nn.BatchNorm1d(self.cha_input*self.K)
        self.conv3 = nn.Conv1d(
            in_channels=self.cha_input*(self.K),
            out_channels=self.cha_input*(self.K//2),
            kernel_size=3,
            padding=1,
            # groups=self.cha_hidden,
            bias=True
        )

        self.bn_cvs = nn.ModuleList()
        self.convs = nn.ModuleList()
        for i in range(6):
            self.bn_cvs.append(nn.BatchNorm1d(self.cha_input*(self.K//2)))
            self.convs.append(nn.Conv1d(
                in_channels=self.cha_input*(self.K//2),
                out_channels=self.cha_input*(self.K//2),
                kernel_size=3,
                padding=1,
                # groups=self.cha_hidden,
                bias=True
            ))

        self.bn_cv10 = nn.BatchNorm1d(self.cha_input*(self.K//2))
        self.conv10 = nn.Conv1d(
            in_channels=self.cha_input*(self.K//2),
            out_channels=self.cha_output,
            kernel_size=3,
            padding=1,
            # groups=self.cha_hidden,
            bias=True
        )

    def forward(self, x):
        x = self.dropout1(self.bn1(x))
        x = nn.functional.celu(self.dense1(x))
        x = x.reshape(x.shape[0], self.cha_input,
                      self.sign_size1)

        x = self.bn_cv1(x)
        x = nn.functional.relu(self.conv1(x))
        x = self.ave_pool1(x)

        x_input = x
        x = self.dropout2(self.bn_cv2(x))
        x = nn.functional.relu(self.conv2(x))  # -> (|b|,24,32)
        x = x + x_input

        x = self.bn_cv3(x)
        x = nn.functional.relu(self.conv3(x))  # -> (|b|,6,32)

        for i in range(6):
            x_input = x
            x = self.bn_cvs[i](x)
            x = nn.functional.relu(self.convs[i](x))
            x = x + x_input

        x = self.bn_cv10(x)
        x = nn.functional.relu(self.conv10(x))

        return x


class TransEmbedding(nn.Module):

    def __init__(
        self,
        df=None,
        device='cpu',
        dropout=0.2,
        in_feats_dim=82,
        cat_features=None,
        neigh_features: dict = None,
        att_head_num: int = 4,  # yelp 4 amazon 5 S-FFSD 9
        neighstat_uni_dim=64
    ):
        """
        Initialize the attribute embedding and feature learning compoent

        :param df: the feature (|train_idx|, |feat|)
        :param device: where to train model
        :param dropout: the dropout rate
        :param in_feats_dim: the shape of input feature in dimension 1
        :param cat_features: category features
        :param neigh_features: neighbor riskstat features
        :param att_head_num: attention head number for riskstat embeddings
        """
        super(TransEmbedding, self).__init__()
        self.time_pe = PosEncoding(dim=in_feats_dim, device=device, base=100)

        self.cat_table = nn.ModuleDict({col: nn.Embedding(max(df[col].unique(
        ))+1, in_feats_dim).to(device) for col in cat_features if col not in {"Labels", "Time"}})

        if isinstance(neigh_features, dict):
            self.nei_table = Tabular1DCNN2(input_dim=len(
                neigh_features), embed_dim=in_feats_dim)

        self.att_head_num = att_head_num
        self.att_head_size = int(in_feats_dim / att_head_num)
        self.total_head_size = in_feats_dim
        self.lin_q = nn.Linear(in_feats_dim, self.total_head_size)
        self.lin_k = nn.Linear(in_feats_dim, self.total_head_size)
        self.lin_v = nn.Linear(in_feats_dim, self.total_head_size)

        self.lin_final = nn.Linear(in_feats_dim, in_feats_dim)
        self.layer_norm = nn.LayerNorm(in_feats_dim, eps=1e-8)

        self.neigh_mlp = nn.Linear(in_feats_dim, 1)

        self.neigh_add_mlp = nn.ModuleList([nn.Linear(in_feats_dim, in_feats_dim) for i in range(
            len(neigh_features.columns))]) if isinstance(neigh_features, pd.DataFrame) else None

        self.label_table = nn.Embedding(
            3, in_feats_dim, padding_idx=2).to(device)
        self.time_emb = None
        self.emb_dict = None
        self.label_emb = None
        self.cat_features = cat_features
        self.neigh_features = neigh_features
        self.forward_mlp = nn.ModuleList(
            [nn.Linear(in_feats_dim, in_feats_dim) for i in range(len(cat_features))])
        self.dropout = nn.Dropout(dropout)

    def forward_emb(self, cat_feat):
        if self.emb_dict is None:
            self.emb_dict = self.cat_table
        # print(self.emb_dict)
        # print(df['trans_md'])
        support = {col: self.emb_dict[col](
            cat_feat[col]) for col in self.cat_features if col not in {"Labels", "Time"}}
        return support

    def transpose_for_scores(self, input_tensor):
        new_x_shape = input_tensor.size(
        )[:-1] + (self.att_head_num, self.att_head_size)
        # (|batch|, feat_num, dim) -> (|batch|, feta_num, head_num, head_size)
        input_tensor = input_tensor.view(*new_x_shape)
        return input_tensor.permute(0, 2, 1, 3)

    def forward_neigh_emb(self, neighstat_feat):
        cols = neighstat_feat.keys()
        tensor_list = []
        for col in cols:
            tensor_list.append(neighstat_feat[col])
        neis = torch.stack(tensor_list).T
        input_tensor = self.nei_table(neis)

        mixed_q_layer = self.lin_q(input_tensor)
        mixed_k_layer = self.lin_k(input_tensor)
        mixed_v_layer = self.lin_v(input_tensor)

        q_layer = self.transpose_for_scores(mixed_q_layer)
        k_layer = self.transpose_for_scores(mixed_k_layer)
        v_layer = self.transpose_for_scores(mixed_v_layer)

        att_scores = torch.matmul(q_layer, k_layer.transpose(-1, -2))
        att_scores = att_scores / sqrt(self.att_head_size)

        att_probs = nn.Softmax(dim=-1)(att_scores)
        # dropout?
        context_layer = torch.matmul(att_probs, v_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context_layer.size()[:-2] + (self.total_head_size,)
        context_layer = context_layer.view(*new_context_shape)
        hidden_states = self.lin_final(context_layer)
        # dropout?
        # hidden_states = self.layer_norm(hidden_states + input_tensor)
        hidden_states = self.layer_norm(hidden_states)

        return hidden_states, cols
        # return input_tensor, cols

    def forward(self, cat_feat: dict, neighstat_feat: dict):
        support = self.forward_emb(cat_feat)
        cat_output = 0
        nei_output = 0
        for i, k in enumerate(support.keys()):
            # if k =='time_span':
            #    print(df[k].shape)
            support[k] = self.dropout(support[k])
            support[k] = self.forward_mlp[i](support[k])
            cat_output = cat_output + support[k]

        if neighstat_feat is not None:
            nei_embs, cols_list = self.forward_neigh_emb(neighstat_feat)
            nei_output = self.neigh_mlp(nei_embs).squeeze(-1)

            # nei_output = nei_embs.mean(axis=-1)

        return cat_output, nei_output


class RGTAN(nn.Module):
    def __init__(self,
                 in_feats,
                 hidden_dim,
                 n_layers,
                 n_classes,
                 heads,
                 activation,
                 skip_feat=True,
                 gated=True,
                 layer_norm=True,
                 post_proc=True,
                 n2v_feat=True,
                 drop=None,
                 ref_df=None,
                 cat_features=None,
                 neigh_features=None,
                 nei_att_head=4,
                 device='cpu'):
        """
        Initialize the RGTAN-GNN model
        :param in_feats: the shape of input feature
        :param hidden_dim: model hidden layer dimension
        :param n_layers: the number of GTAN layers
        :param n_classes: the number of classification
        :param heads: the number of multi-head attention 
        :param activation: the type of activation function
        :param skip_feat: whether to skip some feature
        :param gated: whether to use gate
        :param layer_norm: whether to use layer regularization
        :param post_proc: whether to use post processing
        :param n2v_feat: whether to use n2v features
        :param drop: whether to use drop
        :param ref_df: whether to refer other node features
        :param cat_features: category features
        :param neigh_features: neighbor statistic features
        :param nei_att_head: multihead attention for neighbor riskstat features
        :param device: where to train model
        """

        super(RGTAN, self).__init__()
        self.in_feats = in_feats  # feature dimension
        self.hidden_dim = hidden_dim  # 64
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.heads = heads  # [4,4,4]
        self.activation = activation  # PRelu
        # self.input_drop = lambda x: x
        self.input_drop = nn.Dropout(drop[0])
        self.drop = drop[1]
        self.output_drop = nn.Dropout(self.drop)
        # self.pn = PairNorm(mode=pairnorm)
        if n2v_feat:
            self.n2v_mlp = TransEmbedding(
                ref_df, device=device, in_feats_dim=in_feats, cat_features=cat_features, neigh_features=neigh_features, att_head_num=nei_att_head)
            self.nei_feat_dim = len(neigh_features.keys()) if isinstance(
                neigh_features, dict) else 0
        else:
            self.n2v_mlp = lambda x: x
            self.nei_feat_dim = 0
        self.layers = nn.ModuleList()
        self.layers.append(nn.Embedding(
            n_classes+1, in_feats + self.nei_feat_dim, padding_idx=n_classes))
        self.layers.append(
            nn.Linear(self.in_feats + self.nei_feat_dim, self.hidden_dim*self.heads[0]))
        self.layers.append(
            nn.Linear(self.in_feats + self.nei_feat_dim, self.hidden_dim*self.heads[0]))
        self.layers.append(nn.Sequential(nn.BatchNorm1d(self.hidden_dim*self.heads[0]),
                                         nn.PReLU(),
                                         nn.Dropout(self.drop),
                                         nn.Linear(self.hidden_dim *
                                                   self.heads[0], in_feats + self.nei_feat_dim)
                                         ))

        # build multiple layers
        self.layers.append(TransformerConv(in_feats=self.in_feats + self.nei_feat_dim,
                                           out_feats=self.hidden_dim,
                                           num_heads=self.heads[0],
                                           skip_feat=skip_feat,
                                           gated=gated,
                                           layer_norm=layer_norm,
                                           activation=self.activation))

        for l in range(0, (self.n_layers - 1)):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.layers.append(TransformerConv(in_feats=self.hidden_dim * self.heads[l - 1],
                                               out_feats=self.hidden_dim,
                                               num_heads=self.heads[l],
                                               skip_feat=skip_feat,
                                               gated=gated,
                                               layer_norm=layer_norm,
                                               activation=self.activation))
        if post_proc:
            self.layers.append(nn.Sequential(nn.Linear(self.hidden_dim * self.heads[-1], self.hidden_dim * self.heads[-1]),
                                             nn.BatchNorm1d(
                                                 self.hidden_dim * self.heads[-1]),
                                             nn.PReLU(),
                                             nn.Dropout(self.drop),
                                             nn.Linear(self.hidden_dim * self.heads[-1], self.n_classes)))
        else:
            self.layers.append(nn.Linear(self.hidden_dim *
                               self.heads[-1], self.n_classes))

    def forward(self, blocks, features, labels, n2v_feat=None, neighstat_feat=None):
        """
        :param blocks: train blocks
        :param features: train features
        :param labels: train labels
        :param n2v_feat: whether to use n2v features
        :param neighstat_feat: neighbor riskstat features
        """
        if n2v_feat is None and neighstat_feat is None:
            h = features
        else:
            cat_h, nei_h = self.n2v_mlp(n2v_feat, neighstat_feat)
            h = features + cat_h
            if isinstance(nei_h, torch.Tensor):
                h = torch.cat([h, nei_h], dim=-1)

        label_embed = self.input_drop(self.layers[0](labels))
        label_embed = self.layers[1](
            h) + self.layers[2](label_embed)  # 2926, 2926, 256
        # label_embed = self.layers[1](h)
        label_embed = self.layers[3](label_embed)
        h = h + label_embed

        for l in range(self.n_layers):
            h = self.output_drop(self.layers[l+4](blocks[l], h))

        logits = self.layers[-1](h)

        return logits
