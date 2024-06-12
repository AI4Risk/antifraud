import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv

class Layer_AGG(nn.Module):
    def __init__(self, in_feat, out_feat, drop_rate=0.6,weight=1,num_layers =2,layers_tree=2):
        super(Layer_AGG, self).__init__()
        self.drop_rate = drop_rate
        self.weight = weight
        self.num_layers = num_layers
        self.layers_tree = layers_tree
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = in_feat if i==0 else out_feat
            self.convs.append(SAGEConv(in_channels,out_feat))
        self.conv_tree = nn.ModuleList()
        self.gating_networks = nn.ModuleList()
        for i in range(0,layers_tree):
            self.conv_tree.append(SAGEConv(in_feat,out_feat))
            self.gating_networks.append(nn.Linear(out_feat, 1))
        self.bias = nn.Parameter(torch.zeros(layers_tree))  


    def forward(self, x, edge_index):
        h = x
        layer_outputs = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index[0])
            if i != self.num_layers - 1:  # No activation and dropout on the last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.drop_rate, training=self.training)
                
        for i in range(0,self.layers_tree):
            temp = self.conv_tree[i](h,edge_index[1][i])
            temp = F.relu(temp)
            temp = F.dropout(temp,p=self.drop_rate,training=self.training)
            layer_outputs.append(temp)
        # print(layer_outputs[0].shape)

        weighted_sums = [self.gating_networks[i](layer_outputs[i]) for i in range(self.layers_tree)]
        
        # print(weighted_sums[0].shape)
        
        alpha = F.softmax(torch.stack(weighted_sums, dim=-1), dim=-1)

        # print(alpha.shape)
        x_tree = torch.zeros_like(layer_outputs[0])  
        for i in range(self.layers_tree):
        
            weight = alpha[:, :, i]  
            x_tree += layer_outputs[i] * weight

        return x+self.weight*x_tree
    
class multi_HOGRL_Model(nn.Module):
    def __init__(self,in_feat,out_feat,relation_nums = 3, hidden = 32,drop_rate=0.6,weight = 1,num_layers = 2,layers_tree=2):
        super(multi_HOGRL_Model, self).__init__()
        self.relation_nums=relation_nums
        self.drop_rate = drop_rate
        self.weight = weight
        self.layers_tree = layers_tree
        for i in range(relation_nums):
            setattr(self,'Layers'+str(i),Layer_AGG(in_feat,hidden,self.drop_rate,self.weight,num_layers,self.layers_tree))
        self.linear=nn.Linear(hidden*relation_nums,out_feat)

    
    def forward(self, x, edge_index):

        layer_outputs = []

        for i in range(self.relation_nums):
            layer_output = getattr(self, 'Layers' + str(i))(x, edge_index[i])
            layer_outputs.append(layer_output)

        x_temp = torch.cat(layer_outputs, dim=1)

        x = self.linear(x_temp)
        x = F.log_softmax(x, dim=1)
        return x,x_temp


class Graphsage(nn.Module):
    def __init__(self, in_feat,out_feat):
        super(Graphsage, self).__init__()
        self.conv1 = SAGEConv(in_feat, out_feat)
        self.conv2 = SAGEConv(out_feat, out_feat)
        # self.conv1 = GCNConv(in_feat, out_feat)
        # self.conv2 = GCNConv(out_feat, out_feat)
        self.linear = nn.Linear(out_feat,2)


    def forward(self,x,edge_index):
        x = self.conv1(x,edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x,edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.linear(x)
        x = F.log_softmax(x,dim=1)
        return x