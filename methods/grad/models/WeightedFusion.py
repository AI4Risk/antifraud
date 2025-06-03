import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import math
import dgl
import sympy
import scipy
from torch import nn
from torch.nn import init
import time
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, precision_score, confusion_matrix, average_precision_score
import numpy as np
from dgl.nn import GraphConv

class PolyConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 theta,
                 activation=F.leaky_relu,
                 lin=False,
                 bias=False):
        super().__init__()
        self._theta = theta
        self._k = len(self._theta)
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation
        self.linear = nn.Linear(in_feats, out_feats, bias)
        self.lin = lin
        # self.reset_parameters()
        # self.linear2 = nn.Linear(out_feats, out_feats, bias)

    def reset_parameters(self):
        if self.linear.weight is not None:
            init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, graph, feat):
        def unnLaplacian(feat, D_invsqrt, graph):
            """ Operation Feat * D^-1/2 A D^-1/2 """
            graph.ndata['h'] = feat * D_invsqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return feat - graph.ndata.pop('h') * D_invsqrt

        with graph.local_scope():
            D_invsqrt = torch.pow(graph.in_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            h = self._theta[0]*feat
            for k in range(1, self._k):
                feat = unnLaplacian(feat, D_invsqrt, graph)
                h += self._theta[k]*feat
        if self.lin:
            h = self.linear(h)
            h = self.activation(h)
        return h
    
class PolyConvBatch(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 theta,
                 activation=F.leaky_relu,
                 lin=False,
                 bias=False):
        super().__init__()
        self._theta = theta
        self._k = len(self._theta)
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation

    def reset_parameters(self):
        if self.linear.weight is not None:
            init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, block, feat):
        def unnLaplacian(feat, D_invsqrt, block):
            """ Operation Feat * D^-1/2 A D^-1/2 """
            block.srcdata['h'] = feat * D_invsqrt
            block.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return feat - block.srcdata.pop('h') * D_invsqrt

        with block.local_scope():
            D_invsqrt = torch.pow(block.out_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            h = self._theta[0]*feat
            for k in range(1, self._k):
                feat = unnLaplacian(feat, D_invsqrt, block)
                h += self._theta[k]*feat
        return h
    
def calculate_theta2(d):
    thetas = []
    x = sympy.symbols('x')
    for i in range(d+1):
        f = sympy.poly((x/2) ** i * (1 - x/2) ** (d-i) / (scipy.special.beta(i+1, d+1-i)))
        coeff = f.all_coeffs()
        inv_coeff = []
        for i in range(d+1):
            inv_coeff.append(float(coeff[d-i]))
        thetas.append(inv_coeff)
    return thetas

class WeightFusion(nn.Module):
    def __init__(self, global_args, in_feats, h_feats, num_classes, graph, relations_idx, device, d=2):
        super().__init__()
        self.g = graph
        self.thetas = calculate_theta2(d=d)
        self.h_feats = h_feats
        self.WFconv = [PolyConv(h_feats, h_feats, theta, lin=False) for theta in self.thetas]
        self.GCNconv = [GraphConv(h_feats, h_feats).to(device) for theta in self.thetas] # w/o WFusion
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats*len(self.WFconv), h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.act = nn.ReLU()
        self.relations_idx=relations_idx
        self.device=device
        self.global_args=global_args
        
        self.query=nn.Linear(h_feats,h_feats)
        self.key=nn.Linear(h_feats,h_feats)
        self.value=nn.Linear(h_feats,h_feats)
        self.sqrt_d=h_feats**0.5
        
        self.relation_weights = nn.Parameter(torch.ones(len(relations_idx), 1,1))
        
        # print(self.thetas)
        for param in self.parameters():
            print(type(param), param.size())

    def forward(self, in_feat, graph):
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h_all = []

        cnt=0

        for etype in graph.canonical_etypes: # for GCNconv is not for conv
            graph = dgl.add_self_loop(graph, etype=etype) 

        for relation in graph.canonical_etypes:
            if cnt not in self.relations_idx:
                cnt=cnt+1
                continue
            else:
                cnt=cnt+1
            
            h_final = torch.zeros([len(in_feat), 0]).to(self.device)
            if self.global_args.WFusion_use_WFusion:
                for conv in self.WFconv: 
                    h0 = conv(graph[relation].to(self.device), h)
                    h_final = torch.cat([h_final, h0], -1)
                    # print(h_final.shape)
                h = self.linear3(h_final)
                h_all.append(h)
            else:
                for conv in self.GCNconv: 
                    h0 = conv(graph[relation].to(self.device), h)
                    h_final = torch.cat([h_final, h0], -1)
                    # print(h_final.shape)
                h = self.linear3(h_final)
                h_all.append(h)
        
        h_all=torch.stack(h_all)
        normalized_weights = F.softmax(self.relation_weights, dim=-1)
        h_all=torch.sum(h_all*normalized_weights,dim=0)
        
        h_all = self.act(h_all)
        h_all = self.linear4(h_all)
        return h_all

# threshold adjusting for best macro f1
def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs[:,1] > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre

def WFusionTrain(model, graph_WFusion, args, train_mask,val_mask,test_mask):
    features = graph_WFusion.ndata['feature'].to(args.device)
    labels = graph_WFusion.ndata['label'].cpu()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=0.01)
    best_auc,best_f1, final_tf1, final_trec, final_tpre, final_tmf1, final_tauc, final_tap = 0., 0., 0., 0., 0., 0., 0., 0.

    weight = (1-labels[train_mask]).sum().item() / labels[train_mask].sum().item()
    print('cross entropy weight: ', weight)

    # 用于记录loss和AUC
    losses = []
    auc_scores = []
    
    labels = graph_WFusion.ndata['label'].to(args.device)

    time_start = time.time()
    for e in range(args.WFusion_epochs):
        model.train()
        logits = model(features,graph_WFusion)
        # print(f'!!!logit:{logits.device},label:{labels.device}')
        loss = F.cross_entropy(logits[train_mask], labels[train_mask], weight=torch.tensor([1., weight]).to(args.device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        probs = logits.softmax(1)
        
        # 确保概率和标签都在CPU上，并转换为Numpy数组
        probs_cpu = probs.detach().cpu().numpy()
        labels_cpu = labels.cpu().numpy()

        # 修改val_mask确保也在CPU上，并为Numpy数组
        val_mask_cpu = val_mask.cpu().numpy()
        test_mask_cpu = test_mask.cpu().numpy()

        # 调用 get_best_f1 函数，确保输入为CPU上的Numpy数组
        f1, thres = get_best_f1(labels_cpu[val_mask_cpu], probs_cpu[val_mask_cpu])

        # 以下操作也需要确保所有数据都在CPU上，并且为Numpy数组
        preds = np.zeros_like(labels_cpu)  # 使用CPU上的Numpy数组
        preds[probs_cpu[:, 1] > thres] = 1

        
        # f1, thres = get_best_f1(labels[val_mask], probs[val_mask])
        # preds = numpy.zeros_like(labels)
        # preds[probs[:, 1] > thres] = 1
        trec = recall_score(labels_cpu[test_mask_cpu], preds[test_mask_cpu])
        tpre = precision_score(labels_cpu[test_mask_cpu], preds[test_mask_cpu])
        tmf1 = f1_score(labels_cpu[test_mask_cpu], preds[test_mask_cpu], average='macro')
        tauc = roc_auc_score(labels_cpu[test_mask_cpu], probs_cpu[test_mask_cpu][:, 1])
        tap = average_precision_score(labels_cpu[test_mask_cpu], probs_cpu[test_mask_cpu][:, 1])

        if best_f1 < f1:
            best_f1 = f1
            
        if best_auc < tauc:
            best_auc = tauc
            final_trec = trec
            final_tpre = tpre
            final_tmf1 = tmf1
            final_tauc = tauc
            final_tap = tap
   
        print('Epoch {}, loss: {:.4f}, val mf1: {:.4f}, val auc: {:.4f}, val ap: {:.4f}, (best {:.4f}), (best_auc {:.4f})'.format(e, loss, f1, tauc, tap, best_f1, best_auc))

        losses.append(loss.item())
        auc_scores.append(tauc)


    time_end = time.time()
    print('time cost: ', time_end - time_start, 's')
    print('Test: REC {:.2f} PRE {:.2f} MF1 {:.2f} AUC {:.2f} AP {:.2f}'.format(final_trec*100,
                                                                    final_tpre*100, final_tmf1*100, final_tauc*100, final_tap*100))

    return final_tauc, final_tap, losses,auc_scores
