import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.transforms import GDC
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score, roc_auc_score
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from scipy.io import loadmat
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import scipy.io
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import networkx as nx

import numpy as np

from tqdm import tqdm

import argparse
import os

from utils.dataProcess import loadDataset, mergeGraphDataList, GDCAugment, data4WFusionTrain
from utils.MyUtils import color_print
from utils.args import argVar

from models.WeightedFusion import WeightFusion, WFusionTrain


def main():
    final_ap=[]
    final_auc=[]
    for i in tqdm((1,)):
        args = argVar()
        # print(args)
        # prepare data
        graph_dgl, graph_pyg, train_mask, val_mask, test_mask = loadDataset(dataset=args.dataset, train_ratio=args.train_ratio)
        in_feats = graph_dgl.ndata['feature'].shape[1]
        num_classes = 2

        if args.GuiDDPM_sample_with_guidance:
            syn_relation_filename = f"./Generation/SynRelation_{args.dataset}_{args.GuiDDPM_sample_diffusion_steps}Samplesteps_{args.GuiDDPM_train_steps}Trainsteps_subgraphsize_{args.nodes_per_subgraph}_guided.pt"
        else:
            syn_relation_filename = f"./Generation/SynRelation_{args.dataset}_{args.GuiDDPM_sample_diffusion_steps}Samplesteps_{args.GuiDDPM_train_steps}Trainsteps_subgraphsize_{args.nodes_per_subgraph}_unguided.pt"

        # syn_relation_filename='./Generation/ddpm_yelp_dict_even-v2.pt'

        syn_relation_dict=torch.load(syn_relation_filename)

        graph_syn=mergeGraphDataList(args=args, graph_pyg=graph_pyg, syn_relation_dict=syn_relation_dict)

        color_print(f'!!!!! Strat gdc augment')
        graph_gdc_list=[]
        for avg_degree in tqdm(args.WFusion_gdc_syn_avg_degree):
            filename=f'./Generation/GDCAugGraph/GDC_SynRelation_{args.dataset}_{args.GuiDDPM_sample_diffusion_steps}Samplesteps_{args.GuiDDPM_train_steps}Trainsteps_avgdegree_{avg_degree}.pt'

            if os.path.exists(filename):
                gdc_aug_graph=torch.load(filename)
                color_print(f'gdc_aud{len(graph_gdc_list)+1} is load from {filename}') 
            else:
                gdc_aug_graph=GDCAugment(graph_pyg_type=graph_syn, avg_degree=avg_degree)
                torch.save(gdc_aug_graph,filename)
                color_print(f'gdc_aud{len(graph_gdc_list)+1} is saved in {filename}') 
            graph_gdc_list.append(gdc_aug_graph)

        for avg_degree in tqdm(args.WFusion_gdc_raw_avg_degree):
            filename=f'./Generation/GDCAugGraph/GDC_RawRelation_{args.dataset}_{args.GuiDDPM_sample_diffusion_steps}Samplesteps_{args.GuiDDPM_train_steps}Trainsteps_avgdegree_{avg_degree}.pt'

            if os.path.exists(filename):
                gdc_aug_graph=torch.load(filename)
                color_print(f'gdc_aud{len(graph_gdc_list)+1} is load from {filename}') 
            else:
                gdc_aug_graph=GDCAugment(graph_pyg_type=graph_pyg, avg_degree=avg_degree)
                torch.save(gdc_aug_graph,filename)
                color_print(f'gdc_aud{len(graph_gdc_list)+1} is saved in {filename}') 
            graph_gdc_list.append(gdc_aug_graph)
            
        color_print(f'!!!!! Finish gdc augment')

        graph_WFusion=data4WFusionTrain(graph_pyg=graph_pyg, 
                                        graph_syn=graph_syn, 
                                        graph_gdc_list=graph_gdc_list
                                        )

        print(f"{graph_WFusion.ndata['feature'].shape}")

        model_WFusion=WeightFusion(global_args=args, in_feats=graph_WFusion.ndata['feature'].shape[1], h_feats=args.WFusion_hid_dim, num_classes=num_classes, graph=graph_WFusion, d=args.WFusion_order, relations_idx=args.WFusion_relation_index, device=args.device).to(args.device)

        auc,ap,losses_2,auc_2=WFusionTrain(model_WFusion, graph_WFusion, args,graph_WFusion.ndata['train_mask'],graph_WFusion.ndata['val_mask'],graph_WFusion.ndata['test_mask'])
        final_ap.append(ap)
        final_auc.append(auc)

    color_print(f'auc:{final_auc}')
    color_print(f'ap:{final_ap}')



if __name__=='__main__':
    main()
