from utils.MyUtils import color_print
from utils.args import argVar
from utils.dataProcess import loadDataset, nodeSelect, nodeSample, sampleCheck
from models.SupGCL import SupGCL
# my part
from models.GuiDDPM import GuiDDPM

import torch

import os
import argparse

def SupGCL_module(args, graph_pyg, graph_pyg_selected):
    model_SupGCL=SupGCL(global_args=args,
                          graph_pyg_selected=graph_pyg_selected,
                          nodes_per_subgraph=args.nodes_per_subgraph,
                          device=args.device,
                          num_train_part=args.SupGCL_num_train_part,
                          batch_size=args.SupGCL_batch_size)
    
    SupGCL_para_filename=f'./ModelPara/SupGCLPara/SupGCL_{args.dataset}_{args.SupGCL_epochs}epochs_subgraphsize_{args.nodes_per_subgraph}.pt'

    if args.SupGCL_train_flag and not os.path.exists(SupGCL_para_filename):
        model_SupGCL.train(epochs=args.SupGCL_epochs)
        model_SupGCL.save_model(path=SupGCL_para_filename)
    else:
        model_SupGCL.load_model(path=SupGCL_para_filename)
    
    if args.SupGCL_visualize_flag:
        model_SupGCL.visualize()

    return model_SupGCL.project(graph_pyg=graph_pyg), model_SupGCL

def GuiDDPM_module(args, graph_pyg_supgcl, node_groups, edge_index_unselected, guidance):
    GuiDDPM_para_filename = f"./ModelPara/GuiDDPMPara/GuiDDPM_{args.dataset}_{args.GuiDDPM_train_steps}steps_subgraphsize_{args.nodes_per_subgraph}.pt"
    if args.GuiDDPM_sample_with_guidance:
        syn_relation_filename = f"./Generation/SynRelation_{args.dataset}_{args.GuiDDPM_sample_diffusion_steps}Samplesteps_{args.GuiDDPM_train_steps}Trainsteps_subgraphsize_{args.nodes_per_subgraph}_guided.pt"
    else:
        syn_relation_filename = f"./Generation/SynRelation_{args.dataset}_{args.GuiDDPM_sample_diffusion_steps}Samplesteps_{args.GuiDDPM_train_steps}Trainsteps_subgraphsize_{args.nodes_per_subgraph}_unguided.pt"

    model_DDPM=GuiDDPM(global_args=args,
                      graph_pyg_supgcl=graph_pyg_supgcl,
                      node_groups=node_groups, 
                      edge_index_unselected=edge_index_unselected,
                      guidance=guidance, 
                      train_flag=args.GuiDDPM_train_flag, 
                      model_path=GuiDDPM_para_filename,
                      syn_relation_filename=syn_relation_filename,
                      device=args.device)

    # if args.GuiDDPM_train_flag and not os.path.exists(GuiDDPM_para_filename):
    if args.GuiDDPM_train_flag:
        if os.path.exists(GuiDDPM_para_filename):
            color_print(f'!!!!! GuiDDPM Parameter is loaded from {GuiDDPM_para_filename} Success')
        else:
            model_DDPM.train(train_steps=args.GuiDDPM_train_steps)
            model_DDPM.save_model(GuiDDPM_para_filename)
    else:
        model_DDPM.sample()

def sampleAnalysis(node_groups, nodes_per_subgraph):
    cnt=0
    num_edge=0
    wrong_num=0
    for g in node_groups:
        if g.x.shape[0]!=nodes_per_subgraph:
            color_print(g.x.shape[0],cnt)
            color_print(f'!!!!! Sampling check false')
            wrong_num=wrong_num+1
        cnt=cnt+1
        num_edge+=g.edge_index.shape[1]
    if not wrong_num:
        color_print(f'!!!!! Sampling check true')
    print(num_edge)

def main():
    args=argVar()
    
    print(f'device: {args.device}')

    # prepare data
    graph_dgl, graph_pyg, train_mask, val_mask, test_mask = loadDataset(dataset=args.dataset, train_ratio=args.train_ratio)

    edge_types = graph_dgl.canonical_etypes

    print(graph_pyg.y[train_mask].sum(),graph_pyg.y[val_mask].sum(),graph_pyg.y[test_mask].sum())

    graph_pyg_selected, edge_index_unselected=nodeSelect(graph_pyg=graph_pyg, nodes_per_subgraph=32)

    # Supervised graph contrastive learning (SupGCL)
    graph_pyg_supgcl, model_SupGCL=SupGCL_module(args=args,
                                            graph_pyg=graph_pyg,
                                            graph_pyg_selected=graph_pyg_selected)

    # node sample
    node_groups=nodeSample(graph_pyg=graph_pyg_supgcl, nodes_per_subgraph=args.nodes_per_subgraph)
    sampleCheck(node_groups=node_groups, nodes_per_subgraph=args.nodes_per_subgraph)
    sampleAnalysis(node_groups=node_groups, nodes_per_subgraph=args.nodes_per_subgraph)
    # exit()

    # GuiDDPM
    GuiDDPM_module(args=args, 
                   graph_pyg_supgcl=graph_pyg_supgcl,
                   node_groups=node_groups,
                   edge_index_unselected=edge_index_unselected, 
                   guidance=model_SupGCL)
    
    # Weighted Filter
    



if __name__=='__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 
    main()
