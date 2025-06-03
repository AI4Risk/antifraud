from .MyUtils import color_print, pyg_data_to_dgl_graph

import torch

import dgl
from dgl.data import FraudAmazonDataset, FraudYelpDataset

import torch_geometric
from torch_geometric.transforms import GDC
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.utils import to_dense_adj, dense_to_sparse

from sklearn.model_selection import train_test_split

from tqdm import tqdm

def splitDataset(index, label, train_ratio):
    idx_train, idx_rest, y_train, y_rest = train_test_split(index, label[index], stratify=label[index],
                                                            train_size=train_ratio,
                                                            random_state=2, shuffle=True)
    idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                            test_size=0.5,
                                                            random_state=2, shuffle=True)
    train_mask = torch.zeros([len(label)]).bool()
    val_mask = torch.zeros([len(label)]).bool()
    test_mask = torch.zeros([len(label)]).bool()
    train_mask[idx_train] = 1
    val_mask[idx_valid] = 1
    test_mask[idx_test] = 1

    return train_mask, val_mask, test_mask

def nodeSelect(graph_pyg, nodes_per_subgraph):
    num = graph_pyg.x.shape[0]
    num_selected = num - num % nodes_per_subgraph

    if num_selected:
        feat_selected=graph_pyg.x[:num_selected]
        label_selected=graph_pyg.y[:num_selected]
        train_mask_selected=graph_pyg.train_mask[:num_selected]
        test_mask_selected=graph_pyg.test_mask[:num_selected]
        val_mask_selected=graph_pyg.val_mask[:num_selected]
        
        edge_index_selected_mask=(graph_pyg.edge_index[0] < num_selected) & (graph_pyg.edge_index[1] < num_selected)
        edge_index_unselected=graph_pyg.edge_index[:,~edge_index_selected_mask]
        edge_index_selected=graph_pyg.edge_index[:,edge_index_selected_mask]

        graph_pyg_selected=Data(x=feat_selected,
                                edge_index=edge_index_selected,
                                y=label_selected,
                                train_mask=train_mask_selected,
                                val_mask=val_mask_selected,
                                test_mask=test_mask_selected)
        
        print(f'num_seleced: {num_selected}/{num}')

    return graph_pyg_selected, edge_index_unselected

def nodeSample(graph_pyg, nodes_per_subgraph):
    color_print(f'!!!!! Start node sampling')
    num = graph_pyg.x.shape[0]
    num_selected = num - num % nodes_per_subgraph

    subgraphs=[]

    for i in tqdm(range(num_selected//nodes_per_subgraph)):
        start_idx = i * nodes_per_subgraph
        
        subset=torch.arange(start_idx,start_idx+nodes_per_subgraph)
        
        sub_edge_index = torch_geometric.utils.subgraph(subset,graph_pyg.edge_index,num_nodes=graph_pyg.num_nodes)[0] - i*nodes_per_subgraph
        sub_x=graph_pyg.x[start_idx:start_idx + nodes_per_subgraph]
        sub_new_x=graph_pyg.new_x[start_idx:start_idx + nodes_per_subgraph]
        sub_y=graph_pyg.y[start_idx:start_idx + nodes_per_subgraph]
        sub_adj=to_dense_adj(sub_edge_index,max_num_nodes=nodes_per_subgraph)[0]
        sub_train_mask=graph_pyg.train_mask[start_idx:start_idx+nodes_per_subgraph]
        sub_val_mask=graph_pyg.val_mask[start_idx:start_idx+nodes_per_subgraph]
        sub_test_mask=graph_pyg.test_mask[start_idx:start_idx+nodes_per_subgraph]

        sub_data=Data(x=sub_x,new_x=sub_new_x,edge_index=sub_edge_index,y=sub_y,adj=sub_adj,train_mask=sub_train_mask,val_mask=sub_val_mask,test_mask=sub_test_mask)

        subgraphs.append(sub_data)
    
    color_print(f'!!!!! Finish node sampling')

    return subgraphs

def sampleCheck(node_groups, nodes_per_subgraph):
    cnt=0
    wrong_num=0
    for g in node_groups:
        if g.x.shape[0]!=nodes_per_subgraph:
            color_print(g.x.shape[0],cnt)
            color_print(f'!!!!! Sampling check false')
            wrong_num=wrong_num+1
        cnt=cnt+1
    if not wrong_num:
        color_print(f'!!!!! Sampling check true')

def loadDataset(dataset, train_ratio):
    if dataset=='yelp':
        graph_dgl = FraudYelpDataset()[0]
        graph_dgl = dgl.to_homogeneous(graph_dgl,ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
        graph_dgl = dgl.add_self_loop(graph_dgl)

        feat = graph_dgl.ndata['feature'].float()
        label = graph_dgl.ndata['label'].long()
        edge_index = torch.stack(graph_dgl.edges(etype=('_N', '_E', '_N')))
        index = list(range(len(label)))

        train_mask, val_mask, test_mask = splitDataset(index,label,train_ratio)

        graph_pyg=Data(x=feat,edge_index=edge_index,y=label,train_mask=train_mask,val_mask=val_mask,test_mask=test_mask)
        graph_dgl.ndata['train_mask']=train_mask
        graph_dgl.ndata['val_mask']=val_mask
        graph_dgl.ndata['test_mask']=test_mask

    elif dataset=='amazon':
        graph_dgl = FraudAmazonDataset()[0]
        graph_dgl = dgl.to_homogeneous(graph_dgl,ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
        graph_dgl = dgl.add_self_loop(graph_dgl)

        feat = graph_dgl.ndata['feature'].float()
        label = graph_dgl.ndata['label'].long()
        edge_index = torch.stack(graph_dgl.edges(etype=('_N', '_E', '_N')))
        index = list(range(3305, len(label)))

        train_mask, val_mask, test_mask = splitDataset(index,label,train_ratio)

        graph_pyg=Data(x=feat,edge_index=edge_index,y=label,train_mask=train_mask,val_mask=val_mask,test_mask=test_mask)
        graph_dgl.ndata['train_mask']=train_mask
        graph_dgl.ndata['val_mask']=val_mask
        graph_dgl.ndata['test_mask']=test_mask

    elif dataset=='blogcatalog':
        bc_path = f'../bc.pt'
        graph_pyg = torch.load(bc_path)[0]
        graph_pyg.edge_index = graph_pyg.edge_indexes[0]
        graph_dgl = pyg_data_to_dgl_graph(graph_pyg)

        label = graph_dgl.ndata['label'].long()
        index = list(range(len(label)))
        train_mask, val_mask, test_mask = splitDataset(index,label,train_ratio)

        graph_pyg.train_mask=train_mask
        graph_pyg.val_mask=val_mask
        graph_pyg.test_mask=test_mask
        graph_dgl.ndata['train_mask']=train_mask
        graph_dgl.ndata['val_mask']=val_mask
        graph_dgl.ndata['test_mask']=test_mask
    else:
        print(f'{dataset}-graph_dgl not exist')
        return None

    
    color_print(f'{dataset}_graph_dgl: ')
    print(f'{graph_dgl}')
    color_print(f'{dataset}_graph_pyg: ')
    print(f'{graph_pyg}')
    color_print(f'train_val_test split')
    print(f'train_num: {train_mask.sum()}; val_num: {val_mask.sum()}; test_num: {test_mask.sum()}')

    return graph_dgl, graph_pyg, train_mask, val_mask, test_mask

def mergeGraphDataList(args, graph_pyg, syn_relation_dict):
    # 初始化合并后的组件
    x, y, new_x, edge_index = [], [], [], []
    adj_matrices = []
    ddpm_train_mask, ddpm_val_mask, ddpm_test_mask = [], [], []
    
    cum_num_nodes = 0  # 累积节点数，用于更新edge_index

    new_x_ssupgcl=syn_relation_dict['graph_pyg_ssupgcl_new_x']

    for data in syn_relation_dict['syn_relation_list']:
    # for data in syn_relation_dict['new_data_list']:
        # 更新节点特征和标签
        x.append(data.x)
        new_x.append(data.new_x)
        y.append(data.y)
        
        # 更新edge_index
        current_edge_index = data.edge_index + cum_num_nodes
        edge_index.append(current_edge_index)
        
        # 更新邻接矩阵
        adj_matrices.append(data.adj)
        
        # 更新掩码
        ddpm_train_mask.append(data.train_mask)
        ddpm_val_mask.append(data.val_mask)
        ddpm_test_mask.append(data.test_mask)
        
        # 更新节点累计数
        cum_num_nodes += data.num_nodes

    node_per_subgraph=args.nodes_per_subgraph
    num_unselected=graph_pyg.x.shape[0] % node_per_subgraph
    num_unselected

    # 合并所有组件
    new_x = torch.cat(new_x, dim=0)
    new_x = torch.cat([new_x, new_x_ssupgcl[-num_unselected:]],dim=0)

    x = torch.cat(x, dim=0)
    x = torch.cat([x, graph_pyg.x[-num_unselected:]],dim=0)
    y = torch.cat(y, dim=0)
    y = torch.cat([y, graph_pyg.y[-num_unselected:]],dim=0)
    edge_index = torch.cat(edge_index, dim=1)
    # edge_index = torch.cat([edge_index, ddpm_yelp_unselected_edge_index], dim=1)
    edge_index = torch.cat([edge_index, syn_relation_dict['unselected_edge_index']], dim=1)
    # edge_index = torch.cat([edge_index, syn_relation_dict['yelp_unselected_edge_index']], dim=1)
    ddpm_train_mask = torch.cat(ddpm_train_mask, dim=0)
    ddpm_train_mask = torch.cat([ddpm_train_mask, graph_pyg.train_mask[-num_unselected:]], dim=0)
    ddpm_val_mask = torch.cat(ddpm_val_mask, dim=0)
    ddpm_val_mask = torch.cat([ddpm_val_mask, graph_pyg.val_mask[-num_unselected:]], dim=0)
    ddpm_test_mask = torch.cat(ddpm_test_mask, dim=0)
    ddpm_test_mask = torch.cat([ddpm_test_mask, graph_pyg.test_mask[-num_unselected:]], dim=0)

    # 创建新的Data对象
    if args.dataset=='blogcatalog' and args.GuiDDPM_sample_with_guidance: # blogcatalog 要降维
        merged_data = Data(x=new_x, edge_index=edge_index, y=y, train_mask=ddpm_train_mask, val_mask=ddpm_val_mask, test_mask=ddpm_test_mask)
    else:
        merged_data = Data(x=x, edge_index=edge_index, y=y, train_mask=ddpm_train_mask, val_mask=ddpm_val_mask, test_mask=ddpm_test_mask)
    
    return merged_data
    
def GDCAugment(graph_pyg_type, avg_degree):
    # 应用 GDC 转换
    gdc = GDC(self_loop_weight=1, normalization_in='sym',
            normalization_out='col', diffusion_kwargs=dict(method='ppr', alpha=0.05),
            sparsification_kwargs=dict(method='threshold', avg_degree=avg_degree), exact=True)
    graph_pyg_type.transform = gdc
    gdc_transformed_data = gdc(graph_pyg_type)  # 应用 GDC 转换
    
    return gdc_transformed_data


def data4WFusionTrain(graph_pyg, graph_syn, graph_gdc_list):
    new_edges_dict = {}
    new_edges_dict[('node', 'relation0', 'node')] = (graph_pyg.edge_index[0], graph_pyg.edge_index[1])
    new_edges_dict[('node', 'relation1', 'node')] = (graph_syn.edge_index[0], graph_syn.edge_index[1])

    for idx, graph_gdc in enumerate(graph_gdc_list, start=2):
        new_edges_dict[(f'node', f'relation{idx}', f'node')] = (graph_gdc.edge_index[0], graph_gdc.edge_index[1])

    # 设置节点总数
    num_nodes = graph_syn.x.shape[0]

    # 创建异构图时显式指定节点的数量
    graph_WFusion = dgl.heterograph(new_edges_dict, num_nodes_dict={'node': num_nodes})
    
    # 设置节点特征和标签
    graph_WFusion.nodes['node'].data['feature'] = graph_syn.x
    graph_WFusion.nodes['node'].data['label'] = graph_syn.y
    graph_WFusion.nodes['node'].data['train_mask'] = graph_syn.train_mask
    graph_WFusion.nodes['node'].data['test_mask'] = graph_syn.test_mask
    graph_WFusion.nodes['node'].data['val_mask'] = graph_syn.val_mask

    return graph_WFusion