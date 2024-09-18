import pickle
import random as rd
import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
import copy as cp
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, average_precision_score,confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import torch
import copy as cp
import os
from sklearn.metrics import confusion_matrix

filelist = {
    'amz_upu': 'amz_upu_adjlists.pickle',
    'amz_usu': 'amz_usu_adjlists.pickle',
    'amz_uvu': 'amz_uvu_adjlists.pickle',
    'yelp_rsr': 'yelp_rsr_adjlists.pickle',
    'yelp_rtr': 'yelp_rtr_adjlists.pickle',
    'yelp_rur': 'yelp_rur_adjlists.pickle'
}

file_matrix_prefix = {
    'amz_upu': 'amazon_upu_matrix_',
    'amz_usu': 'amazon_usu_matrix_',
    'amz_uvu': 'amazon_uvu_matrix_',
    'yelp_rsr': 'yelpnet_rsr_matrix_decompision_',
    'yelp_rtr': 'yelpnet_rtr_matrix_decompision_',
    'yelp_rur': 'yelpnet_rur_matrix_decompision_'
}


def calculate_g_mean(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)

    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    

    g_mean = np.sqrt(sensitivity * specificity)
    return g_mean


def dict_to_edge_index(edge_dict):
    source_nodes = []
    target_nodes = []

    for src, targets in edge_dict.items():
        for target in targets:
            source_nodes.append(src)
            target_nodes.append(target)

    edge_index = [source_nodes, target_nodes]
    return torch.LongTensor(edge_index)



def numpy_array_to_edge_index(np_array):

    assert np_array.ndim == 2 and np_array.shape[0] == np_array.shape[1], "Input must be a square matrix."

    # Find the indices of nonzero elements (edges)
    rows, cols = np.nonzero(np_array)

    # Stack them to create edge index
    edge_index = np.vstack((rows, cols))

    # Convert to PyTorch tensor
    edge_index_tensor = torch.from_numpy(edge_index).long()

    return edge_index_tensor

def load_data(data,k=2, prefix=''):
    """
    Load graph, feature, and label given dataset name
    """
    pickle_file = {}
    matrix_prefix = {}
    for key in filelist: # update the file paths
        pickle_file[key] = os.path.join(prefix, filelist[key])
        matrix_prefix[key] = os.path.join(prefix, file_matrix_prefix[key])
    
    if data == 'yelp':
        data_file = loadmat(os.path.join(prefix, 'YelpChi.mat'))
        labels = data_file['label'].flatten()
        feat_data = data_file['features'].todense().A
        
        with open(pickle_file['yelp_rur'], 'rb') as file:
            relation1 = pickle.load(file)
        file.close()
        relation1 = dict_to_edge_index(relation1)
        relation1_tree = []
        for i in range(1,k+1):
            file_name = '{}{}.pkl'.format(matrix_prefix['yelp_rur'], i)
            with open(file_name,'rb') as file:
                tree = pickle.load(file)
            file.close()
            relation1_tree.append(numpy_array_to_edge_index(tree))
        with open(pickle_file['yelp_rtr'], 'rb') as file:
            relation2 = pickle.load(file)
        file.close()
        relation2 = dict_to_edge_index(relation2)
        relation2_tree = []
        for i in range(1,k+1):
            file_name = '{}{}.pkl'.format(matrix_prefix['yelp_rtr'], i)
            with open(file_name,'rb') as file:
                tree = pickle.load(file)
            file.close()
            relation2_tree.append(numpy_array_to_edge_index(tree))
        with open(pickle_file['yelp_rsr'], 'rb') as file:
            relation3 = pickle.load(file)
        file.close()
        relation3 = dict_to_edge_index(relation3)
        relation3_tree = []
        for i in range(1,k+1):
            file_name = '{}{}.pkl'.format(matrix_prefix['yelp_rsr'], i)
            with open(file_name,'rb') as file:
                tree = pickle.load(file)
            file.close()
            relation3_tree.append(numpy_array_to_edge_index(tree))
        return [[relation1,relation1_tree],[relation2,relation2_tree],[relation3,relation3_tree]],feat_data,labels
    elif data == 'amazon':
        data_file = loadmat(os.path.join(prefix, 'Amazon.mat'))
        labels = data_file['label'].flatten()
        feat_data = data_file['features'].todense().A
        # load the preprocessed adj_lists
        with open(pickle_file['amz_upu'], 'rb') as file:
            relation1 = pickle.load(file)
        file.close()
        relation1 = dict_to_edge_index(relation1)
        relation1_tree = []
        for i in range(1,k+1):
            file_name = '{}{}.pkl'.format(matrix_prefix['amz_upu'], i)
            with open(file_name,'rb') as file:
                tree = pickle.load(file)
            file.close()
            relation1_tree.append(numpy_array_to_edge_index(tree))
        with open(pickle_file['amz_usu'], 'rb') as file:
            relation2 = pickle.load(file)
        file.close()
        relation2 =  dict_to_edge_index(relation2)
        relation2_tree = []
        for i in range(1,k+1):
            file_name = '{}{}.pkl'.format(matrix_prefix['amz_usu'], i)
            with open(file_name,'rb') as file:
                tree = pickle.load(file)
            file.close()
            relation2_tree.append(numpy_array_to_edge_index(tree))
        with open(pickle_file['amz_uvu'], 'rb') as file:
            relation3 = pickle.load(file)
        file.close()
        relation3_tree = []
        for i in range(1,k+1):
            file_name = '{}{}.pkl'.format(matrix_prefix['amz_uvu'], i)
            with open(file_name,'rb') as file:
                tree = pickle.load(file)
            file.close()
            relation3_tree.append(numpy_array_to_edge_index(tree))
        relation3 = dict_to_edge_index(relation3)
        
        return [[relation1,relation1_tree],[relation2,relation2_tree],[relation3,relation3_tree]],feat_data,labels
    elif data=='CCFD':
        assert False,'CCFD dataset is secret, please contact the author for the dataset.'
        
        data_file= loadmat(os.path.join(prefix, 'CCFD.mat'))
        labels = data_file['labels'].flatten()
        feat_data = data_file['features']
        with open('../data/net_source_CCFD.pickle', 'rb') as file:
            relation1 = pickle.load(file)
        file.close()
        relation1 = dict_to_edge_index(relation1)
        relation1_tree = []
        for i in range(1,k+1):
            file_name = f'../data/CCFD_r1_matrix_{k}.pkl'
            with open(file_name,'rb') as file:
                tree = pickle.load(file)
            file.close()
            relation1_tree.append(numpy_array_to_edge_index(tree))	
        return [[relation1,relation1_tree]],feat_data,labels
        


def Visualization(labels, embedding, prefix):
    train_pos, train_neg = pos_neg_split(list(range(len(labels))), labels)
    sampled_idx_train = undersample(train_pos, train_neg, scale=1)
    tsne = TSNE(n_components=2, random_state=43)
    sampled_idx_train = np.array(sampled_idx_train)
    sampled_idx_train = np.random.choice(sampled_idx_train, size=5000, replace=True)
    ps = embedding[sampled_idx_train]
    ls = labels[sampled_idx_train]

    X_reduced = tsne.fit_transform(ps)

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X_reduced)
    print(X_scaled.shape)
    
    plt.figure(figsize=(8, 8))

    plt.scatter(X_scaled[ls == 0, 0], X_scaled[ls == 0, 1], c='#14517C', label='Label 0', s=3)

    plt.scatter(X_scaled[ls == 1, 0], X_scaled[ls == 1, 1], c='#FA7F6F', label='Label 1', s=3)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    plt.xticks([])
    plt.yticks([])

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    filepath = os.path.join(prefix, 'HOGRL.png')
    plt.savefig(filepath)
    plt.show()
    
def normalize(mx):
    """
        Row-normalize sparse matrix
        Code from https://github.com/williamleif/graphsage-simple/
    """
    rowsum = np.array(mx.sum(1)) + 0.01
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx



def pos_neg_split(nodes, labels):
    """
    Find positive and negative nodes given a list of nodes and their labels
    :param nodes: a list of nodes
    :param labels: a list of node labels
    :returns: the spited positive and negative nodes
    """
    pos_nodes = []
    neg_nodes = cp.deepcopy(nodes)
    aux_nodes = cp.deepcopy(nodes)
    for idx, label in enumerate(labels):
        if label == 1:
            pos_nodes.append(aux_nodes[idx])
            neg_nodes.remove(aux_nodes[idx])

    return pos_nodes, neg_nodes


def undersample(pos_nodes, neg_nodes, scale=1):
    """
    Under-sample the negative nodes
    :param pos_nodes: a list of positive nodes
    :param neg_nodes: a list negative nodes
    :param scale: the under-sampling scale
    :return: a list of under-sampled batch nodes
    """

    aux_nodes = cp.deepcopy(neg_nodes)
    aux_nodes = rd.sample(aux_nodes, k=int(len(pos_nodes)*scale))
    batch_nodes = pos_nodes + aux_nodes

    return batch_nodes

def calculate_g_mean(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sensitivities = []
    for i in range(len(cm)):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
        sensitivities.append(sensitivity)
    g_mean = np.prod(sensitivities) ** (1 / len(sensitivities))
    return g_mean