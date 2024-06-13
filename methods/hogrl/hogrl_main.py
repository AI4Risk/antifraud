import random
import torch
from sklearn.model_selection import train_test_split
from .hogrl_model import *
from .hogrl_utils import *
# from .hogrl_utils import *
import numpy as np
import random as rd
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, average_precision_score

def test(idx_eval, y_eval, gnn_model, feat_data, edge_indexs):
    gnn_model.eval()
    logits, _ = gnn_model(feat_data, edge_indexs)
    x_softmax = torch.exp(logits).cpu().detach()
    positive_class_probs = x_softmax[:, 1].numpy()[np.array(idx_eval)]
    auc_score = roc_auc_score(np.array(y_eval), np.array(positive_class_probs))
    ap_score = average_precision_score(np.array(y_eval), np.array(positive_class_probs))
    label_prob = (np.array(positive_class_probs) >= 0.5).astype(int)
    f1_score_val = f1_score(np.array(y_eval), label_prob, average='macro')
    g_mean = calculate_g_mean(np.array(y_eval), label_prob)

    return auc_score, ap_score, f1_score_val, g_mean

def hogrl_main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print('loading data...')
    prefix = os.path.join(os.path.dirname(__file__), "..", "..", "data/")
    edge_indexs,feat_data,labels = load_data(args['dataset'],args['layers_tree'], prefix)
    
    np.random.seed(args['seed'])
    random.seed(args['seed'])
    
    if args['dataset'] == 'yelp' or args['dataset'] == 'CCFD':
        assert args['dataset'] != 'CCFD', 'Due to confidentiality agreements, we are unable to provide the CCFD data.'
        
        index = list(range(len(labels)))
        idx_train_val, idx_test, y_train_val, y_test = train_test_split(index, labels, stratify=labels, test_size=args['test_size'], random_state=2, shuffle=True)
        idx_train, idx_val, y_train, y_val = train_test_split(idx_train_val, y_train_val, stratify=y_train_val, test_size=args['val_size'], random_state=2, shuffle=True)
    elif args['dataset'] == 'amazon':
        # 0-3304 are unlabeled nodes
        index = list(range(3305, len(labels)))
        idx_train_val, idx_test, y_train_val, y_test = train_test_split(index, labels[3305:], stratify=labels[3305:], test_size=args['test_size'], random_state=2, shuffle=True)
        idx_train, idx_val, y_train, y_val = train_test_split(idx_train_val, y_train_val, stratify=y_train_val, test_size=args['val_size'], random_state=2, shuffle=True)
    
    train_pos, train_neg = pos_neg_split(idx_train, y_train)

    gnn_model = multi_HOGRL_Model(feat_data.shape[1],2,len(edge_indexs),args['emb_size'],args['drop_rate'],args['weight'],args['layers'],args['layers_tree']).to(device)
    for edge_index in edge_indexs:
        edge_index[0] = edge_index[0].to(device)
        edge_index[1] = [tensor.to(device) for tensor in edge_index[1]]
            
    # labels = torch.tensor(labels).to(device)
    feat_data = torch.tensor(feat_data).float().to(device)

    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.005, weight_decay=5e-5)
    batch_size = args['batch_size']
    
    best_val_auc = 0.0
    best_model_state = None
    
    print('training...')
    for epoch in range(args['num_epochs']):
        gnn_model.train()
        loss = 0
        # randomly under-sampling negative nodes for each epoch
        sampled_idx_train = undersample(train_pos, train_neg, scale=1)
        rd.shuffle(sampled_idx_train)

        num_batches = int(len(sampled_idx_train) / batch_size ) + 1
        for batch in range(num_batches):
            i_start = batch * batch_size
            i_end = min((batch + 1) * batch_size, len(sampled_idx_train))
            batch_nodes = sampled_idx_train[i_start:i_end]
            batch_label = torch.tensor(labels[np.array(batch_nodes)]).long().to(device)
            optimizer.zero_grad()
            out,_ = gnn_model(feat_data,edge_indexs)
            batch_nodes_tensor = torch.tensor(batch_nodes, dtype=torch.long, device=device)
            loss = F.nll_loss(out[batch_nodes_tensor], batch_label)

            # loss = F.nll_loss(out[np.array(batch_nodes)], batch_label)
            loss.backward()
            optimizer.step()
            loss += loss.item()
            #print(loss.item())

        if epoch % 10 == 9: # validate every 10 epochs 
            val_auc, val_ap, val_f1, val_g_mean = test(idx_val, y_val, gnn_model, feat_data, edge_indexs)
            print(f'Epoch: {epoch}, Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}, Val F1: {val_f1:.4f}, Val G-mean: {val_g_mean:.4f}')
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = gnn_model.state_dict() 

    # test
    gnn_model.load_state_dict(best_model_state)  
    test_auc, test_ap, test_f1, test_g_mean = test(idx_test, y_test, gnn_model, feat_data, edge_indexs)
    print(f'Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}, '
        f'Test F1: {test_f1:.4f}, Test G-mean: {test_g_mean:.4f}')

    out,embedding = gnn_model(feat_data,edge_indexs)
    
    print('generating embedding visualization...')
    Visualization(labels,embedding.cpu().detach(), prefix)
