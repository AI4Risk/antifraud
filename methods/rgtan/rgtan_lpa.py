import copy


def load_lpa_subtensor(
    node_feat,  # (|all|, feat_dim)
    work_node_feat,
    neigh_feat: dict,
    neigh_padding_dict: dict,  # {"degree":6, ...}
    labels,  # (|all|,)
    seeds,  # (|batch|,)
    input_nodes,  # (|batch_all|,)
    device,
    blocks,
):
    """
    Put the input data into the device
    :param node_feat: the feature of input nodes
    :param work_node_feat: the feature of work nodes
    :param neigh_feat: neighborhood stat feature -> pd.DataFrame
    :param neigh_padding_dict: padding length of neighstat features
    :param labels: the labels of nodes
    :param seeds: the index of one batch data 
    :param input_nodes: the index of batch input nodes -> batch all size!!!
    :param device: where to train model
    :param blocks: dgl blocks
    """
    # masking to avoid label leakage
    if "1hop_riskstat" in neigh_feat.keys() and len(blocks) >= 2:
        # nei_hop1 = get_k_neighs(graph, seeds, 1)
        nei_hop1 = blocks[-2].dstdata['_ID']
        neigh_feat['1hop_riskstat'][nei_hop1] = 0

    if "2hop_riskstat" in neigh_feat.keys() and len(blocks) >= 3:
        # nei_hop2 = get_k_neighs(graph, seeds, 2)
        nei_hop2 = blocks[-3].dstdata['_ID']
        neigh_feat['2hop_riskstat'][nei_hop2] = 0

    batch_inputs = node_feat[input_nodes].to(device)
    batch_work_inputs = {i: work_node_feat[i][input_nodes].to(
        device) for i in work_node_feat if i not in {"labels"}}  # cat feats

    batch_neighstat_inputs = None

    if neigh_feat:
        batch_neighstat_inputs = {col: neigh_feat[col][input_nodes].to(
            device) for col in neigh_feat.keys()}

    batch_labels = labels[seeds].to(device)
    train_labels = copy.deepcopy(labels)
    propagate_labels = train_labels[input_nodes]  # (|input_nodes|,) 45324
    propagate_labels[:seeds.shape[0]] = 2
    return batch_inputs, batch_work_inputs, batch_neighstat_inputs, batch_labels, propagate_labels.to(device)
