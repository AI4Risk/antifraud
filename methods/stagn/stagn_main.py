import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import dgl
import torch.nn.functional as F
from scipy.stats import zscore
from methods.stagn.stagn_2d import stagn_2d_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from feature_engineering.data_engineering import span_data_2d


def to_pred(logits: torch.Tensor) -> list:
    with torch.no_grad():
        pred = F.softmax(logits, dim=1).cpu()
        pred = pred.argmax(dim=1)
    return pred.numpy().tolist()


def stagn_train_2d(
    features,
    labels,
    train_idx,
    test_idx,
    g,
    num_classes: int = 2,
    epochs: int = 18,
    attention_hidden_dim: int = 150,
    lr: float = 3e-3,
    device: str = "cpu"
):
    g = g.to(device)
    model = stagn_2d_model(
        time_windows_dim=features.shape[2],
        feat_dim=features.shape[1],
        num_classes=num_classes,
        attention_hidden_dim=attention_hidden_dim,
        g=g,
        device=device
    )
    model.to(device)

    features = torch.from_numpy(features).to(device)
    features.transpose_(1, 2)
    labels = torch.from_numpy(labels).to(device)

    unique_labels, counts = torch.unique(labels, return_counts=True)
    weights = (1 / counts)*len(labels)/len(unique_labels)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = torch.nn.CrossEntropyLoss(weights)

    for epoch in range(epochs):
        optimizer.zero_grad()

        out = model(features, g)
        loss = loss_func(out[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        pred = to_pred(out[train_idx])
        true = labels[train_idx].cpu().numpy()
        pred = np.array(pred)
        print(f"Epoch: {epoch}, loss: {loss:.4f}, auc: {roc_auc_score(true, pred):.4f}, F1: {f1_score(true, pred, average='macro'):.4f}, AP: {average_precision_score(true, pred):.4f}")

    with torch.no_grad():
        out = model(features, g)
        pred = to_pred(out[test_idx])
        true = labels[test_idx].cpu().numpy()
        pred = np.array(pred)
        print(
            f"test set | auc: {roc_auc_score(true, pred):.4f}, F1: {f1_score(true, pred, average='macro'):.4f}, AP: {average_precision_score(true, pred):.4f}")


def stagn_main(
    features,
    labels,
    test_ratio,
    g,
    mode: str = "2d",
    epochs: int = 18,
    attention_hidden_dim: int = 150,
    lr: float = 0.003,
    device="cpu",
):
    train_idx, test_idx = train_test_split(
        np.arange(features.shape[0]), test_size=test_ratio, stratify=labels)

    # y_pred = np.zeros(shape=test_label.shape)
    if mode == "2d":
        stagn_train_2d(
            features,
            labels,
            train_idx,
            test_idx,
            g,
            epochs=epochs,
            attention_hidden_dim=attention_hidden_dim,
            lr=lr,
            device=device
        )
    else:
        raise NotImplementedError("Not supported mode.")


def load_stagn_data(args: dict):
    # load S-FFSD dataset for base models
    data_path = "data/S-FFSD.csv"
    feat_df = pd.read_csv(data_path)
    train_size = 1 - args['test_size']
    method = args['method']
    # ICONIP16 & AAAI20 requires higher dimensional data
    if os.path.exists("data/features.npy"):
        features, labels = np.load(
            "data/features.npy"), np.load("data/labels.npy")
    else:
        features, labels = span_data_2d(feat_df)
        np.save("data/features.npy", features)
        np.save("data/labels.npy", labels)

    sampled_df = feat_df[feat_df['Labels'] != 2]
    sampled_df = sampled_df.reset_index(drop=True)

    all_nodes = pd.concat([sampled_df['Source'], sampled_df['Target']]).unique()
    encoder = LabelEncoder().fit(all_nodes)  
    encoded_source = encoder.transform(sampled_df['Source'])
    encoded_tgt = encoder.transform(sampled_df['Target'])  

    loc_enc = OneHotEncoder()
    loc_feature = np.array(loc_enc.fit_transform(
        sampled_df['Location'].to_numpy()[:, np.newaxis]).todense())
    loc_feature = np.hstack(
        [zscore(sampled_df['Amount'].to_numpy())[:, np.newaxis], loc_feature])

    g = dgl.DGLGraph()
    g.add_edges(encoded_source, encoded_tgt, data={
                "feat": torch.from_numpy(loc_feature).to(torch.float32)})
    # g = dgl.add_self_loop(g)
    return features, labels, g
