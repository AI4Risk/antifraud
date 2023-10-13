from math import floor, ceil
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score, average_precision_score
import os
from .mcnn_model import mcnn, to_pred


def mcnn_main(
    train_feature_dir,
    train_label_dir,
    test_feature_dir,
    test_label_dir,
    epochs=30,
    batch_size=512,
    lr=1e-3,
    device="cpu"
):
    train_feature = torch.from_numpy(np.load(train_feature_dir, allow_pickle=True)).to(
        dtype=torch.float32).to(device)
    train_feature.transpose_(1, 2)
    train_label = torch.from_numpy(np.load(train_label_dir, allow_pickle=True)).to(
        dtype=torch.long).to(device)
    test_feature = torch.from_numpy(np.load(test_feature_dir, allow_pickle=True)).to(
        dtype=torch.float32).to(device)
    test_feature.transpose_(1, 2)
    test_label = torch.from_numpy(np.load(test_label_dir, allow_pickle=True)).to(
        dtype=torch.long).to(device)

    model = mcnn()
    model.to(device)

    # anti label imbalance
    unique_labels, counts = torch.unique(train_label, return_counts=True)
    weights = (1 / counts)*len(train_label)/len(unique_labels)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = torch.nn.CrossEntropyLoss(weights)

    batch_num = ceil(len(train_label) / batch_size)
    for epoch in range(epochs):

        loss = 0.
        pred = []

        for batch in (range(batch_num)):
            optimizer.zero_grad()

            batch_mask = list(
                range(batch*batch_size, min((batch+1)*batch_size, len(train_label))))

            output = model(train_feature[batch_mask])

            batch_loss = loss_func(output, train_label[batch_mask])
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss.item()
            # print(to_pred(output))
            pred.extend(to_pred(output))

        true = train_label.cpu().numpy()
        pred = np.array(pred)
        print(
            f"Epoch: {epoch}, loss: {(loss / batch_num):.4f}, auc: {roc_auc_score(true, pred):.4f}, F1: {f1_score(true, pred, average='macro'):.4f}, AP: {average_precision_score(true, pred):.4f}")
        # print(confusion_matrix(true, pred))

    batch_num_test = ceil(len(test_label) / batch_size)
    with torch.no_grad():
        pred = []
        for batch in (range(batch_num_test)):
            optimizer.zero_grad()
            batch_mask = list(
                range(batch*batch_size, min((batch+1)*batch_size, len(test_label))))
            output = model(
                test_feature[batch_mask])
            pred.extend(to_pred(output))

        true = test_label.cpu().numpy()
        pred = np.array(pred)
        print(
            f"test set | auc: {roc_auc_score(true, pred):.4f}, F1: {f1_score(true, pred, average='macro'):.4f}, AP: {average_precision_score(true, pred):.4f}")
