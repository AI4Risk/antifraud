# Script to prune stan model
import torch.nn.utils.prune as prune
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
import time
import os
from math import ceil
import torch.nn.functional as F

def to_pred(logits: torch.Tensor) -> list:
    with torch.no_grad():
        pred = F.softmax(logits, dim=1).cpu()
        pred = pred.argmax(dim=1)
    return pred.numpy().tolist()

# Count parameters after pruning
def count_nonzero_parameters(model):
    nonzero_params = 0
    for param in model.parameters():
        # Only count parameters with gradients (ignoring those without, like biases in certain configurations)
        if param.requires_grad:
            # Use torch's nonzero function and count the resulting tensor's size along dimension 0
            nonzero_params += torch.nonzero(param, as_tuple=False).size(0)
    return nonzero_params

def eval_model(model, x_test, y_test, batch_size, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    feats_test = x_test
    labels_test = y_test

    batch_num_test = ceil(len(labels_test) / batch_size)
    with torch.no_grad():
        pred = []
        for batch in range(batch_num_test):
            optimizer.zero_grad()
            batch_mask = list(
                range(batch*batch_size, min((batch+1)*batch_size, len(labels_test))))
            output = model(feats_test[batch_mask])
            pred.extend(to_pred(output))

        true = labels_test.cpu().numpy()
        pred = np.array(pred)
        print(
            f"test set | auc: {roc_auc_score(true, pred):.4f}, F1: {f1_score(true, pred, average='macro'):.4f}, AP: {average_precision_score(true, pred):.4f}")
       
def fine_tune(model, x_train, y_train, batch_size, lr, device, epochs):
    nume_feats = x_train
    labels = y_train

    nume_feats.requires_grad = False
    labels.requires_grad = False

    nume_feats.to(device)
    labels = labels.to(device)

    # anti label imbalance
    unique_labels, counts = torch.unique(labels, return_counts=True)
    weights = (1 / counts)*len(labels)/len(unique_labels)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = torch.nn.CrossEntropyLoss(weights)

    batch_num = ceil(len(labels) / batch_size)
    for epoch in range(epochs):

        loss = 0.
        pred = []

        for batch in (range(batch_num)):
            optimizer.zero_grad()

            batch_mask = list(
                range(batch*batch_size, min((batch+1)*batch_size, len(labels))))

            output = model(nume_feats[batch_mask])

            batch_loss = loss_func(output, labels[batch_mask])
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss.item()
            # print(to_pred(output))
            pred.extend(to_pred(output))

        true = labels.cpu().numpy()
        pred = np.array(pred)
        print(
            f"Epoch: {epoch}, loss: {(loss / batch_num):.4f}, auc: {roc_auc_score(true, pred):.4f}, F1: {f1_score(true, pred, average='macro'):.4f}, AP: {average_precision_score(true, pred):.4f}")
