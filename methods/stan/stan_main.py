import numpy as np
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from math import floor, ceil
from methods.stan.stan_2d import stan_2d_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score, average_precision_score


def to_pred(logits: torch.Tensor) -> list:
    with torch.no_grad():
        pred = F.softmax(logits, dim=1).cpu()
        pred = pred.argmax(dim=1)
    return pred.numpy().tolist()


def att_train_2d(
    x_train,
    y_train,
    x_test,
    y_test,
    num_classes: int = 2,
    epochs: int = 18,
    batch_size: int = 256,
    attention_hidden_dim: int = 150,
    lr: float = 3e-3,
    device: str = "cpu"
):
    model = stan_2d_model(
        time_windows_dim=x_train.shape[1],
        feat_dim=x_train.shape[2],
        num_classes=num_classes,
        attention_hidden_dim=attention_hidden_dim,
    )
    model.to(device)

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
        # print(confusion_matrix(true, pred))

    # feats_test = torch.from_numpy(
    #     x_test).to(dtype=torch.float32).to(device)
    # feats_test.transpose_(1, 2)
    # labels_test = torch.from_numpy(y_test).to(dtype=torch.long)
    feats_test = x_test
    labels_test = y_test

    batch_num_test = ceil(len(labels_test) / batch_size)
    with torch.no_grad():
        pred = []
        for batch in range(batch_num):
            optimizer.zero_grad()
            batch_mask = list(
                range(batch*batch_size, min((batch+1)*batch_size, len(labels_test))))
            output = model(feats_test[batch_mask])
            pred.extend(to_pred(output))

        true = labels_test.cpu().numpy()
        pred = np.array(pred)
        print(
            f"test set | auc: {roc_auc_score(true, pred):.4f}, F1: {f1_score(true, pred, average='macro'):.4f}, AP: {average_precision_score(true, pred):.4f}")
        # print(confusion_matrix(true, pred))


def stan_main(
    train_feature_dir,
    train_label_dir,
    test_feature_dir,
    test_label_dir,
    mode: str = "2d",
    num_classed: int = 2,
    epochs: int = 18,
    batch_size: int = 256,
    attention_hidden_dim: int = 150,
    lr: float = 0.003,
    device="cpu",
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

    # y_pred = np.zeros(shape=test_label.shape)
    if mode == "2d":
        att_train_2d(
            train_feature,
            train_label,
            test_feature,
            test_label,
            epochs=epochs,
            batch_size=batch_size,
            attention_hidden_dim=attention_hidden_dim,
            lr=lr,
            device=device
        )
