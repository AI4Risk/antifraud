import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from math import floor, ceil
from methods.stan import stan_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, f1_score, average_precision_score
from torch.nn.utils import prune
import os
import io


def to_pred(logits: torch.Tensor) -> list:
    with torch.no_grad():
        pred = F.softmax(logits, dim=1).cpu()
        pred = pred.argmax(dim=1)
    return pred.numpy().tolist()

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
    
    return model


def eval_model(model, x_test, y_test, batch_size, lr):
    feats_test = x_test
    labels_test = y_test

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    batch_num_test = ceil(len(labels_test) / batch_size)
    with torch.no_grad():
        pred = []
        start = datetime.now()
        for batch in range(batch_num_test):
            optimizer.zero_grad()
            batch_mask = list(
                range(batch*batch_size, min((batch+1)*batch_size, len(labels_test))))
            output = model(feats_test[batch_mask])
            pred.extend(to_pred(output))

        end = datetime.now()
        inference_time = end - start
        print(f"Time taken for inference: {inference_time}")

        true = labels_test.cpu().numpy()
        pred = np.array(pred)
        print(
            f"test set | auc: {roc_auc_score(true, pred):.4f}, F1: {f1_score(true, pred, average='macro'):.4f}, AP: {average_precision_score(true, pred):.4f}")
        cm = confusion_matrix(true, pred)
        cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legitimate", "Fraud"])
        fig, ax = plt.subplots(figsize=(10, 10))
        fig.suptitle('STAN Confusion Matrix')
        cm_disp.plot(ax=ax)
        fig.show()

def create_model(x, num_classes, attention_hidden_dim):
    return stan_model(
        time_windows_dim=x.shape[1],
        spatio_windows_dim=x.shape[2],
        feat_dim=x.shape[3],
        num_classes=num_classes,
        attention_hidden_dim=attention_hidden_dim,
    )

# Count parameters after pruning
def count_nonzero_parameters(model):
    nonzero_params = 0
    for param in model.parameters():
        # Only count parameters with gradients (ignoring those without, like biases in certain configurations)
        if param.requires_grad:
            # Use torch's nonzero function and count the resulting tensor's size along dimension 0
            nonzero_params += torch.nonzero(param, as_tuple=False).size(0)
    return nonzero_params
        

def stan_train(
    train_feature_dir,
    train_label_dir,
    test_feature_dir,
    test_label_dir,
    save_path: str,
    num_classes: int = 2,
    mode: str = "3d",
    epochs: int = 18,
    batch_size: int = 256,
    attention_hidden_dim: int = 150,
    lr: float = 3e-3,
    device: str = "cpu"
):
    x_train = torch.from_numpy(np.load(train_feature_dir, allow_pickle=True)).to(
        dtype=torch.float32).to(device)
    y_train = torch.from_numpy(np.load(train_label_dir, allow_pickle=True)).to(
        dtype=torch.long).to(device)
    x_test = torch.from_numpy(np.load(test_feature_dir, allow_pickle=True)).to(
        dtype=torch.float32).to(device)
    y_test = torch.from_numpy(np.load(test_label_dir, allow_pickle=True)).to(
        dtype=torch.long).to(device)

    # y_pred = np.zeros(shape=test_label.shape)
    if mode == "3d":
        model = create_model(x_train, num_classes, attention_hidden_dim)
        model.to(device)

        model = fine_tune(model, x_train, y_train, batch_size, lr, device, epochs)
        eval_model(model, x_test, y_test, batch_size, lr)

    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved at {save_path}")

def stan_test(
    test_feature_dir,
    test_label_dir,
    path: str,
    num_classes: int = 2,
    batch_size: int = 256,
    attention_hidden_dim: int = 150,
    lr: float = 3e-3,
    device: str = "cpu",
):
    x_test = torch.from_numpy(np.load(test_feature_dir, allow_pickle=True)).to(
        dtype=torch.float32).to(device)
    y_test = torch.from_numpy(np.load(test_label_dir, allow_pickle=True)).to(
        dtype=torch.long).to(device)

    print("Loading model: ", path)

    model = create_model(x_test, num_classes, attention_hidden_dim)

    if device == "cpu":
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(path))
    model.to(device)

    eval_model(model, x_test, y_test, batch_size, lr)
        

def stan_prune(
        train_feature_dir,
        train_label_dir,
        test_feature_dir,
        test_label_dir,
        load_path: str,
        batch_size=256,
        attention_hidden_dim=150,
        lr=3e-3,
        num_classes=2,
        device='cpu',
        fine_tune_epochs=4,
        prune_iter=3,
        prune_perct=0.1
    ):

    x_train = torch.from_numpy(np.load(train_feature_dir, allow_pickle=True)).to(
        dtype=torch.float32).to(device)
    y_train = torch.from_numpy(np.load(train_label_dir, allow_pickle=True)).to(
        dtype=torch.long).to(device)
    x_test = torch.from_numpy(np.load(test_feature_dir, allow_pickle=True)).to(
        dtype=torch.float32).to(device)
    y_test = torch.from_numpy(np.load(test_label_dir, allow_pickle=True)).to(
        dtype=torch.long).to(device)
    
    model = create_model(x_train, num_classes, attention_hidden_dim)

    if device == "cpu":
        model.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(load_path))
    model.to(device)

    print(f"Number of parameters in original model: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Iterative pruning
    tmp = prune_iter
    while prune_iter > 0:
        prune_iter -= 1

        # Prune the Conv3d layer
        prune.l1_unstructured(model.conv, name='weight', amount=prune_perct)

        # Prune each Linear layer within 'linears'
        for name, module in model.named_children():
            if name == 'linears':
                for name_seq, module_seq in module.named_children():
                    if isinstance(module_seq, torch.nn.Linear):
                        prune.l1_unstructured(module_seq, name='weight', amount=prune_perct)


        # Retrain to regain lost accuracy
        model = fine_tune(model, x_train, y_train, batch_size, lr, device, fine_tune_epochs)
        eval_model(model, x_test, y_test, batch_size, lr)
        print("*" * 3 + f" Prune iteration {tmp - prune_iter} complete " + "*" * 3)
    
    # Make pruning permanent
    for name, module in model.named_modules():
        for hook in list(module._forward_pre_hooks.values()):
            if isinstance(hook, torch.nn.utils.prune.BasePruningMethod):
                prune.remove(module, 'weight')

    print(f"Number of parameters in pruned model: {count_nonzero_parameters(model)} parameters")

    # save model
    torch.save(model.state_dict(), load_path.replace('.pt', '_pruned.pt'))

def stan_quant(
        train_feature_dir,
        train_label_dir,
        test_feature_dir,
        test_label_dir,
        load_path: str,
        device='cpu',
        num_classes=2,
        attention_hidden_dim=150,
):
    print(device)
    x_train = torch.from_numpy(np.load(train_feature_dir, allow_pickle=True)).to(
        dtype=torch.float32).to(device)
    y_train = torch.from_numpy(np.load(train_label_dir, allow_pickle=True)).to(
        dtype=torch.long).to(device)
    x_test = torch.from_numpy(np.load(test_feature_dir, allow_pickle=True)).to(
        dtype=torch.float32).to(device)
    y_test = torch.from_numpy(np.load(test_label_dir, allow_pickle=True)).to(
        dtype=torch.long).to(device)
    
    model = create_model(x_train, num_classes, attention_hidden_dim)
    if device == "cpu":
        model.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(load_path))
    model.to(device)

    # Check supported backends
    supported_backends = torch.backends.quantized.supported_engines

    # Print available backends
    print("Supported quantization backends: ", supported_backends)

    # Priority order of backends
    # ARM CPUs need qnnpack backend
    preferred_order = ['fbgemm', 'qnnpack']
    for backend in preferred_order:
        if backend in supported_backends:
            # Set the supported backend
            print(backend, "is supported")
            torch.backends.quantized.engine = backend
            save_backend = backend
            break

    # For Conv3d layers, use static quantization
    # Linear layers will use dynamic which has no qconfig
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv3d):
            module.qconfig = torch.ao.quantization.get_default_qconfig(save_backend)

    # Wrap the model with quant observers
    model = torch.ao.quantization.QuantWrapper(model)

    # Prepare
    torch.ao.quantization.prepare(model, inplace=True)

    # calibrate the model
    model.eval()
    with torch.no_grad():
        model(x_train)

    # convert the model to a quantized model
    quant_model = torch.ao.quantization.convert(model)

    #### Dynamic quantization for Linear layers
    # this and the saving of the model doesn't work on ARM CPUs
    final_quant_model = torch.quantization.quantize_dynamic(
        quant_model, 
        {torch.nn.Linear}, 
        dtype=torch.qint8
    )

    # Save the quantized model
    torch.save(final_quant_model.state_dict(), load_path.replace('.pt', '_quantized.pt'))