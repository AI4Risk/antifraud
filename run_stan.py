import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from config import Config
from feature_engineering.data_engineering import data_engineer_benchmark, span_data_2d, span_data_3d
from methods.stan.stan_main import stan_main
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import pickle
import dgl
from scipy.io import loadmat
import yaml

logger = logging.getLogger(__name__)


def base_load_data(args: dict):
    # load S-FFSD dataset for base models
    data_path = "data/S-FFSD.csv"
    feat_df = pd.read_csv(data_path)
    train_size = 1 - args['test_size']
    method = args['method']
    # for ICONIP16 & AAAI20
    if os.path.exists("data/tel_3d.npy"):
        return
    features, labels = span_data_3d(feat_df)

    trf, tef, trl, tel = train_test_split(
        features, labels, train_size=train_size, stratify=labels, shuffle=True)
    trf_file, tef_file, trl_file, tel_file = args['trainfeature'], args[
        'testfeature'], args['trainlabel'], args['testlabel']

    np.save(trf_file, trf)
    np.save(tef_file, tef)
    np.save(trl_file, trl)
    np.save(tel_file, tel)
    return


def main(args):
    base_load_data(args)
    stan_main(
        args['trainfeature'],
        args['trainlabel'],
        args['testfeature'],
        args['testlabel'],
        mode='3d',
        epochs=args['epochs'],
        batch_size=args['batch_size'],
        attention_hidden_dim=args['attention_hidden_dim'],
        lr=args['lr'],
        device=args['device']
    )


if __name__ == "__main__":
    with open("config/stan_cfg.yaml") as file:
        args = yaml.safe_load(file)
    args['method'] = 'stan'

    main(args)