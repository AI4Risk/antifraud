import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from feature_engineering.data_engineering import span_data_3d
from methods.stan_main import stan_test, stan_train, stan_prune, stan_quant
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

logger = logging.getLogger(__name__)


def base_load_data(args: dict):
    # load S-FFSD dataset for base models
    data_path = "data/S-FFSD.csv"
    feat_df = pd.read_csv(data_path)
    train_size = 1 - args['test_size']
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

    if args['mode'] == 'train':
        stan_train(
            args['trainfeature'], 
            args['trainlabel'], 
            args['testfeature'],
            args['testlabel'],
            args['train_save_path'],
            num_classes=2,
            mode='3d',
            epochs=args['epochs'],
            batch_size=args['batch_size'],
            attention_hidden_dim=args['attention_hidden_dim'],
            lr=args['lr'],
            device=args['device']
        )

    elif args['mode'] == 'test':
        stan_test(
            args['testfeature'],
            args['testlabel'],
            args['train_save_path'],
            num_classes=2,
            batch_size=args['batch_size'],
            attention_hidden_dim=args['attention_hidden_dim'],
            lr=args['lr'],
            device=args['device'],
        )

    elif args['mode'] == 'prune':
        stan_prune(
            args['trainfeature'],
            args['trainlabel'],
            args['testfeature'],
            args['testlabel'],
            args['train_save_path'],
            batch_size=args['batch_size'],
            attention_hidden_dim=args['attention_hidden_dim'],
            lr=args['lr'],
            num_classes=2,
            device=args['device'],
            fine_tune_epochs= 3,
            prune_iter=args['prune_iter'],
            prune_perct=0.1
        )

    elif args['mode'] == 'quantize':
        stan_quant(
            args['trainfeature'],
            args['trainlabel'],
            args['testfeature'],
            args['testlabel'],
            args['train_save_path'],
            device=args['device'],
            num_classes=2,
            attention_hidden_dim=args['attention_hidden_dim']
        )

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument("--mode", default=str, help="Mode to run the script in: 'train' or 'test'")
    mode = vars(parser.parse_args())['mode']  # dict
    print(mode)

    with open("config/stan_cfg.yaml") as file:
        args = yaml.safe_load(file)
    args['method'] = 'stan'
    args['mode'] = mode

    main(args)