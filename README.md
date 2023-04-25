# AntiFraud
A Financial Fraud Detection Framework.

Source codes implementation of papers:
- `MCNN`: Credit card fraud detection using convolutional neural networks, in ICONIP 2016. 
- `STAN`: Spatio-temporal attention-based neural network for credit card fraud detection, in AAAI2020
- `STAGN`: Graph Neural Network for Fraud Detection via Spatial-temporal Attention, in TKDE2020
- `GTAN`: Semi-supervised Credit Card Fraud Detection via Attribute-driven Graph Representation, in AAAI2023
- `RGTAN`: Enhancing Attribute-driven Fraud Detection with Risk-aware Graph Representation, 



## Usage

### Data processing
1. Run `unzip /data/Amazon.zip` and `unzip /data/YelpChi.zip` to unzip the datasets; 
2. Run `python feature_engineering/data_process.py
`
to pre-process all datasets needed in this repo.

### Training & Evalutaion
<!-- 
To use fraud detection baselines including GBDT, LSTM, etc., simply run

```
python main.py --method LSTM
python main.py  --method GBDT
```
You may change relevant configurations in `config/base_cfg.yaml`. -->

To test implementations of `MCNN`, `STAN` and `STAGN`, run
```
python main.py --method mcnn
python main.py --method stan
python main.py --method stagn
```
Configuration files can be found in `config/mcnn_cfg.yaml`, `config/stan_cfg.yaml` and `config/stagn_cfg.yaml`, respectively.

Models in `GTAN` and `RGTAN` can be run via:
```
python main.py --method gtan
python main.py --method rgtan
```
For specification of hyperparameters, please refer to `config/gtan_cfg.yaml` and `config/rgtan_cfg.yaml`.



### Data Description

There are three datasets, YelpChi, Amazon and S-FFSD, utilized for model experiments in this repository.

<!-- YelpChi and Amazon can be downloaded from [here](https://github.com/YingtongDou/CARE-GNN/tree/master/data) or [dgl.data.FraudDataset](https://docs.dgl.ai/api/python/dgl.data.html#fraud-dataset).

Put them in `/data` directory and run `unzip /data/Amazon.zip` and `unzip /data/YelpChi.zip` to unzip the datasets. -->

YelpChi and Amazon datasets are from [CARE-GNN](https://dl.acm.org/doi/abs/10.1145/3340531.3411903), whose original source data can be found in [this repository](https://github.com/YingtongDou/CARE-GNN/tree/master/data).

S-FFSD is a simulated & small version of finacial fraud semi-supervised dataset. Description of S-FFSD are listed as follows:
|Name|Type|Range|Note|
|--|--|--|--|
|Time|np.int32|from $\mathbf{0}$ to $\mathbf{N}$|$\mathbf{N}$ denotes the number of trasactions.  |
|Source|string|from $\mathbf{S_0}$ to $\mathbf{S}_{ns}$|$ns$ denotes the number of transaction senders.|
|Target|string|from $\mathbf{T_0}$  to $\mathbf{T}_{nt}$ | $nt$ denotes the number of transaction reveicers.|
|Amount|np.float32|from **0.00** to **np.inf**|The amount of each transaction. |
|Location|string|from $\mathbf{L_0}$  to $\mathbf{L}_{nl}$ |$nl$ denotes the number of transacation locations.|
|Type|string|from $\mathbf{TP_0}$ to $\mathbf{TP}_{np}$|$np$ denotes the number of different transaction types. |
|Labels|np.int32|from **0** to **2**|**2** denotes **unlabeled**||


> We are looking for interesting public datasets! If you have any suggestions, please let us know!

## Test Result
The performance of five models tested on three datasets are listed as follows:
| |YelpChi| | |Amazon| | |S-FFSD| | |
|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
| |AUC|F1|AP|AUC|F1|AP|AUC|F1|AP|
|MCNN||- | -| -| -| -|0.7129|0.6861|0.3309|
|STAN|- |- | -| -| -| -|0.7422|0.6698|0.3324|
|STAGN|- |- | -| -| -| -|0.7659|0.6852|0.3599|
|GTAN|0.9241|0.7988|0.7513|0.9630|0.9213|0.8838|0.8286|0.7336|0.6585|
|RGTAN|0.9498|0.8492|0.8241|0.9705|0.9198|0.8925|0.8461|0.7513|0.6939|

> `MCNN`, `STAN` and `STAGN` are presently not applicable to YelpChi and Amazon datasets.

## Repo Structure
The repository is organized as follows:
- `models/`: the pre-trained models for each method. The readers could either train the models by themselves or directly use our pre-trained models;
- `data/`: dataset files;
- `config/`: configuration files for different models;
- `feature_engineering/`: data processing;
- `methods/`: implementations of models;
- `main.py`: organize all models;
- `requirements.txt`: package dependencies;

    
## Requirements
```
python           3.7
scikit-learn     1.0.2
pandas           1.3.5
numpy            1.21.6
networkx         2.6.3
scipy            1.7.3
torch            1.12.1+cu113
dgl-cu113        0.8.1
```
## Citing

If you find *Antifraud* is useful for your research, please consider citing the following papers:

    @inproceedings{Xiang2023SemiSupervisedCC,
        title={Semi-supervised Credit Card Fraud Detection via Attribute-driven Graph Representation},
        author={Sheng Xiang and Mingzhi Zhu and Dawei Cheng and Enxia Li and Ruihui Zhao and Yi Ouyang and Ling Chen and Yefeng Zheng},
        booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
        year={2023}
    }
    @article{cheng2020graph,
        title={Graph Neural Network for Fraud Detection via Spatial-temporal Attention},
        author={Cheng, Dawei and Wang, Xiaoyang and Zhang, Ying and Zhang, Liqing},
        journal={IEEE Transactions on Knowledge and Data Engineering},
        year={2020},
        publisher={IEEE}
    }
    @inproceedings{cheng2020spatio,
        title={Spatio-temporal attention-based neural network for credit card fraud detection},
        author={Cheng, Dawei and Xiang, Sheng and Shang, Chencheng and Zhang, Yiyi and Yang, Fangzhou and Zhang, Liqing},
        booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
        volume={34},
        number={01},
        pages={362--369},
        year={2020}
    }
    @inproceedings{fu2016credit,
        title={Credit card fraud detection using convolutional neural networks},
        author={Fu, Kang and Cheng, Dawei and Tu, Yi and Zhang, Liqing},
        booktitle={International Conference on Neural Information Processing},
        pages={483--490},
        year={2016},
        organization={Springer}
    }
