# AntiFraud
A Credit Card Fraud Detection Framework. Codes and details are coming soon...

Source codes implementation of papers:
- To do...
- AAAI2023: Semi-supervised Credit Card Fraud Detection via Attribute-driven Graph Representation
- AAAI2020: Spatio-temporal attention-based neural network for credit card fraud detection

## Usage

#### Packaging
Git clone this repository, go to the home folder of *antifraud*, run: 

     python setup.py sdist

Pip installable package will be generated in dist/*.tar.gz
#### Install

Simply run below command:

    pip install ./dist/antifraud-0.1.0.tar.gz


#### General Options

You can check out the other options available to use with *Ternary* using:

     python -m antifraud --help

- --method, the processing method, includes: logistic, Adamboost, GBDT, LSTM, cnn,
                                 cnn-att-2d, cnn-att-3d, etc.
- --trainfeature, train feature data
- --trainlabel, train label data
- --testfeature, test feature data
- --testlabel, test label data


#### Example
Generate annotator command:

     python -m antifraud --method GBDT --train data/train.csv --test data/test.csv

#### Data Description


- is_fraud;  0 denotes legitimate transactions and 1 means fraud transactions.
- card_id; unique ID for credit card cardholders. 
- time_stamp; the transaction time.
- loc_cty; the transaction location. 
- loc_merch; the merchant (receiver) of the transaction.  
- amt_grant; the granted transaction amount.  
- amt_purch; the issued transaction amount.
   

> We are looking for interesting public datasets! If you have any suggestions, please let us know!

## Citing

If you find *Antifraud* is useful for your research, please consider citing the following papers:

    @inproceedings{Xiang2023SemiSupervisedCC,
      title={Semi-supervised Credit Card Fraud Detection via Attribute-driven Graph Representation},
      author={Sheng Xiang and Mingzhi Zhu and Dawei Cheng and Enxia Li and Ruihui Zhao and Yi Ouyang and Ling Chen and Yefeng Zheng},
      booktitle={AAAI},
      year={2023}
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
    
## Requirements

-  Python>=3.5
-  numpy>=1.14.3
-  scikit-learn>=0.20.0
-  pytest>=3.6.3
-  pandas>=0.23.3
-  networkx>=2.0
-  scipy>=0.19.1
-  matplotlib>=2.0.2
-  tensorflow==1.12.0
-  keras==2.2.0
-  xgboost>=0.81
