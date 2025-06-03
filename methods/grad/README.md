# Grad: Guided Relation Diffusion Generation for Graph Augmentation in Graph Fraud Detection

[Grad: Guided Relation Diffusion Generation for Graph Augmentation in Graph Fraud Detection | Proceedings of the ACM on Web Conference 2025](https://dl.acm.org/doi/abs/10.1145/3696410.3714520)



Several generation results of amazon and yelp can be find in https://huggingface.co/Muyiiiii-HF/WWW25-Grad/tree/main .



## Requirements

This code requires the following:

- python==3.9

- pytorch==1.12.1+cu113

  - ```
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
    ```

  - ```
    pip uninstall numpy
    ```

  - ```
    pip install numpy==1.26.0
    ```

- dgl==0.9.1+cu113

  - ```
    pip install dgl-cu113 -f https://data.dgl.ai/wheels/repo.html
    ```

- pygcl

  - ```
    pip install PyGCL
    ```

- mpi4py

  - ```
    conda install mpi4py
    ```

  - or

  - ```
    pip install mpi4py
    ```

- if error in pip

  - ```
    apt-get update
    ```

  - ```
    apt-get install mpich
    ```

  - ```
    pip install mpi4py
    ```

- torch_geometric==2.2.0

- improved-diffusion

  - ```
    cd models
    ```

  - ```
    pip install -e .
    ```

## Datasets

Dataset is processed in `dataProcess.py`

```python
def loadDataset(dataset, train_ratio):
    if dataset=='yelp':
        graph_dgl = FraudYelpDataset()[0]
        graph_dgl = dgl.to_homogeneous(graph_dgl,ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
        graph_dgl = dgl.add_self_loop(graph_dgl)

        feat = graph_dgl.ndata['feature'].float()
        label = graph_dgl.ndata['label'].long()
        edge_index = torch.stack(graph_dgl.edges(etype=('_N', '_E', '_N')))
        index = list(range(len(label)))

        train_mask, val_mask, test_mask = splitDataset(index,label,train_ratio)

        graph_pyg=Data(x=feat,edge_index=edge_index,y=label,train_mask=train_mask,val_mask=val_mask,test_mask=test_mask)
        graph_dgl.ndata['train_mask']=train_mask
        graph_dgl.ndata['val_mask']=val_mask
        graph_dgl.ndata['test_mask']=test_mask

    elif dataset=='amazon':
        graph_dgl = FraudAmazonDataset()[0]
        graph_dgl = dgl.to_homogeneous(graph_dgl,ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
        graph_dgl = dgl.add_self_loop(graph_dgl)

        feat = graph_dgl.ndata['feature'].float()
        label = graph_dgl.ndata['label'].long()
        edge_index = torch.stack(graph_dgl.edges(etype=('_N', '_E', '_N')))
        index = list(range(3305, len(label)))

        train_mask, val_mask, test_mask = splitDataset(index,label,train_ratio)

        graph_pyg=Data(x=feat,edge_index=edge_index,y=label,train_mask=train_mask,val_mask=val_mask,test_mask=test_mask)
        graph_dgl.ndata['train_mask']=train_mask
        graph_dgl.ndata['val_mask']=val_mask
        graph_dgl.ndata['test_mask']=test_mask

    elif dataset=='blogcatalog':
        bc_path = f'../bc.pt'
        graph_pyg = torch.load(bc_path)[0]
        graph_pyg.edge_index = graph_pyg.edge_indexes[0]
        graph_dgl = pyg_data_to_dgl_graph(graph_pyg)

        label = graph_dgl.ndata['label'].long()
        index = list(range(len(label)))
        train_mask, val_mask, test_mask = splitDataset(index,label,train_ratio)

        graph_pyg.train_mask=train_mask
        graph_pyg.val_mask=val_mask
        graph_pyg.test_mask=test_mask
        graph_dgl.ndata['train_mask']=train_mask
        graph_dgl.ndata['val_mask']=val_mask
        graph_dgl.ndata['test_mask']=test_mask
    else:
        print(f'{dataset}-graph_dgl not exist')
        return None
```

## Usage

**The args are in the utils/MyUtils.py**

```python
class argVar:
    def __init__(self):
        self.dataset='amazon' # 'amazon', 'yelp',
        self.train_ratio=0.4
        self.nodes_per_subgraph=32
        self.num_classes=5
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.SupGCL_batch_size=2
        self.SupGCL_num_train_part=20
        self.SupGCL_epochs=50
        self.SupGCL_train_flag=True
        self.SupGCL_visualize_flag=False ### visualization of SupGCL
  
        self.GuiDDPM_train_flag=False # train or sample
        self.GuiDDPM_train_steps=6000 
        self.GuiDDPM_train_diffusion_steps=1000
        self.GuiDDPM_train_diffusion_batch_size=20
        self.GuiDDPM_sample_diffusion_batch_size=256
        self.GuiDDPM_sample_guidance_scale=10
        self.GuiDDPM_sample_with_guidance=True 
        self.GuiDDPM_sample_diffusion_steps=100
  
        self.WFusion_hid_dim=256
        self.WFusion_order=5
        self.WFusion_epochs=250
        self.WFusion_gdc_syn_avg_degree=[]
        self.WFusion_gdc_raw_avg_degree=[]
        if self.dataset=='amazon':
            self.WFusion_relation_index=[0,] # only original and generated relations, no gdc
        elif self.dataset=='yelp':
            self.WFusion_relation_index=[0, 1,]
        self.WFusion_use_WFusion=True ###
```

1. **Train diffusion**
   Can set the `self.GuiDDPM_train_steps` for your like.

   ```python
   self.GuiDDPM_train_flag=True
   ```

   ```
   python generation_main.py
   ```

2. **Sample with guidance**

   ```python
   self.GuiDDPM_train_flag=False
   ```

   ```
   python generation_main.py
   ```

3. **Detection**

   Several generation results of amazon and yelp can be find in https://huggingface.co/Muyiiiii-HF/WWW25-Grad/tree/main .

   ```
   python detect_main.py
   ```
