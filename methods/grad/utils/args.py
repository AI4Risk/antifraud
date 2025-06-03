import torch

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