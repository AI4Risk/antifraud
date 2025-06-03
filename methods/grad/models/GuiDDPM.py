import argparse
import inspect
import torch

from improved_diffusion import gaussian_diffusion as gd
from improved_diffusion.respace import SpacedDiffusion, space_timesteps
from improved_diffusion.unet import SuperResModel, UNetModel
from improved_diffusion import logger
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.train_util import TrainLoop
from improved_diffusion import dist_util

import torch.distributed as dist
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

import sys
sys.path.append('../')
from utils.MyUtils import color_print, save_pic_iterly

NUM_CLASSES=1000

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def create_model_and_diffusion(
    graph_size,
    class_cond, # 生成模型是有条件的还是无条件的
    learn_sigma, # 要不要使用固定的方差，还是用学习来的方差
    sigma_small,
    # unet相关
    num_channels,
    num_res_blocks,
    num_heads,
    num_heads_upsample,
    attention_resolutions, # unet中使用attention的位置
    dropout,
    # diffusion/model相关
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
):
    '''
    生成并返回训练用的model以及前向扩散过程的diffusion类
    '''
    print(class_cond)

    # 模型代码
    model = create_model(
        graph_size,
        num_channels,
        num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
    )
    # 扩散过程代码
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion

def create_model(
    graph_size,
    num_channels,
    num_res_blocks,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
):
    # if graph_size == 256:
    #     channel_mult = (1, 1, 2, 2, 4, 4)
    # elif graph_size == 64:
    #     channel_mult = (1, 2, 3, 4)
    # elif graph_size == 32:
    #     channel_mult = (1, 2, 2, 2)
    # else:
    #     raise ValueError(f"unsupported image size: {graph_size}")
    
    if graph_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif graph_size == 128:
        channel_mult = (1, 2, 3, 4)
    elif graph_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif graph_size == 32:
        channel_mult = (1, 2, 2, 2)
    elif graph_size == 16:
        channel_mult = (1, 2, 2, 2)
    else:
        raise ValueError(f"unsupported image size: {graph_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(graph_size // int(res))

    num_classes=(NUM_CLASSES if class_cond else None)

    print(num_classes)

    return UNetModel(
        in_channels=1,
        model_channels=num_channels,
        out_channels=(1 if not learn_sigma else 2),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
    )

def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    '''
    生成扩散过程的框架
    '''
    # 确定加噪方案
    betas = gd.get_named_beta_schedule(noise_schedule, steps)

    # 确定loss type
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE # DDPM

    # timestep
    if not timestep_respacing:
        timestep_respacing = [steps]
    
    # 生成最终的diffusion对象
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )

class DDPMTrainDataset(torch.utils.data.Dataset):
    def __init__(self, datalist):
        self.datalist=datalist

        self.length=len(datalist)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # return self.datalist[idx].x, self.datalist[idx].y, self.datalist[idx].adj, self.datalist[idx].train_mask, self.datalist[idx].val_mask, self.datalist[idx].test_mask
        return self.datalist[idx].adj.unsqueeze(dim=0),{}

class DDPMSampleDataset(torch.utils.data.Dataset):
    def __init__(self, datalist):
        self.datalist=datalist

        self.length=len(datalist)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # 获取指定索引的数据项
        data_item = self.datalist[idx]

        # 将数据项组织成字典形式
        data_dict = {
            'x': data_item.new_x,
            'y': data_item.y,
            'adj': data_item.adj,
            'train_mask': data_item.train_mask,
            'val_mask': data_item.val_mask,
            'test_mask': data_item.test_mask
        }

        # 返回数据字典
        return data_item.adj.unsqueeze(dim=0) ,data_dict

def model_and_diffusion_defaults(train_flag, diffusion_steps):
    """
    Defaults for training.
    """
    if train_flag:
        color_print(f'Training diffusion step: {diffusion_steps}')
        return dict(
            graph_size=32,
            num_channels=128,
            num_res_blocks=2,
            num_heads=4,
            num_heads_upsample=-1,
            attention_resolutions="16,8",
            dropout=0.0,
            learn_sigma=False,
            sigma_small=False,
            class_cond=False,
            diffusion_steps=diffusion_steps,
            noise_schedule="linear",
            timestep_respacing="",
            use_kl=False,
            predict_xstart=False,
            rescale_timesteps=True,
            rescale_learned_sigmas=True,
            use_checkpoint=False,
            use_scale_shift_norm=True,
        )
    else:
        color_print(f'Sampling diffusion step: {diffusion_steps}')
        return dict(
            graph_size=32,
            num_channels=128,
            num_res_blocks=2,
            num_heads=4,
            num_heads_upsample=-1,
            attention_resolutions="16,8",
            dropout=0.0,
            learn_sigma=False,
            sigma_small=False,
            class_cond=False,
            diffusion_steps=diffusion_steps,
            noise_schedule="linear",
            timestep_respacing="",
            use_kl=False,
            predict_xstart=False,
            rescale_timesteps=True,
            rescale_learned_sigmas=True,
            use_checkpoint=False,
            use_scale_shift_norm=True,
        )

def train_create_argparser(args, train_flag,diffusion_steps):
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=args.GuiDDPM_train_diffusion_batch_size,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults(train_flag,diffusion_steps))
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def sample_create_argparser(args, num_samples, train_flag,diffusion_steps):
    defaults = dict(
        clip_denoised=True,
        num_samples=num_samples,
        batch_size=args.GuiDDPM_sample_diffusion_batch_size,
        use_ddim=False,
        model_path="",
        classifier_scale=args.GuiDDPM_sample_guidance_scale,
    )
    defaults.update(model_and_diffusion_defaults(train_flag,diffusion_steps))
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

class GuiDDPM:
    def __init__(self, global_args, graph_pyg_ssupgcl, node_groups, edge_index_unselected, guidance, train_flag, model_path, syn_relation_filename, device):
        self.device=device
        self.graph_pyg_ssupgcl=graph_pyg_ssupgcl
        self.node_groups=node_groups
        self.train_flag=train_flag
        self.model_path=model_path
        self.guidance=guidance
        self.global_args=global_args
        self.edge_index_unselected=edge_index_unselected
        self.syn_relation_filename=syn_relation_filename
        self.train_dataset=None
        self.sample_dataset=None
        self.train_data_loader=None
        self.sample_data_loader=None
        self.new_data_list=None

        dist_util.setup_dist()
    
    def trainDataLoader(self):
        data_loader=torch.utils.data.DataLoader(
                dataset=self.train_dataset,
                batch_size=self.train_args.batch_size,
                shuffle=True,num_workers=32,
                drop_last=True)
        while True:
            yield from data_loader

    def train(self, train_steps):
        self.train_args=train_create_argparser(args=self.global_args,
                                               train_flag=self.train_flag,
                                               diffusion_steps=self.global_args.GuiDDPM_train_diffusion_steps).parse_args(args=[])
        logger.configure(dir='tmp/')

        logger.log("creating model and diffusion")
        self.model, self.diffusion=create_model_and_diffusion(
            # 根据模型和diffusion的key从args中找出对应的超参数
            **args_to_dict(self.train_args,model_and_diffusion_defaults(self.train_flag,self.global_args.GuiDDPM_train_diffusion_steps).keys())
        )
        self.model.to(self.device)

        color_print(f'GuiDDPM INFO:')
        print(f'{self.model}')

        # 确定采样的alpha的分布
        self.schedule_sampler=create_named_schedule_sampler(self.train_args.schedule_sampler, self.diffusion)

        self.train_dataset=DDPMTrainDataset(self.node_groups)

        logger.log("creating data loader...")
        self.train_data_loader = self.trainDataLoader()

        logger.log("training...")
        TrainLoop(
            model=self.model,
            diffusion=self.diffusion,
            data=self.train_data_loader,
            batch_size=self.train_args.batch_size,
            microbatch=self.train_args.microbatch,
            lr=self.train_args.lr,
            ema_rate=self.train_args.ema_rate,
            log_interval=self.train_args.log_interval,
            save_interval=self.train_args.save_interval,
            resume_checkpoint=self.train_args.resume_checkpoint,
            use_fp16=self.train_args.use_fp16,
            fp16_scale_growth=self.train_args.fp16_scale_growth,
            schedule_sampler=self.schedule_sampler,
            weight_decay=self.train_args.weight_decay,
            lr_anneal_steps=train_steps,
        ).run_loop()

    def sample(self):
        self.sample_args=sample_create_argparser(args=self.global_args,
                                                 num_samples=len(self.node_groups), 
                                                 train_flag=self.train_flag,
                                                 diffusion_steps=self.global_args.GuiDDPM_sample_diffusion_steps).parse_args(args=[])
        logger.configure(dir='tmp/')

        logger.log("creating model and diffusion")
        self.model, self.diffusion=create_model_and_diffusion(
            # 根据模型和diffusion的key从args中找出对应的超参数
            **args_to_dict(self.sample_args,model_and_diffusion_defaults(self.train_flag, self.global_args.GuiDDPM_sample_diffusion_steps).keys())
        )

        self.load_model(self.model_path)

        self.model.to(self.device)
        color_print(f'GuiDDPM INFO:')
        print(f'{self.model}')

        self.sample_dataset=DDPMSampleDataset(self.node_groups)

        self.sample_data_loader=torch.utils.data.DataLoader(
            dataset=self.sample_dataset,
            batch_size=self.sample_args.batch_size,
            shuffle=False,
            num_workers=32,
            drop_last=False,
        )

        self.model.eval()
        logger.log("sampling...")
        color_print(f'!!!!! Start generating new relations')
        all_new_subgraphs = []
        all_labels = []
        model_device = next(self.model.parameters()).device
        for i, batch in enumerate(self.sample_data_loader):        
            # if i<11:
            #     continue
            while self.sample_args.batch_size > batch[0].shape[0]:  
                addition_x = torch.zeros_like(batch[0])
                batch[0] = torch.cat([batch[0],addition_x],dim=0)[:self.sample_args.batch_size,]
                
            print(batch[0].shape)
                
            batch[0]=batch[0].to(model_device)
            model_kwargs = {}
            sub_info = {}
            if self.sample_args.class_cond:
                classes = torch.randint(
                    # low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
                    low=0, high=NUM_CLASSES, size=(self.sample_args.batch_size,), device=self.device
                )
                model_kwargs["y"] = classes

            sub_info["data_dict"]=batch[1]
            
            sample_fn = (
                self.diffusion.p_sample_loop if not self.sample_args.use_ddim else self.diffusion.ddim_sample_loop
            )
            sample = sample_fn(
                self.model,
                (self.sample_args.batch_size, 1, self.sample_args.graph_size, self.sample_args.graph_size),
                clip_denoised=self.sample_args.clip_denoised,
                model_kwargs=model_kwargs,
                progress=True,
                cond_fn=self.cond_fn if self.global_args.GuiDDPM_sample_with_guidance else None,
                noise=batch[0],
                sub_info=sub_info,
            )
            # sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            sample = ((sample + 1) * 0.5).clamp(0, 1).to(torch.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()

            gathered_samples = [torch.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_new_subgraphs.extend([sample.cpu().numpy() for sample in gathered_samples])

            if self.sample_args.class_cond:
                gathered_labels = [
                    torch.zeros_like(classes) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(gathered_labels, classes)
                all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
            logger.log(f"created {len(all_new_subgraphs) * self.sample_args.batch_size} samples")
            color_print(f"created {len(all_new_subgraphs) * self.sample_args.batch_size} samples")


        arr=self.degreeAnalysis(all_new_subgraphs=all_new_subgraphs, all_labels=all_labels)

        self.saveSynRelations(arr)

    def saveSynRelations(self, arr):
        new_adj=torch.tensor(arr.squeeze())
        self.new_data_list=self.node_groups
        for i, d in enumerate(tqdm(self.new_data_list)):
            d.adj=new_adj[i]
            d.edge_index=dense_to_sparse(new_adj[i])[0]
        
        syn_relation_dict={
            'syn_relation_list': self.new_data_list,
            'unselected_edge_index': self.edge_index_unselected,
            'graph_pyg_ssupgcl_new_x': self.graph_pyg_ssupgcl.new_x
        }

        torch.save(syn_relation_dict, self.syn_relation_filename)

        color_print(f'!!!!! SynRelation Dict is saved in {self.syn_relation_filename} Success')
    

    def degreeAnalysis(self, all_new_subgraphs, all_labels):
        arr = np.concatenate(all_new_subgraphs, axis=0)
        arr = arr[: self.sample_args.num_samples]
        if self.sample_args.class_cond: 
            label_arr = np.concatenate(all_labels, axis=0)
            label_arr = label_arr[: self.sample_args.num_samples]
        if dist.get_rank() == 0:
            shape_str = "x".join([str(x) for x in arr.shape])
            out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
            logger.log(f"saving to {out_path}")
            if self.sample_args.class_cond:
                np.savez(out_path, arr, label_arr)
            else:
                np.savez(out_path, arr)

        dist.barrier()
        logger.log("sampling complete")

        # 计算每个张量的元素和
        sums=[]
        for a in arr:
            sums.extend(r.sum().item() for r in a.squeeze())

        
        # 使用matplotlib绘制直方图
        plt.hist(sums, bins=20, alpha=0.75)  # bins指定直方图的柱子数量，alpha指定柱子的透明度
        plt.title('Distribution of Tensor Sums')
        plt.xlabel('Sum')
        plt.ylabel('Frequency')

        plt.show()

        pic_name=f'./tmp/{self.global_args.dataset}_{self.global_args.GuiDDPM_sample_diffusion_steps}sampleSteps_GuiDDPMAnalysis'
        info='GuiDDPM Analysis'
        save_pic_iterly(pic_name=pic_name, postfix='png', info=info)

        return arr

    def save_model(self, path):
        torch.save(self.model.state_dict(),path)
        color_print(f'!!!!! GuiDDPM saving parameter in {path} Success')

    def load_model(self, path):
        para=torch.load(path, map_location=self.device)
        self.model.load_state_dict(para)
        color_print(f'!!!!! GuiDDPM loading parameter from {path} Success')

    def cal_sim(self, x,edge_index):
        # 计算节点特征的L2范数
        x_norm = F.normalize(x, p=2, dim=1)

        # node_similarity = [[] for _ in range(x.shape[0])]
        num_nodes=x.shape[0]
        node_similarity = torch.zeros(num_nodes,num_nodes)

        # 遍历每条边，计算相似度
        for i, j in edge_index.t():
            sim = torch.dot(x_norm[i], x_norm[j])
            node_similarity[i.item()][j.item()] = node_similarity[i.item()][j.item()] + sim.item()
            # node_similarity[j.item()][i.item()] = node_similarity[j.item()][i.item()] + sim.item()
        
        sim_mean=node_similarity.mean()
        sim_std=node_similarity.std()
        
        sim_norm=((node_similarity-sim_mean)/sim_std).unsqueeze(dim=0)
        
        return sim_norm

    def cal_sim_loop(self, xs,adjs):
        sim_arrs=[]
        for x,adj in zip(xs,adjs):
            edge_index=dense_to_sparse(adj)[0]
            sim_arrs.append(self.cal_sim(x,edge_index))
        # print(len(sim_arrs),sim_arrs[0].shape)
        sim_arrs=torch.stack(sim_arrs,dim=0)
        
        while self.sample_args.batch_size > sim_arrs.shape[0]:  
            addition_x = torch.zeros_like(sim_arrs)
            sim_arrs = torch.cat([sim_arrs,addition_x],dim=0)[:self.sample_args.batch_size,]
            
        return torch.tensor(sim_arrs)

    def degree_penalty_loss(self, adj_matrix, scale_factor=0.1):
        # 计算每个节点的度
        degree = adj_matrix.sum(dim=1)
        # 惩罚项为度的平方和
        penalty = (degree ** 2).sum()
        return scale_factor * penalty

    def cond_fn(self, xs,adjs):
        # xin_grads=[]
        total_xin_grads=None
        total_xin_grads_list=[]
        
        with torch.enable_grad():
            for x,adj in zip(xs,adjs):
                x_in=x.detach().requires_grad_(True).to(self.device)
                
                # z,z1,z2=encoder_model(x_in,None,None)
                h1,h2=[self.guidance.encoder_model.project(x) for x in [x_in,x_in]]
                
                extra_pos_mask=adj.to(self.device)
                extra_pos_mask.fill_diagonal_(False)
                extra_pos_mask = torch.cat([extra_pos_mask, extra_pos_mask], dim=1).to(self.device)
                extra_pos_mask.fill_diagonal_(True)
                
                extra_neg_mask=(1-adj).to(self.device)
                extra_neg_mask.fill_diagonal_(False)
                extra_neg_mask = torch.cat([extra_neg_mask, extra_neg_mask], dim=1).to(self.device)
                
                loss=self.guidance.contrast_model(h1=h1,h2=h2,extra_pos_mask=extra_pos_mask,extra_neg_mask=extra_neg_mask)
                
                # 累加梯度
                additional_loss = self.degree_penalty_loss(adj, scale_factor=100)
                if total_xin_grads is None:
                    total_xin_grads = torch.autograd.grad(loss,x_in)[0] + additional_loss
                else:
                    # total_xin_grads = (total_xin_grads + additional_loss)
                    total_xin_grads = (total_xin_grads + torch.autograd.grad(loss,x_in)[0] + additional_loss)

                # total_xin_grads_list.append(torch.autograd.grad(loss,x_in)[0] + additional_loss)
                # total_xin_grads=torch.stack(total_xin_grads_list, dim=0)

            #     xin_grads.append(torch.autograd.grad(loss,x_in)[0])
            
            # xin_grads=torch.stack(xin_grads)
            # print(F.softmax(total_xin_grads)/x.shape[0])
            # print(xs,adjs)
            
            # print(total_xin_grads.shape,x_in.shape,xs.shape)

            return F.softmax(total_xin_grads) * self.sample_args.classifier_scale
