import argparse
import numpy as np
import random
import torch
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.utils.data import Subset, DataLoader

from transformers import AdamW, get_linear_schedule_with_warmup
from typing import List
import time
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

from watchog.dataset import TableDataset, SupCLTableDataset
from watchog.model import SupCLforTable, UnsupCLforTable, SupclLoss

def train_step(train_iter, model, optimizer, scheduler, accelerator, criterion, hp):
    """Perform a single training step

    Args:
        train_iter (Iterator): the train data loader
        model (SupCLforTable, UnsupCLforTable): the model
        optimizer (Optimizer): the optimizer (Adam or AdamW)
        scheduler (LRScheduler): learning rate scheduler
        scaler (GradScaler): gradient scaler for fp16 training
        criterion (SupclLoss)
        hp (Namespace): other hyper-parameters (e.g., fp16)

    Returns:
        None
    """
    train_loss = 0
    device = accelerator.device
    batch_same_label_cnt = []
    for i, batch in enumerate(train_iter):
        if hp.mode == 'simclr':
            # original unsupervised contrastive learning
            x_ori, x_aug, cls_indices = batch
        else:
            # contrastive learning with metadata as supervision
            x_ori, x_aug, cls_indices, y_ori, y_aug = batch
            y = torch.cat((y_ori.to(device), y_aug.to(device)))
            
            # record the # of positive pairs within a batch
            if len(y.shape) == 1:
                cnts = sum((y.unsqueeze(1).repeat(1, y.shape[0]) == y)).tolist()
            else:
                cnts = sum((torch.sum(torch.eq(y.unsqueeze(1).repeat(1,y.shape[0],1), 
                                               y.unsqueeze(0).repeat(y.shape[0],1,1)), dim=-1) == y.shape[-1])).tolist()
            batch_same_label_cnt.extend([int(_)-1 for _ in cnts])

        x_ori = x_ori.to(device)
        x_aug = x_aug.to(device)
        optimizer.zero_grad()
        
        if hp.mode == 'simclr':
            loss = model(x_ori, x_aug, cls_indices, mode='simclr')
        elif hp.mode == 'supcon':
            loss = model(x_ori, x_aug, cls_indices, y, mode='supcon')
        elif hp.mode == 'supcon_ddp':
            # Dummy vectors for allgather
            z = model(x_ori, x_aug, cls_indices, None, mode='supcon')
            z_list = [torch.zeros_like(z) for _ in range(torch.distributed.get_world_size())]
            y_list = [torch.zeros_like(y) for _ in range(torch.distributed.get_world_size())]
            # Allgather
            torch.distributed.all_gather(tensor_list=z_list, tensor=z.contiguous())
            torch.distributed.all_gather(tensor_list=y_list, tensor=y.contiguous())
            # Allgather results do not have gradients
            z_list[torch.distributed.get_rank()] = z
            y_list[torch.distributed.get_rank()] = y
            # Get full batch embeddings: (bs x N, hidden)
            z1 = torch.cat(z_list, 0)
            y1 = torch.cat(y_list, 0)
            loss = criterion(z1, y1)
                
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        if i % 10 == 0: # monitoring
            print(f"step: {i}, loss: {loss.item()}")
        train_loss += loss.item()
        del loss
        
    if hp.mode != 'simclr':
        print("Num of dp and avg # of pos matches in one batch", len(batch_same_label_cnt), sum(batch_same_label_cnt) / len(batch_same_label_cnt)) 
        
    return train_loss


def train(accelerator, trainset, hp, validset=None):

    # initialize model, optimizer, and LR scheduler
    device = accelerator.device
    if hp.mode in ['simclr']:
        model = UnsupCLforTable(hp, device=device, lm=hp.lm).to(device)
    else:
        model = SupCLforTable(hp, device=device, lm=hp.lm).to(device)
        if hp.pretrained_model_path != '':
            model.load_from_pretrained_model(hp.pretrained_model_path)
        
    optimizer = AdamW(model.parameters(), lr=hp.lr)
    
    padder = trainset.pad
    # create the DataLoaders
    train_iter = data.DataLoader(dataset=trainset, shuffle=True,
                                 batch_size=hp.batch_size,
                                 collate_fn=padder)
    if validset is not None:
        valid_iter = data.DataLoader(dataset=validset, shuffle=True,
                                 batch_size=hp.batch_size,
                                 collate_fn=padder)

    num_steps = (len(trainset) // hp.batch_size) * hp.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_steps)

    model, optimizer, train_iter, scheduler = accelerator.prepare(
        model, optimizer, train_iter, scheduler
    )
    criterion = SupclLoss(temperature=hp.temperature)
    
    if accelerator.is_local_main_process and hp.save_model > 0:
        directory = os.path.join(hp.logdir, hp.pretrain_data)
        if not os.path.exists(directory):
            os.makedirs(directory)
        directory = os.path.join(hp.logdir, hp.pretrain_data, hp.mode)
        if not os.path.exists(directory):
            os.makedirs(directory)

    for epoch in range(1, hp.n_epochs+1):
        # train
        accelerator.print("Epoch {} starts.".format(epoch))
        start_time = time.time()
        model.train()

        train_loss = train_step(train_iter, model, optimizer, scheduler, accelerator, criterion, hp)

        # save the checkpoints
        if accelerator.is_local_main_process and hp.save_model > 0:
            # save for every hp.save_model epochs
            if epoch % hp.save_model != 0:
                continue

            ckpt_path = os.path.join(hp.logdir, hp.pretrain_data, hp.mode, 
                hp.lm+'_'+str(hp.size)+'_'+str(hp.n_epochs)+'_'+str(hp.batch_size)+'_'+str(hp.max_len)+'_'+str(hp.lr)+'_' + \
                    str(hp.augment_op)+'_'+str(hp.sample_meth)+'_'+str(hp.table_order)+'_'+str(hp.temperature)+'_'+ \
                    str(hp.run_id)+'_last.pt')

            ckpt = {'model': model.state_dict(), 'hp': hp}
            accelerator.save(ckpt, ckpt_path)

        end_time = time.time()
        accelerator.print("Epoch {} training ends, took {} secs.".format(epoch, end_time - start_time))
        accelerator.print("   Training loss=%f"  %train_loss)
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_data", type=str, default="wikitables") # dataset for pretraining
    parser.add_argument("--pretrained_model_path", type=str, default="") # pretrained checkpoint 
    parser.add_argument("--data_path",type=str, default="./data/doduo")
    parser.add_argument("--mode", type=str, default="simclr") # simclr for original CL, supcon for CL using metadata
    parser.add_argument("--logdir", type=str, default="results/") # directory to store model checkpoints
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--size", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--lm", type=str, default='bert-base-uncased')
    parser.add_argument("--projector", type=int, default=768)
    parser.add_argument("--augment_op", type=str, default='sample_row,sample_row')
    parser.add_argument("--table_order", type=str, default='column')
    parser.add_argument("--sample_meth", type=str, default='head')
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--save_model", type=int, default=5)
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--gpus", type=str, default="0")
    
    
    hp = parser.parse_args()
    # set seed
    seed = hp.run_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision="no" if not hp.fp16 else "fp16", kwargs_handlers=[ddp_kwargs])

    if hp.mode in ['simclr']:
        '''unsupervised'''
        if "viznet" in hp.pretrain_data:
            path = hp.data_path + 'tables/'
        elif "wikitables" in hp.pretrain_data:
            path = hp.data_path + '/train_tables.jsonl'
            valid_path = hp.data_path + 'dev_tables.jsonl'
        else:
            path = 'data/%s/tables' % hp.pretrain_data
        with accelerator.main_process_first():
            trainset = TableDataset.from_hp(path, hp)
            if "wikitables" in hp.pretrain_data:
                validset = TableDataset.from_hp(valid_path, hp)
                trainset.load_from_wikitables(path)
                validset.load_from_wikitables(valid_path)
            elif "gittables" in hp.pretrain_data:
                trainset.load_from_gittables(
                    hp.data_path + '/gittables/parsed_corpus/parsed_32_{}.pkl',
                    hf_rank_path=hp.data_path + '/gittables/gittables.processed.header.freq.json'
                )
        if "wikitables" in hp.pretrain_data:
            train(accelerator, trainset, hp, validset)
        else:
            train(accelerator, trainset, hp)
    else:
        '''supervised'''
        with accelerator.main_process_first():
            trainset = SupCLTableDataset.from_hp(hp.data_path, hp)
            if "wikitables" in hp.pretrain_data:
                trainset.load_from_wikitables_headerprocessed(
                    hp.data_path + '/train_tables.jsonl',
                    hp.data_path + 'processed.header.freq.json'
                )
            elif "gittables" in hp.pretrain_data:
                trainset.load_from_gittables(
                    hp.data_path + '/gittables/parsed_corpus/parsed_32_{}.pkl',
                    hf_rank_path=hp.data_path + '/gittables/gittables.processed.header.freq.json'
                )
        train(accelerator, trainset, hp)
    
        
