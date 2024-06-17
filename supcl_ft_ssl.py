import argparse
import json
import math
import os
import random
from time import time
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from collections import OrderedDict, Counter


import pytrec_eval
import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from accelerate import Accelerator
import torch.nn.functional as F

from watchog.dataset import (
    TURLColTypeTablewiseDataset,
    TURLRelExtTablewiseDataset,
    SatoCVTablewiseDataset,
    ColPoplTablewiseDataset
)


from watchog.dataset import TableDataset, SupCLTableDataset, SatoCVTablewiseDatasetWithDA, TURLColTypeTablewiseDatasetWithDA, TURLRelExtTablewiseDatasetWithDA
from watchog.model import SupCLforTable, UnsupCLforTable, BertMultiPairPooler, BertForMultiOutputClassification, lm_mp
from watchog.utils import f1_score_multilabel, load_checkpoint, collate_fn
from accelerate import DistributedDataParallelKwargs

BASE_DATA_PATH2 = '/efs/pretrain_datasets/'



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    
def create_ssl_batches(epoch, l_set, u_set, padder,
                            num_aug=2,
                            batch_size=16):
    """Create batches for SSL
    Each batch is the concatenation of (1) a labeled batch, (2) an augmented
    labeled batch (having the same order of (1) ), (3) an unlabeled batch,
    and (4) multiple augmented unlabeled batches of the same order
    of (3).
    Args:
        l_set: the train set with augmented examples
        u_set: the unlabeled set with augmented examples
        num_aug (int, optional): number of unlabeled augmentations to be created
        batch_size (int, optional): batch size (of each component)
    Returns:
        list of list: the created batches
    """
    mixed_batches = []
    num_labeled = len(l_set)
    l_index = np.random.permutation(num_labeled)
    print(epoch, len(l_set), len(u_set))
    u_order = list(range(len(u_set)))
    random.shuffle(u_order)
    u_order = np.array(u_order)

    u_index = np.random.permutation(num_labeled) + num_labeled * epoch
    u_index = (u_index + len(u_set)) % len(u_set)
    u_index = u_order[u_index]
    
    l_batch = []
    l_batch_aug = []
    u_batch = []
    u_batch_aug = [[] for _ in range(num_aug)]
    for i, idx in enumerate(l_index):
        u_idx = u_index[i]
        ll = l_set[idx]
        l_batch.append({'data': ll['ori'], 'label': ll['label_ori']})
        l_batch_aug.append({'data': ll['aug'][0], 'label': torch.empty(0).cuda()})
        # add augmented examples of unlabeled with dummy label
        uu = u_set[u_idx]
        u_batch.append({'data': uu['ori'], 'label': torch.empty(0).cuda()})
        for uid in range(num_aug):
            u_batch_aug[uid].append({'data': uu['aug'][uid], 'label': torch.empty(0).cuda()})

        if len(l_batch) == batch_size or i == len(l_index) - 1:
            batches = l_batch + u_batch + l_batch_aug
            for ub in u_batch_aug:
                batches += ub
            mixed_batches.append(padder(batches))
            l_batch.clear()
            l_batch_aug.clear()
            u_batch.clear()
            for ub in u_batch_aug:
                ub.clear()
        
    random.shuffle(mixed_batches)

    return mixed_batches

def create_bl_batches(task, epoch, l_set, u_set, padder,
                            num_aug=2,
                            u_ratio=7,
                            batch_size=16):
    """Create batches for SSL
    Each batch is the concatenation of (1) a labeled batch, (2) an augmented
    labeled batch (having the same order of (1) ), (3) an unlabeled batch,
    and (4) multiple augmented unlabeled batches of the same order
    of (3).
    Args:
        l_set: the train set with augmented examples
        u_set: the unlabeled set with augmented examples
        num_aug (int, optional): number of unlabeled augmentations to be created
        u_ratio (int, optional): ratio of unlabeled examples
        batch_size (int, optional): batch size (of each component)
    Returns:
        list of list: the created batches
    """
    mixed_batches = []
    num_labeled = len(l_set)
    l_index = np.random.permutation(num_labeled)
    print(epoch, len(l_set), len(u_set))
    u_order = list(range(len(u_set)))
    random.shuffle(u_order)
    u_order = np.array(u_order)
    u_index = np.random.permutation(num_labeled) + num_labeled * epoch
    u_index = (u_index + len(u_set)) % len(u_set)
    u_index = u_order[u_index]

    l_batch = []    
    l_batch_aug = []
    u_batch_aug = [[] for _ in range(num_aug)]
    u_idx = 0
    l_col_cnt = 0
    u_col_cnt = 0
    for i, idx in enumerate(l_index):
        ll = l_set[idx]
        l_col_idx = [l_col_cnt+_ for _ in range(len(ll['label_ori']))]
        l_col_cnt += len(l_col_idx)
        l_batch.append({'data': ll['ori'], 'label': ll['label_ori'], 'idx': l_col_idx})
        l_batch_aug.append({'data': ll['aug'][0], 'label': torch.empty(0).cuda(), 'idx': l_col_idx})
        # add augmented examples of unlabeled
        for _ in range(u_ratio):
            uu = u_set[u_idx+_]
            if task == 'turl-re':
                u_col_idx = [u_col_cnt+__ for __ in range(len(uu['cls_indexes'])-1)]
                u_col_cnt += len(uu['cls_indexes'])-1
            else:
                u_col_idx = [u_col_cnt+__ for __ in range(len(uu['cls_indexes']))]
                u_col_cnt += len(uu['cls_indexes'])
            for uid in range(num_aug):
                u_batch_aug[uid].append({'data': uu['aug'][uid], 'label': torch.empty(0).cuda(), 'idx': u_col_idx})
        u_idx += u_ratio
        if len(l_batch) == batch_size or i == len(l_index) - 1:
            batches = l_batch + l_batch_aug
            for ub in u_batch_aug:
                batches += ub

            mixed_batches.append(padder(batches))
            l_batch.clear()
            l_batch_aug.clear()
            for ub in u_batch_aug:
                ub.clear()
    random.shuffle(mixed_batches)

    return mixed_batches, l_col_cnt, u_col_cnt

def get_column_logits(logits, cls_indexes, drop_first_col=False):
    '''retrieve the logits for each CLS token'''
    
    if len(logits.shape) == 2:
        logits = logits.unsqueeze(0)
    filtered_logits = torch.zeros(cls_indexes.shape[0],
                                logits.shape[2]).to(device)
    for n in range(cls_indexes.shape[0]):
        i, j = cls_indexes[n]
        logit_n = logits[i, j, :]
        filtered_logits[n] = logit_n
    return filtered_logits

def ssl_with_ps(epoch, task, ssl_tag, model, cls_token_id, loss_fn, batch, num_aug=2, u_lambda=0.5):
    """Perform one iteration of ssl"""
    x = batch['data'].T
    y = batch['label']
    # two batches of labeled and two batches of unlabeled
    batch_size = x.size()[0] // (num_aug + 3)
    
    # the unlabeled half
    u0 = x[batch_size:2*batch_size]
    # augmented
    aug_x = x[2*batch_size:3*batch_size]

    # augmented unlabeled
    u_augs = []
    for uid in range(num_aug):
        u_augs.append(x[(3+uid)*batch_size:(4+uid)*batch_size])

    # labeled + augmented unlabeled
    x = torch.cat((x[:batch_size], x[3*batch_size:]))

    # label guessing
    model.eval()
    u_guesses = []
    u_aug_enc_list = []

    for x_u in u_augs:
        # it is fine to switch the order of x_u and u0 in this case
        u_logits, u_aug_enc = model(x_u,
                               get_enc=True)
        cls_indexes = torch.nonzero(x_u == tokenizer.cls_token_id)
        u_aug_col_logits = get_column_logits(u_logits, cls_indexes)
        # softmax
        u_guess = F.softmax(u_aug_col_logits, dim=-1)
        u_guess = u_guess.detach()
        u_guesses.append(u_guess)

        # save u_aug_enc
        u_aug_enc_list.append(u_aug_enc)

    # averaging
    u_guess = sum(u_guesses) / len(u_guesses)

    # temperature sharpening
    T = 0.5
    u_power = u_guess.pow(1/T)
    u_guess = u_power / u_power.sum(dim=-1, keepdim=True)

    # make duplicate of u_guess
    if len(u_guess.size()) == 2:
        u_guess = u_guess.repeat(num_aug, 1)
    else:
        u_guess = u_guess.repeat(num_aug, 1, 1)

    vocab = u_guess.shape[-1]
    # switch back to training mode
    model.train()

    # convert y to one-hot
    if task in ['turl', 'turl-re']:
        y_onehot = y
    else:
        y_onehot = F.one_hot(y.long(), vocab).float()
    l_y_size = y.shape[0]
    y_concat = torch.cat((y_onehot, u_guess))

    # forward
    logits = model(x)

    x_cls_indexes = torch.nonzero(x[:batch_size] == cls_token_id)
    u_cls_indexes = torch.nonzero(x[batch_size:] == cls_token_id)
    l_col_logits = get_column_logits(logits[:batch_size], x_cls_indexes)
    u_col_logits = get_column_logits(logits[batch_size:], u_cls_indexes)    

    l_pred = l_col_logits.view(-1, vocab)
    u_pred = u_col_logits.view(-1, vocab)
    y = y_concat
    l_y = y[:l_y_size].view(-1, vocab)
    u_y = y[l_y_size:].view(-1, vocab)
    
    loss_x = loss_fn(l_pred, l_y)
    if task in ['turl', 'turl-re']:
        loss_u = loss_fn(u_pred, u_y)
    else:
        loss_u = F.mse_loss(u_pred, u_y)

    loss = loss_x + loss_u * u_lambda
    
    return loss, l_pred, loss_x.item(), loss_u.item()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shortcut_name",
        default="bert-base-uncased",
        type=str,
        help="Huggingface model shortcut name ",
    )
    parser.add_argument(
        "--max_length",
        default=128,
        type=int,
        help=
        "The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--epoch",
        default=30,
        type=int,
        help="Number of epochs for training",
    )
    parser.add_argument(
        "--random_seed",
        default=4649,
        type=int,
        help="Random seed",
    )
    
    parser.add_argument(
        "--train_n_seed_cols",
        default=-1,
        type=int,
        help="number of seeding columns in training",
    )

    parser.add_argument(
        "--num_classes",
        default=78,
        type=int,
        help="Number of classes",
    )
    parser.add_argument("--fp16",
                        action="store_true",
                        default=False,
                        help="Use FP16")
    parser.add_argument("--warmup",
                        type=float,
                        default=0.,
                        help="Warmup ratio")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--task",
                        type=str,
                        default="sato0",
                        choices=[
                            "sato0", "sato1", "sato2", "sato3", "sato4",
                            "msato0", "msato1", "msato2", "msato3", "msato4",
                            "turl", "turl-re", 
                            "col-popl-turl-0", "col-popl-turl-1"
                        ],
                        help="Task names}")
    parser.add_argument("--colpair",
                        action="store_true",
                        help="Use column pair embedding")
    parser.add_argument("--metadata",
                        action="store_true",
                        help="Use column header metadata")
    parser.add_argument("--from_scratch",
                        action="store_true",
                        help="Training from scratch")
    parser.add_argument("--cl_tag",
                        type=str,
                        default="viznet/None/model_drop_col_tfidf_entity_column_0",
                        help="path to the pre-trained file")
    parser.add_argument("--dropout_prob",
                        type=float,
                        default=0.5)
    parser.add_argument("--eval_test",
                        action="store_true",
                        help="evaluate on testset and do not save the model file")
    parser.add_argument("--small_tag",
                        type=str,
                        default="",
                        help="e.g., by_table_t5_v1")
    parser.add_argument("--data_path",
                        type=str,
                        default="/efs/task_datasets/")
    parser.add_argument("--pretrained_ckpt_path",
                        type=str,
                        default="./results/")
    parser.add_argument("--temperature",
                        type=float,
                        default=0.5)
    parser.add_argument("--p_cutoff",
                        type=float,
                        default=0.95)
    # mlflow tag
    parser.add_argument("--mlflow_tag", type=str, default=None)
    parser.add_argument("--ssl", type=str, default='')
    parser.add_argument("--augment_op", type=str, default='sample_row4')
    parser.add_argument("--alpha_aug", type=float, default=0.5)
    parser.add_argument("--num_aug", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--u_ratio", type=int, default=1)
    parser.add_argument("--u_lambda", type=float, default=0.01)
    
    args = parser.parse_args()
    args.cl_tag_output = args.cl_tag.replace('tfidf_entity_column', '').replace('mosato_distilbert-base-uncased', '')
    task = args.task
    if args.small_tag != "":
        args.eval_test = True
    if args.ssl != '':
        if 'labelguess' in args.ssl:
            args.ssl += '_{}-na{}-a{}-ul{}'.format(
                args.augment_op, args.num_aug, args.alpha, args.u_lambda
            )
        elif 'bl' in args.ssl or 'cpl' in args.ssl:
            args.ssl += '_{}-na{}-ul{}-ur{}-pc{}'.format(
                args.augment_op, args.num_aug, args.u_lambda, args.u_ratio, args.p_cutoff
            )
        else:
            args.ssl += '_{}-na{}-ul{}'.format(
                args.augment_op, args.num_aug, args.u_lambda
            )
            
    task_num_class_dict = {
        "sato0": 78,
        "sato1": 78,
        "sato2": 78,
        "sato3": 78,
        "sato4": 78,
        "msato0": 78,
        "msato1": 78,
        "msato2": 78,
        "msato3": 78,
        "msato4": 78,
        "turl": 255,
        "turl-re": 121,
        "col-popl-turl-0": 5407,
        "col-popl-turl-1": 5407
    }

    
    args.num_classes = task_num_class_dict[task]
    if args.colpair:
        assert "turl-re" == task, "colpair can be only used for Relation Extraction"
    if args.metadata:
        assert "turl-re" == task or "turl" == task, "metadata can be only used for TURL datasets"
    
    print("args={}".format(json.dumps(vars(args))))

    max_length = args.max_length
    batch_size = args.batch_size
    num_train_epochs = args.epoch
    
    shortcut_name = args.shortcut_name

    tag_name_col = "mosato"

    if args.colpair and args.metadata:
        taskname = "{}-colpair-metadata".format(task)
    elif args.colpair:
        taskname = "{}-colpair".format(task)
    elif args.metadata:
        taskname = "{}-metadata".format(task)
    else:
        taskname = "".join(task)

    if args.eval_test:
        if args.from_scratch:
            tag_name = "ssl_outputs/{}/{}_{}_{}-bs{}-ml{}-ne{}-do{}".format(
                taskname, tag_name_col, "{}-fromscratch".format(shortcut_name),  args.ssl,
                batch_size, max_length, num_train_epochs, args.dropout_prob)
        else:
            tag_name = "ssl_outputs/{}/{}_{}_{}_{}-bs{}-ml{}-ne{}-do{}".format(
                taskname, args.cl_tag_output.replace('/', '-'), tag_name_col, shortcut_name, args.ssl,
                batch_size, max_length, num_train_epochs, args.dropout_prob)
    else:
        if args.from_scratch:
            tag_name = "model/{}_{}_{}_{}-bs{}-ml{}-ne{}-do{}".format(
                taskname, tag_name_col, "{}-fromscratch".format(shortcut_name), args.ssl,
                batch_size, max_length, num_train_epochs, args.dropout_prob)
        else:
            tag_name = "model/{}_{}_{}_{}_{}-bs{}-ml{}-ne{}-do{}".format(
                args.cl_tag_output.replace('/', '-'), taskname, tag_name_col, shortcut_name, args.ssl,
                batch_size, max_length, num_train_epochs, args.dropout_prob)
    
    if args.eval_test:
        if args.small_tag != '':
            tag_name += '-' + args.small_tag
    
    print(tag_name)

    dirpath = os.path.dirname(tag_name)
    if not os.path.exists(dirpath):
        print("{} not exists. Created".format(dirpath))
        os.makedirs(dirpath)
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = accelerator.device
    ckpt_path = '{}/{}.pt'.format(args.pretrained_ckpt_path, args.cl_tag)
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_hp = ckpt['hp']
    print(ckpt_hp)
    setattr(ckpt_hp, 'batch_size', args.batch_size)
    setattr(ckpt_hp, 'hidden_dropout_prob', args.dropout_prob)
    setattr(ckpt_hp, 'shortcut_name', args.shortcut_name)
    setattr(ckpt_hp, 'num_labels', args.num_classes)
    
    pre_model, trainset = load_checkpoint(ckpt)
    tokenizer = trainset.tokenizer

    if task == "turl-re" and args.colpair:
        model = BertForMultiOutputClassification(ckpt_hp, device=device, lm=ckpt['hp'].lm, col_pair='Pair')
    else:
        model = BertForMultiOutputClassification(ckpt_hp, device=device, lm=ckpt['hp'].lm)
        

    if not args.from_scratch:
        model.bert = pre_model.bert
        # model.projector = pre_model.projector
    if task == "turl-re" and args.colpair and ckpt['hp'].lm != 'distilbert':
        config = BertConfig.from_pretrained(lm_mp[ckpt['hp'].lm])
        model.bert.pooler = BertMultiPairPooler(config).to(device)
        print("Use column-pair pooling")

    del pre_model

    padder = collate_fn(trainset.tokenizer.pad_token_id)
    # with accelerator.main_process_first():
    if True:
        if task in [
                "sato0", "sato1", "sato2", "sato3", "sato4", "msato0",
                "msato1", "msato2", "msato3", "msato4"
        ]:
            cv = int(task[-1])

            if task[0] == "m":
                multicol_only = True
            else:
                multicol_only = False

            dataset_cls = SatoCVTablewiseDataset
            aug_dataset_cls = SatoCVTablewiseDatasetWithDA
            train_dataset = aug_dataset_cls(cv=cv,
                                    split="train",
                                    tokenizer=tokenizer,
                                    max_length=max_length,
                                    multicol_only=multicol_only,
                                    train_ratio=1.0,
                                    device=device,
                                    small_tag=args.small_tag,
                                    augment_op=args.augment_op,
                                    num_aug=1)
            valid_dataset = dataset_cls(cv=cv,
                                        split="valid",
                                        tokenizer=tokenizer,
                                        max_length=max_length,
                                        multicol_only=multicol_only,
                                        train_ratio=1.0,
                                        device=device,
                                        small_tag=args.small_tag)

            train_sampler = RandomSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset,
                                            sampler=train_sampler,
                                            batch_size=batch_size)
                                        #   collate_fn=collate_fn)
                                        # collate_fn=padder)
            valid_dataloader = DataLoader(valid_dataset,
                                            batch_size=batch_size,
                                        #   collate_fn=collate_fn)
                                        collate_fn=padder)
            test_dataset = dataset_cls(cv=cv,
                                        split="test",
                                        tokenizer=tokenizer,
                                        max_length=max_length,
                                        multicol_only=multicol_only,
                                        device=device)
            test_dataloader = DataLoader(test_dataset,
                                            batch_size=batch_size,
                                            collate_fn=padder)
            unlabeled_dataset = aug_dataset_cls(cv=cv,
                                    split="unlabeled",
                                    tokenizer=tokenizer,
                                    max_length=max_length,
                                    multicol_only=multicol_only,
                                    train_ratio=1.0,
                                    device=device,
                                    small_tag=args.small_tag,
                                    augment_op=args.augment_op,
                                    num_aug=args.num_aug)
            unlabeled_sampler = RandomSampler(unlabeled_dataset)
            unlabeled_dataloader = DataLoader(unlabeled_dataset,
                                        sampler=unlabeled_sampler,
                                        batch_size=batch_size * args.u_ratio)
        
        elif "turl" in task:
            if task in ["turl"]:
                # TODO: Double-check if it is compatible with single/multi-column data
                if args.small_tag == "":
                    filepath = "data/doduo/table_col_type_serialized{}.pkl".format(
                        '_with_metadata' if args.metadata else ''
                    )
                else:
                    filepath = "data/turl_small/table_col_type_serialized{}_{}.pkl".format(
                        '_with_metadata' if args.metadata else '', 
                        args.small_tag
                    )
                dataset_cls = TURLColTypeTablewiseDataset
                aug_dataset_cls = TURLColTypeTablewiseDatasetWithDA
                
            elif task in ["turl-re"]:
                if args.small_tag == "":
                    filepath = "data/doduo/table_rel_extraction_serialized{}.pkl".format(
                        '_with_metadata' if args.metadata else ''
                    )
                else:
                    filepath = "data/turl-re_small/table_rel_extraction_serialized{}_{}.pkl".format(
                        '_with_metadata' if args.metadata else '',
                        args.small_tag
                    )
                dataset_cls = TURLRelExtTablewiseDataset
                aug_dataset_cls = TURLRelExtTablewiseDatasetWithDA
            else:
                raise ValueError("turl tasks must be turl or turl-re.")

            train_dataset = aug_dataset_cls(filepath=filepath,
                                        split="train",
                                        tokenizer=tokenizer,
                                        max_length=max_length,
                                        multicol_only=False,
                                        train_ratio=1.0,
                                        device=device,
                                        augment_op=args.augment_op,
                                        num_aug=1)
            valid_dataset = dataset_cls(filepath=filepath,
                                        split="dev",
                                        tokenizer=tokenizer,
                                        max_length=max_length,
                                        multicol_only=False,
                                        device=device)

            # Can be the same
            train_sampler = RandomSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset,
                                            sampler=train_sampler,
                                            batch_size=batch_size)
            valid_dataloader = DataLoader(valid_dataset,
                                            batch_size=batch_size,
                                            collate_fn=padder)
            test_dataset = dataset_cls(filepath=filepath,
                                        split="test",
                                        tokenizer=tokenizer,
                                        max_length=max_length,
                                        multicol_only=False,
                                        device=device)
            test_dataloader = DataLoader(test_dataset,
                                            batch_size=batch_size,
                                            collate_fn=padder)
            unlabeled_dataset = aug_dataset_cls(filepath=filepath,
                                    split="unlabeled",
                                    tokenizer=tokenizer,
                                    max_length=max_length,
                                    multicol_only=False,
                                    train_ratio=1.0,
                                    device=device,
                                    augment_op=args.augment_op,
                                    num_aug=args.num_aug)
            unlabeled_sampler = RandomSampler(unlabeled_dataset)
            unlabeled_dataloader = DataLoader(unlabeled_dataset,
                                        sampler=unlabeled_sampler,
                                        batch_size=batch_size * args.u_ratio)
            
        else:
            raise ValueError("task name must be either sato or turl.")


    t_total = len(train_dataloader) * num_train_epochs
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay":
            0.0
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay":
            0.0
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=t_total)

    if "sato" in task:
        loss_fn = CrossEntropyLoss()
    elif "turl" in task:
        loss_fn = BCEWithLogitsLoss()
    else:
        raise ValueError("task name must be either sato or turl.")
    set_seed(args.random_seed)
    
    # model, optimizer, train_dataloader, unlabeled_dataloader, valid_dataloader, scheduler = accelerator.prepare(
    #     model, optimizer, train_dataloader, unlabeled_dataloader, valid_dataloader, scheduler
    # )

    model = model.cuda()
    # Best validation score could be zero
    best_vl_micro_f1 = -1
    best_vl_macro_f1 = -1
    best_vl_micro_f1s_epoch = -1
    best_vl_macro_f1s_epoch = -1
    last_vl_loss = 1e10
    loss_info_list = []
    eval_dict = []
    stuck_epoch = 0
    
    if 'bl' in args.ssl or 'cpl' in args.ssl:
        # train_epoch = 0
        unlabeled_epoch = 0
        # train_dataloader.sampler.set_epoch(train_epoch)
        # unlabeled_dataloader.sampler.set_epoch(unlabeled_epoch)
        # train_iter = iter(train_dataloader)
        unlabeled_iter = iter(unlabeled_dataloader)
    for epoch in range(num_train_epochs):
        
        t1 = time()
        print("Epoch", epoch, "starts")
        tr_loss = 0.
        tr_pred_list = []
        tr_true_list = []
        vl_pred_list = []
        vl_true_list = []
        tr_loss_l = 0.
        tr_loss_u = 0.

        vl_loss = 0.
        # device = accelerator.device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if 'bl' in args.ssl or 'cpl' in args.ssl:
            mixed_batches, total_l_cols, total_u_cols = create_bl_batches(task, epoch, 
                                                    train_dataset, unlabeled_dataset,
                                                    padder,
                                                    num_aug=args.num_aug,
                                                    u_ratio=args.u_ratio,
                                                    batch_size=args.batch_size // 2)
            if 'cpl' in args.ssl:
                selected_label = torch.ones((total_u_cols,), dtype=torch.long, ) * -1
                selected_label = selected_label.to(device)
                classwise_acc = torch.zeros((args.num_classes,)).to(device)

            model.train()
            for batch_idx, batch in enumerate(mixed_batches):
            
                x = batch['data'].T
                y = batch['label']
                # two batches of labeled and two batches of unlabeled
                batch_size = x.size()[0] // (args.num_aug * args.u_ratio + 2)
                x_l_ori = x[:batch_size]
                x_l_aug = x[batch_size:batch_size * 2]
                u_w = x[batch_size * 2:batch_size * (args.u_ratio+2)]
                u_s = x[batch_size * (args.u_ratio+2):]
                x_u_idx = batch['idx'][batch_size * 2:batch_size * (args.u_ratio+2)]
                if 'labelednoda' in args.ssl:
                    inputs = torch.cat((x_l_ori, u_w, u_s)).to(device)
                    cls_indexes_l = torch.nonzero(x_l_ori == tokenizer.cls_token_id)
                else:
                    inputs = torch.cat((x_l_aug, u_w, u_s)).to(device)
                    cls_indexes_l = torch.nonzero(x_l_aug == tokenizer.cls_token_id)
                cls_indexes_u_w = torch.nonzero(u_w == tokenizer.cls_token_id)
                cls_indexes_u_s = torch.nonzero(u_s == tokenizer.cls_token_id)
                if task == 'turl-re':
                    cls_indexes_u_w = cls_indexes_u_w[cls_indexes_u_w[:,1] > 0]
                    cls_indexes_u_s = cls_indexes_u_s[cls_indexes_u_s[:,1] > 0]
                    
                logits = model(inputs)
                logits_l = get_column_logits(logits[:batch_size],  cls_indexes_l)
                logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
                del logits
                logits_u_w = get_column_logits(logits_u_w, cls_indexes_u_w)
                logits_u_s = get_column_logits(logits_u_s, cls_indexes_u_s)
                if task not in ['turl', 'turl-re']:
                    loss_l = loss_fn(logits_l.to(device), y.to(device).long())
                else:
                    loss_l = loss_fn(logits_l, y)
                
                pseudo_label = torch.softmax(logits_u_w.detach()/args.temperature, dim=-1)
                max_probs, pred_u = torch.max(pseudo_label, dim=-1)
                if 'cpl' in args.ssl:
                    pseudo_counter = Counter(selected_label.tolist())
                    if max(pseudo_counter.values()) < total_u_cols:  
                        for i in range(args.num_classes):
                            classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())
                    # Calc the accuracy for each class and then adjust the cutoff confidence score
                    mask = max_probs.ge(args.p_cutoff * (classwise_acc[pred_u] / (2. - classwise_acc[pred_u]))).float() 
                    select = max_probs.ge(args.p_cutoff).long()
                    flat_x_u_idx = [u_col_idx for u_table_idx in x_u_idx for u_col_idx in u_table_idx]
                    flat_x_u_idx = torch.LongTensor(flat_x_u_idx).to(device)
                    if flat_x_u_idx[select == 1].nelement() != 0:
                        selected_label[flat_x_u_idx[select == 1]] = pred_u.long()[select == 1]

                else:
                    mask = max_probs.ge(args.p_cutoff).float()
                
                if 'mseloss' in args.ssl:
                    loss_u = (F.mse_loss(logits_u_s, pseudo_label) * mask).mean()
                else:
                    loss_u = (loss_fn(logits_u_s, pseudo_label) * mask).mean()
                loss = loss_l + args.u_lambda * loss_u
                loss.backward()
                tr_loss += loss.item()
                tr_loss_l += loss_l.item()
                tr_loss_u += loss_u.item()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                # break
        else:
            mixed_batches =  create_ssl_batches(epoch, 
                                                    train_dataset, unlabeled_dataset,
                                                    padder,
                                                    num_aug=args.num_aug,
                                                    batch_size=args.batch_size // 2)
            model.train()
            for batch_idx, batch in enumerate(mixed_batches):
                
                loss, tr_pred, loss_l, loss_u = ssl_with_ps(epoch, task, args.ssl, model, tokenizer.cls_token_id, loss_fn, batch, args.num_aug, u_lambda=args.u_lambda)
                # tr_pred_list += tr_pred.cpu().detach().numpy().tolist()
                # tr_true_list += batch["label"].cpu().detach().numpy().tolist()
                
                # accelerator.backward(loss)
                loss.backward()
                tr_loss += loss.item()
                tr_loss_l += loss_l
                tr_loss_u += loss_u
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
        tr_loss /= (len(train_dataset) / batch_size)
        tr_loss_l /= (len(train_dataset) / batch_size)
        tr_loss_u /= (len(train_dataset) / batch_size)


        if "sato" in task:
            tr_micro_f1 = f1_score(tr_true_list,
                                    tr_pred_list,
                                    average="micro")
            tr_macro_f1 = f1_score(tr_true_list,
                                    tr_pred_list,
                                    average="macro")
            tr_class_f1 = f1_score(tr_true_list,
                                    tr_pred_list,
                                    average=None,
                                    labels=np.arange(args.num_classes))
        elif "turl" in task and "popl" not in task:
            tr_micro_f1, tr_macro_f1, tr_class_f1, _ = f1_score_multilabel(
                tr_true_list, tr_pred_list)
    
        # Validation
        model.eval()
        # with accelerator.main_process_first():
        if True:
            # device = accelerator.device
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            for batch_idx, batch in enumerate(valid_dataloader):
                batch["data"] = batch["data"].to(device)
                cls_indexes = torch.nonzero(
                    batch["data"].T == tokenizer.cls_token_id)
                logits = model(batch["data"].T)
                if len(logits.shape) == 2:
                    logits = logits.unsqueeze(0)
                filtered_logits = torch.zeros(cls_indexes.shape[0],
                                            logits.shape[2]).to(device)
                for n in range(cls_indexes.shape[0]):
                    i, j = cls_indexes[n]
                    logit_n = logits[i, j, :]
                    filtered_logits[n] = logit_n
                if "sato" in task:
                    vl_pred_list += filtered_logits.argmax(
                        1).cpu().detach().numpy().tolist()
                    vl_true_list += batch["label"].cpu().detach().numpy(
                    ).tolist()
                elif "turl" in task:
                    if task == "turl-re":
                        all_preds = (filtered_logits >= math.log(0.5)
                                    ).int().detach().cpu().numpy()
                        all_labels = batch["label"].cpu().detach().numpy()
                        idxes = np.where(all_labels > 0)[0]
                        vl_pred_list += all_preds[idxes, :].tolist()
                        vl_true_list += all_labels[idxes, :].tolist()
                    elif task == "turl":
                        # Threshold value = 0.5
                        vl_pred_list += (filtered_logits >= math.log(0.5)
                                        ).int().detach().cpu().tolist()
                        vl_true_list += batch["label"].cpu().detach(
                        ).numpy().tolist()

                if "sato" in task:
                    loss = loss_fn(filtered_logits, batch["label"])
                elif "turl" in task:
                    loss = loss_fn(filtered_logits, batch["label"].float())

                vl_loss += loss.item()

            vl_loss /= (len(valid_dataset) / batch_size)
            if "sato" in task:
                vl_micro_f1 = f1_score(vl_true_list,
                                        vl_pred_list,
                                        average="micro")
                vl_macro_f1 = f1_score(vl_true_list,
                                        vl_pred_list,
                                        average="macro")
                vl_class_f1 = f1_score(vl_true_list,
                                        vl_pred_list,
                                        average=None,
                                        labels=np.arange(args.num_classes))
            elif "turl" in task:
                vl_micro_f1, vl_macro_f1, vl_class_f1, _ = f1_score_multilabel(
                    vl_true_list, vl_pred_list)
            
            # print(
            #         "Epoch {} ({}): tr_loss={:.7f} tr_macro_f1={:.4f} tr_micro_f1={:.4f}"
            #         .format(epoch, task, tr_loss, tr_macro_f1, tr_micro_f1),
            #         "vl_loss={:.7f} vl_macro_f1={:.4f} vl_micro_f1={:.4f}"
            #         .format(vl_loss, vl_macro_f1, vl_micro_f1))

           
            if not args.eval_test:
                if vl_micro_f1 > best_vl_micro_f1:
                    best_vl_micro_f1 = vl_micro_f1
                    model_savepath = "{}_best_micro_f1.pt".format(tag_name)
                    torch.save(model.state_dict(), model_savepath)

                if vl_macro_f1 > best_vl_macro_f1:
                    best_vl_macro_f1 = vl_macro_f1
                    model_savepath = "{}_best_macro_f1.pt".format(tag_name)
                    torch.save(model.state_dict(), model_savepath)

                loss_info_list.append([
                    tr_loss, tr_macro_f1, tr_micro_f1, vl_loss, vl_macro_f1,
                    vl_micro_f1
                ])
                t2 = time()
                print(
                    "Epoch {} ({}): tr_loss={:.7f} tr_macro_f1={:.4f} tr_micro_f1={:.4f} "
                    .format(epoch, task, tr_loss, tr_macro_f1, tr_micro_f1),
                    "vl_loss={:.7f} vl_macro_f1={:.4f} vl_micro_f1={:.4f} ({:.2f} sec.)"
                    .format(vl_loss, vl_macro_f1, vl_micro_f1, (t2 - t1)))
            else:
                ts_pred_list = []
                ts_true_list = []
                # Test
                for batch_idx, batch in enumerate(test_dataloader):
                    batch["data"] = batch["data"].to(device)
                    cls_indexes = torch.nonzero(
                            batch["data"].T == tokenizer.cls_token_id)
                    logits = model(batch["data"].T)
                    if len(logits.shape) == 2:
                        logits = logits.unsqueeze(0)
                    filtered_logits = torch.zeros(cls_indexes.shape[0],
                                                logits.shape[2]).to(device)
                    for n in range(cls_indexes.shape[0]):
                        i, j = cls_indexes[n]
                        logit_n = logits[i, j, :]
                        filtered_logits[n] = logit_n
                    if "sato" in task:
                        ts_pred_list += filtered_logits.argmax(
                            1).cpu().detach().numpy().tolist()
                        ts_true_list += batch["label"].cpu().detach().numpy(
                        ).tolist()
                    elif "turl" in task:
                        if "turl-re" in task:  # turl-re-colpair
                            all_preds = (filtered_logits >= math.log(0.5)
                                        ).int().detach().cpu().numpy()
                            all_labels = batch["label"].cpu().detach().numpy()
                            idxes = np.where(all_labels > 0)[0]
                            ts_pred_list += all_preds[idxes, :].tolist()
                            ts_true_list += all_labels[idxes, :].tolist()
                        elif task == "turl":
                            ts_pred_list += (filtered_logits >= math.log(0.5)
                                            ).int().detach().cpu().tolist()
                            ts_true_list += batch["label"].cpu().detach(
                            ).numpy().tolist()

                if "sato" in task:
                    ts_micro_f1 = f1_score(ts_true_list,
                                        ts_pred_list,
                                        average="micro")
                    ts_macro_f1 = f1_score(ts_true_list,
                                        ts_pred_list,
                                        average="macro")
                    ts_class_f1 = f1_score(ts_true_list,
                                        ts_pred_list,
                                        average=None,
                                        labels=np.arange(78))
                    ts_conf_mat = confusion_matrix(ts_true_list,
                                                ts_pred_list,
                                                labels=np.arange(78))
                elif "turl" in task:
                    ts_micro_f1, ts_macro_f1, ts_class_f1, ts_conf_mat = f1_score_multilabel(
                        ts_true_list, ts_pred_list)
                
                    

                t2 = time()
                if vl_micro_f1 >= best_vl_micro_f1:
                    best_vl_micro_f1 = vl_micro_f1
                    best_vl_micro_f1s_epoch = epoch
                if vl_macro_f1 >= best_vl_macro_f1:
                    best_vl_macro_f1 = vl_macro_f1
                    best_vl_macro_f1s_epoch = epoch
                loss_info_list.append([
                    tr_loss, tr_loss_l, tr_loss_u, tr_macro_f1, tr_micro_f1, vl_loss, vl_macro_f1,
                    vl_micro_f1, ts_macro_f1, ts_micro_f1,
                    best_vl_macro_f1s_epoch, best_vl_micro_f1s_epoch
                ])
                
                print(
                    "Epoch {} ({}): tr_loss={:.7f} tr_loss_l={:.7f} tr_loss_u={:.7f} tr_macro_f1={:.4f} tr_micro_f1={:.4f}"
                    .format(epoch, task, tr_loss, tr_loss_l, tr_loss_u, tr_macro_f1, tr_micro_f1),
                    "vl_loss={:.7f} vl_macro_f1={:.4f} vl_micro_f1={:.4f}"
                    .format(vl_loss, vl_macro_f1, vl_micro_f1, (t2 - t1)),
                    "ts_macro_f1={:.4f} ts_micro_f1={:.4f} ({:.2f} sec.)"
                    .format(ts_macro_f1, ts_micro_f1, t2-t1))

                metrics_dict = {'epoch': epoch, 'tr_loss':tr_loss, 'tr_loss_l':tr_loss_l, 'tr_loss_u':tr_loss_u, 'tr_macro_f1':tr_macro_f1, 'tr_micro_f1': tr_micro_f1,
                    'vl_loss':vl_loss, 'vl_macro_f1':vl_macro_f1, 'vl_micro_f1':vl_micro_f1, #'vl_class_f1':vl_class_f1,
                    'ts_macro_f1':ts_macro_f1, 'ts_micro_f1':ts_micro_f1, #'ts_class_f1':ts_class_f1, 'ts_conf_mat':ts_conf_mat,
                    'time': t2-t1
                }
                # mlflow.log_metrics(metrics_dict)
                eval_dict.append(metrics_dict)
                if type(ts_class_f1) != list:
                    ts_class_f1 = ts_class_f1.tolist()
                eval_dict[epoch]["ts_class_f1"] = ts_class_f1
                if type(ts_conf_mat) != list:
                    ts_conf_mat = ts_conf_mat.tolist()
                eval_dict[epoch]["confusion_matrix"] = ts_conf_mat
                
            if vl_loss == last_vl_loss:
                stuck_epoch += 1
            else:
                stuck_epoch = 0
                last_vl_loss = vl_loss
            if stuck_epoch > 1:
                break
        

    # with accelerator.main_process_first():
    if True:
        if args.eval_test:
            loss_info_df = pd.DataFrame(loss_info_list,
                                        columns=[
                                            "tr_loss", "tr_loss_l", "tr_loss_u",  
                                            "tr_f1_macro_f1",
                                            "tr_f1_micro_f1", "vl_loss",
                                            "vl_f1_macro_f1", "vl_f1_micro_f1",
                                            "ts_macro_f1", "ts_micro_f1",
                                            "best_vl_macro_f1_epoch", "best_vl_micro_f1_epoch"
                                        ])
            loss_info_df.to_csv("{}_loss_info.csv".format(tag_name))
            output_filepath = "{}_eval.json".format(tag_name)
            with open(output_filepath, "w") as fout:
                json.dump(eval_dict, fout)

        else:
            loss_info_df = pd.DataFrame(loss_info_list,
                                        columns=[
                                            "tr_loss", "tr_f1_macro_f1",
                                            "tr_f1_micro_f1", "vl_loss",
                                            "vl_f1_macro_f1", "vl_f1_micro_f1"
                                        ])
            loss_info_df.to_csv("{}_loss_info.csv".format(tag_name))
