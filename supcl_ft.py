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

import pytrec_eval
import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from accelerate import Accelerator

torch.backends.cuda.matmul.allow_tf32 = True

from watchog.dataset import (
    # collate_fn,
    TURLColTypeTablewiseDataset,
    TURLRelExtTablewiseDataset,
    SatoCVTablewiseDataset,
    ColPoplTablewiseDataset
)

from watchog.dataset import TableDataset, SupCLTableDataset, SemtableCVTablewiseDataset, GittablesTablewiseDataset, GittablesCVTablewiseDataset
from watchog.model import BertMultiPairPooler, BertForMultiOutputClassification, BertForMultiOutputClassificationColPopl
from watchog.model import SupCLforTable, UnsupCLforTable, lm_mp
from watchog.utils import load_checkpoint, f1_score_multilabel, collate_fn, get_col_pred, ColPoplEvaluator
from watchog.utils import task_num_class_dict
from accelerate import DistributedDataParallelKwargs


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    

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
    parser.add_argument("--multi_gpu",
                        action="store_true",
                        default=False,
                        help="Use multiple GPU")
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
                            "gt-dbpedia0", "gt-dbpedia1", "gt-dbpedia2", "gt-dbpedia3", "gt-dbpedia4",
                            "gt-dbpedia-all0", "gt-dbpedia-all1", "gt-dbpedia-all2", "gt-dbpedia-all3", "gt-dbpedia-all4",
                            "gt-schema-all0", "gt-schema-all1", "gt-schema-all2", "gt-schema-all3", "gt-schema-all4",
                            "gt-semtab22-dbpedia", "gt-semtab22-dbpedia0", "gt-semtab22-dbpedia1", "gt-semtab22-dbpedia2", "gt-semtab22-dbpedia3", "gt-semtab22-dbpedia4",
                            "gt-semtab22-dbpedia-all", "gt-semtab22-dbpedia-all0", "gt-semtab22-dbpedia-all1", "gt-semtab22-dbpedia-all2", "gt-semtab22-dbpedia-all3", "gt-semtab22-dbpedia-all4",
                            "gt-semtab22-schema-class-all", "gt-semtab22-schema-property-all",
                            "turl", "turl-re", "col-popl-1", "col-popl-2", "col-popl-3", "row-popl",
                            "col-popl-turl-0", "col-popl-turl-1", "col-popl-turl-2",
                            "col-popl-turl-mdonly-0", "col-popl-turl-mdonly-1", "col-popl-turl-mdonly-2"
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

    args = parser.parse_args()
    task = args.task
    if args.small_tag != "":
        args.eval_test = True
    
    args.num_classes = task_num_class_dict[task]
    if args.colpair:
        assert "turl-re" == task, "colpair can be only used for Relation Extraction"
    if args.metadata:
        assert "turl-re" == task or "turl" == task, "metadata can be only used for TURL datasets"
    if "col-popl":
        # metrics = {
        #     "accuracy": CategoricalAccuracy(tie_break=True),
        # }
        if args.train_n_seed_cols != -1:
            if "col-popl" in task:
                assert args.train_n_seed_cols == int(task[-1]),  "# of seed columns must match"

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
    elif args.train_n_seed_cols == -1 and 'popl' in task:
        taskname = "{}-mix".format(task)
    else:
        taskname = "".join(task)

    if args.eval_test:
        if args.from_scratch:
            tag_name = "outputs/{}/{}_{}-bs{}-ml{}-ne{}-do{}{}".format(
                taskname, tag_name_col, "{}-fromscratch".format(shortcut_name),
                batch_size, max_length, num_train_epochs, args.dropout_prob, 
                '-rs{}'.format(args.random_seed) if args.random_seed != 4649 else '')
        else:
            tag_name = "outputs/{}/{}_{}_{}-bs{}-ml{}-ne{}-do{}{}".format(
                taskname, args.cl_tag.replace('/', '-'), tag_name_col, shortcut_name, 
                batch_size, max_length, num_train_epochs, args.dropout_prob,
                '-rs{}'.format(args.random_seed) if args.random_seed != 4649 else '')
    else:
        if args.from_scratch:
            tag_name = "model/{}_{}_{}-bs{}-ml{}-ne{}-do{}{}".format(
                taskname, tag_name_col, "{}-fromscratch".format(shortcut_name),
                batch_size, max_length, num_train_epochs, args.dropout_prob,
                '-rs{}'.format(args.random_seed) if args.random_seed != 4649 else '')
        else:
            tag_name = "model/{}_{}_{}_{}-bs{}-ml{}-ne{}-do{}{}".format(
                args.cl_tag.replace('/', '-'), taskname, tag_name_col, shortcut_name, 
                batch_size, max_length, num_train_epochs, args.dropout_prob,
                '-rs{}'.format(args.random_seed) if args.random_seed != 4649 else '')
    if args.eval_test:
        if args.small_tag != '':
            tag_name = tag_name.replace('outputs', 'small_outputs')
            tag_name += '-' + args.small_tag
    
    print(tag_name)

    dirpath = os.path.dirname(tag_name)
    if not os.path.exists(dirpath):
        print("{} not exists. Created".format(dirpath))
        os.makedirs(dirpath)
    
    if args.fp16:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    # accelerator = Accelerator(mixed_precision="no" if not args.fp16 else "fp16")   
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision="no" if not args.fp16 else "fp16", kwargs_handlers=[ddp_kwargs])

    device = accelerator.device
    ckpt_path = '{}/{}.pt'.format(args.pretrained_ckpt_path, args.cl_tag)
    # ckpt_path = '/efs/checkpoints/{}.pt'.format(args.cl_tag)
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
    elif "col-popl" in task:
        model = BertForMultiOutputClassificationColPopl(ckpt_hp, device=device, lm=ckpt['hp'].lm, n_seed_cols=int(task[i][-1]), cls_for_md="md" in task)
    else:
        model = BertForMultiOutputClassification(ckpt_hp, device=device, lm=ckpt['hp'].lm)
        

    if not args.from_scratch:
        model.bert = pre_model.bert

    if task == "turl-re" and args.colpair and ckpt['hp'].lm != 'distilbert':
        config = BertConfig.from_pretrained(lm_mp[ckpt['hp'].lm])
        model.bert.pooler = BertMultiPairPooler(config).to(device)
        print("Use column-pair pooling")
        # print(type(model.bert.pooler), model.bert.pooler.hidden_size)

    del pre_model
        
    padder = collate_fn(trainset.tokenizer.pad_token_id)
    with accelerator.main_process_first():
    # if True:
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
            train_dataset = dataset_cls(cv=cv,
                                        split="train",
                                        tokenizer=tokenizer,
                                        max_length=max_length,
                                        multicol_only=multicol_only,
                                        train_ratio=1.0,
                                        device=device,
                                        small_tag=args.small_tag)
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
                                            batch_size=batch_size,
                                        #   collate_fn=collate_fn)
                                        collate_fn=padder)
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
            
        elif "gt-dbpedia" in task or "gt-schema" in task or "gt-semtab" in task:
            if 'dbpedia' in task:
                if 'semtab22' in task:
                    src = 'dbpedia_property'
                else:
                    src = 'dbpedia'
            else:
                if 'semtab22' in task:
                    if 'schema-property' in task:
                        src = 'schema_property'
                    else:
                        src = 'schema_class'
                else:
                    src = 'schema'
            if task[-1] in "01234":
                cv = int(task[-1])
                dataset_cls = GittablesCVTablewiseDataset
                train_dataset = dataset_cls(cv=cv,
                                            split="train",
                                            src=src,
                                            tokenizer=tokenizer,
                                            max_length=max_length,
                                            gt_only='all' not in task,
                                            device=device,
                                            small_tag=args.small_tag)
                valid_dataset = dataset_cls(cv=cv,
                                            split="valid", src=src,
                                            tokenizer=tokenizer,
                                            max_length=max_length,
                                            gt_only='all' not in task,
                                            device=device,
                                            small_tag=args.small_tag)

                train_sampler = RandomSampler(train_dataset)
                train_dataloader = DataLoader(train_dataset,
                                                sampler=train_sampler,
                                                batch_size=batch_size,
                                            #   collate_fn=collate_fn)
                                            collate_fn=padder)
                valid_dataloader = DataLoader(valid_dataset,
                                                batch_size=batch_size,
                                            #   collate_fn=collate_fn)
                                            collate_fn=padder)
                test_dataset = dataset_cls(cv=cv,
                                           split="valid", src=src,
                                            tokenizer=tokenizer,
                                            max_length=max_length,
                                            gt_only='all' not in task,
                                            device=device)
                test_dataloader = DataLoader(test_dataset,
                                                batch_size=batch_size,
                                                collate_fn=padder)    
            else:
                dataset_cls = GittablesTablewiseDataset
                train_dataset = dataset_cls(split="train",
                                            src=src,
                                            tokenizer=tokenizer,
                                            max_length=max_length,
                                            gt_only='all' not in task,
                                            device=device,
                                            base_dirpath="./data/gittables_semtab22",
                                            base_tag='',
                                            small_tag=args.small_tag)
                valid_dataset = dataset_cls(split="test", src=src,
                                            tokenizer=tokenizer,
                                            max_length=max_length,
                                            gt_only='all' not in task,
                                            device=device,
                                            base_dirpath="./data/gittables_semtab22",
                                            base_tag='',
                                            small_tag=args.small_tag)

                train_sampler = RandomSampler(train_dataset)
                train_dataloader = DataLoader(train_dataset,
                                                sampler=train_sampler,
                                                batch_size=batch_size,
                                            #   collate_fn=collate_fn)
                                            collate_fn=padder)
                valid_dataloader = DataLoader(valid_dataset,
                                                batch_size=batch_size,
                                            #   collate_fn=collate_fn)
                                            collate_fn=padder)
                test_dataset = dataset_cls(split="test", src=src,
                                            tokenizer=tokenizer,
                                            max_length=max_length,
                                            gt_only='all' not in task,
                                            base_dirpath="./data/gittables_semtab22",
                                            base_tag='',
                                            device=device)
                test_dataloader = DataLoader(test_dataset,
                                                batch_size=batch_size,
                                                collate_fn=padder)    
                         
        elif "popl" in task:
            if "col-popl" in task:
                padder = collate_fn(trainset.tokenizer.pad_token_id, data_only=False)
            
                if args.train_n_seed_cols == -1:
                    train_filepath = args.data_path + "/col_popl_turl/train_col_popl_seedmix.jsonl"    
                else:
                    train_filepath = args.data_path + "/col_popl_turl/train_col_popl_seed{}.jsonl".format(args.train_n_seed_cols)
                test_n_seed_cols = int(task[-1])
                valid_filepath = args.data_path + "/col_popl_turl/dev_col_popl_seed{}.jsonl".format(test_n_seed_cols)
                test_filepath = args.data_path + "/col_popl_turl/test_col_popl_seed{}.jsonl".format(test_n_seed_cols)
                dataset_cls = ColPoplTablewiseDataset
            else:
                raise ValueError("not col polulation.")

            train_dataset = dataset_cls(filepath=train_filepath,
                                        split="train",
                                        tokenizer=tokenizer,
                                        max_length=max_length,
                                        multicol_only=False,
                                        train_ratio=train_ratio,
                                        use_content="mdonly" not in task,
                                        use_metadata="md" in task,
                                        device=device)
            valid_dataset = dataset_cls(filepath=valid_filepath,
                                        split="dev",
                                        tokenizer=tokenizer,
                                        max_length=max_length,
                                        multicol_only=False,
                                        use_content="mdonly" not in task,
                                        use_metadata="md" in task,
                                        device=device)

            # Can be the same
            train_sampler = RandomSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset,
                                          sampler=train_sampler,
                                          batch_size=batch_size,
                                          collate_fn=padder)
            valid_dataloader = DataLoader(valid_dataset,
                                          batch_size=batch_size,
                                          collate_fn=padder)
            test_dataset = dataset_cls(filepath=test_filepath,
                                       split="test",
                                       tokenizer=tokenizer,
                                       max_length=max_length,
                                       multicol_only=False,
                                       use_content="mdonly" not in task,
                                       use_metadata="md" in task,
                                       device=device)
            test_dataloader = DataLoader(test_dataset,
                                         batch_size=batch_size,
                                         collate_fn=padder)
            valid_evaluator = ColPoplEvaluator(valid_dataset, task_type="col-popl-turl")
            test_evaluator = ColPoplEvaluator(test_dataset, task_type="col-popl-turl")    
            
        elif "turl" in task:
            if task in ["turl"]:
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
            else:
                raise ValueError("turl tasks must be turl or turl-re.")

            train_dataset = dataset_cls(filepath=filepath,
                                        split="train",
                                        tokenizer=tokenizer,
                                        max_length=max_length,
                                        multicol_only=False,
                                        train_ratio=1.0,
                                        device=device)
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
                                            batch_size=batch_size,
                                            collate_fn=padder)
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
    elif "popl" in task:
        loss_fn = CrossEntropyLoss()
    elif "turl" in task:
        loss_fn = BCEWithLogitsLoss()
    else:
        raise ValueError("task name must be either sato or turl.")
    set_seed(args.random_seed)
    
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )

    model = model.to(device)
    # model = model.cuda()
    # Best validation score could be zero
    best_vl_micro_f1 = -1
    best_vl_macro_f1 = -1
    best_vl_micro_f1s_epoch = -1
    best_vl_macro_f1s_epoch = -1
    loss_info_list = []
    eval_dict = []
    for epoch in range(num_train_epochs):
        t1 = time()
        print("Epoch", epoch, "starts")
        model.train()
        tr_loss = 0.
        if "col-popl" in task:
            tr_pred_list = {}
            tr_true_list = {}
            vl_pred_list = {}
            vl_true_list = {}
        else:
            tr_pred_list = []
            tr_true_list = []
            vl_pred_list = []
            vl_true_list = []

        vl_loss = 0.
        device = accelerator.device
        for batch_idx, batch in enumerate(train_dataloader):

            cls_indexes = torch.nonzero(
                batch["data"].T == tokenizer.cls_token_id)
            if "col-popl" in task:
                logits, = model(batch["data"].T, cls_indexes)
                labels = batch["label"].T
                filtered_logits = []
                labels_1d = []
                all_labels = []
                for _, x in enumerate(logits):
                    filtered_logits.append(x.expand(sum(labels[_]>-1), args.num_classes))
                    labels_1d.extend(labels[_][labels[_]>-1])
                    all_labels.append(labels[_][labels[_]>-1].cpu().detach().numpy())
                filtered_logits = torch.cat(filtered_logits, dim=0).to(device)
                labels_1d = torch.as_tensor(labels_1d).to(device)
                all_preds = get_col_pred(logits, labels, batch["idx"], top_k=-1)#.cpu().detach().numpy()
                tr_pred_list.update(all_preds)
                loss = loss_fn(filtered_logits, labels_1d)
            else:
                logits, = model(batch["data"].T)
                if len(logits.shape) == 2:
                    logits = logits.unsqueeze(0)
                
                filtered_logits = torch.zeros(cls_indexes.shape[0],
                                            logits.shape[2]).to(device)
                for n in range(cls_indexes.shape[0]):
                    i, j = cls_indexes[n]
                    logit_n = logits[i, j, :]
                    filtered_logits[n] = logit_n
                 if "sato" in task or "gt-" in task:
                    if 'gt-' in task and '-all' in task:
                        labels = batch["label"].T
                        new_filtered_logits = []
                        for _, x in enumerate(filtered_logits):
                            if labels[_] > -1:
                                new_filtered_logits.append(x)
                        new_filtered_logits = torch.stack(new_filtered_logits, dim=0).to(device)
                        
                        labels_1d = labels[labels > -1]
                        all_labels = labels[labels > -1].cpu().detach().numpy().tolist()
                        tr_pred_list += new_filtered_logits.argmax(
                            1).cpu().detach().numpy().tolist()
                        tr_true_list += all_labels        
                        loss = loss_fn(new_filtered_logits, labels_1d)
                    else:
                        tr_pred_list += filtered_logits.argmax(
                            1).cpu().detach().numpy().tolist()
                        tr_true_list += batch["label"].cpu().detach().numpy().tolist()
                        loss = loss_fn(filtered_logits, batch["label"])

                elif "turl" in task:
                    if task == "turl-re":
                        all_preds = (filtered_logits >= math.log(0.5)
                                    ).int().detach().cpu().numpy()
                        all_labels = batch["label"].cpu().detach().numpy()
                        # Ignore the very first CLS token
                        idxes = np.where(all_labels > 0)[0]
                        tr_pred_list += all_preds[idxes, :].tolist()
                        tr_true_list += all_labels[idxes, :].tolist()
                    elif task == "turl":
                        # Threshold value = 0.5
                        tr_pred_list += (filtered_logits >= math.log(0.5)
                                        ).int().detach().cpu().tolist()
                        tr_true_list += batch["label"].cpu().detach(
                        ).numpy().tolist()
                        
                    loss = loss_fn(filtered_logits, batch["label"].float())

            accelerator.backward(loss)
            # loss.backward()
            tr_loss += loss.item()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
        tr_loss /= (len(train_dataset) / batch_size)


        if "sato" in task or "gt-" in task:
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
        with accelerator.main_process_first():
            device = accelerator.device
            for batch_idx, batch in enumerate(valid_dataloader):
                batch["data"] = batch["data"].to(device)
                cls_indexes = torch.nonzero(
                    batch["data"].T == tokenizer.cls_token_id)
                if "col-popl" in task:
                    logits, = model(batch["data"].T, cls_indexes) 
                    labels = batch["label"].T
                    filtered_logits = []
                    labels_1d = []
                    all_labels = []
                    for _, x in enumerate(logits):
                        filtered_logits.append(x.expand(sum(labels[_]>-1), args.num_classes))
                        labels_1d.extend(labels[_][labels[_]>-1])
                        all_labels.append(labels[_][labels[_]>-1].cpu().detach().numpy())
                    filtered_logits = torch.cat(filtered_logits, dim=0).to(device)
                    labels_1d = torch.as_tensor(labels_1d).to(device)
                    all_preds = get_col_pred(logits, labels, batch["idx"], top_k=500)#.cpu().detach().numpy()
                    vl_pred_list.update(all_preds)
                    loss = loss_fn(filtered_logits, labels_1d)
                else:
                    logits, = model(batch["data"].T)
                    if len(logits.shape) == 2:
                        logits = logits.unsqueeze(0)
                    filtered_logits = torch.zeros(cls_indexes.shape[0],
                                                logits.shape[2]).to(device)
                    for n in range(cls_indexes.shape[0]):
                        i, j = cls_indexes[n]
                        logit_n = logits[i, j, :]
                        filtered_logits[n] = logit_n
                    if "sato" in task or "gt-" in task:
                        if 'gt-' in task and '-all' in task:
                            labels = batch["label"].T
                            new_filtered_logits = []
                            labels_1d = []
                            all_labels = []
                            for _, x in enumerate(filtered_logits):
                                if labels[_] > -1:
                                    new_filtered_logits.append(x)
                                 
                            new_filtered_logits = torch.stack(new_filtered_logits, dim=0).to(device)
                            labels_1d = labels[labels > -1]
                            all_labels = labels[labels > -1].cpu().detach().numpy().tolist()
                            
                            vl_pred_list += new_filtered_logits.argmax(
                                1).cpu().detach().numpy().tolist()
                            vl_true_list += all_labels
                            
                            loss = loss_fn(new_filtered_logits, labels_1d)
                        else:                        
                            vl_pred_list += filtered_logits.argmax(
                                1).cpu().detach().numpy().tolist()
                            vl_true_list += batch["label"].cpu().detach().numpy().tolist()
                            loss = loss_fn(filtered_logits, batch["label"])

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
                        loss = loss_fn(filtered_logits, batch["label"].float())

                vl_loss += loss.item()

            vl_loss /= (len(valid_dataset) / batch_size)
            if "sato" in task or "gt-" in task:
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
            elif "col-popl" in task:
                vl_map, vl_rpr, vl_ndcg_10, vl_ndcg_20,  _ = valid_evaluator.eval_one_run(vl_pred_list)
            elif "turl" in task:
                vl_micro_f1, vl_macro_f1, vl_class_f1, _ = f1_score_multilabel(
                    vl_true_list, vl_pred_list)
            
                
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
                if "popl" in task:
                    ts_pred_list = {}
                    ts_true_list = {}
                else:
                    ts_pred_list = []
                    ts_true_list = []
                # Test
                for batch_idx, batch in enumerate(test_dataloader):
                    batch["data"] = batch["data"].to(device)
                    cls_indexes = torch.nonzero(
                            batch["data"].T == tokenizer.cls_token_id)
                    if "popl" in task:
                        logits, = model(batch["data"].T, cls_indexes)
                        labels = batch["label"].T
                        filtered_logits = []
                        labels_1d = []
                        all_labels = []
                        for _, x in enumerate(logits):
                            filtered_logits.append(x.expand(sum(labels[_]>-1), args.num_classes))
                            labels_1d.extend(labels[_][labels[_]>-1])
                            all_labels.append(labels[_][labels[_]>-1].cpu().detach().numpy())
                        filtered_logits = torch.cat(filtered_logits, dim=0).to(device)
                        labels_1d = torch.as_tensor(labels_1d).to(device)
                        all_preds = get_col_pred(logits, labels, batch["idx"], top_k=500)#.cpu().detach().numpy()
                        ts_pred_list.update(all_preds)
                        
                    else:
                        logits, = model(batch["data"].T)
                        if len(logits.shape) == 2:
                            logits = logits.unsqueeze(0)
                        filtered_logits = torch.zeros(cls_indexes.shape[0],
                                                    logits.shape[2]).to(device)
                        for n in range(cls_indexes.shape[0]):
                            i, j = cls_indexes[n]
                            logit_n = logits[i, j, :]
                            filtered_logits[n] = logit_n
                        if "sato" in task or "gt-" in task:
                            if 'gt-' in task and '-all' in task:
                                labels = batch["label"].T
                                new_filtered_logits = []
                                for _, x in enumerate(filtered_logits):
                                    if labels[_] > -1:
                                        new_filtered_logits.append(x)
                                new_filtered_logits = torch.stack(new_filtered_logits, dim=0).to(device)
                                labels_1d = labels[labels > -1]
                                all_labels = labels[labels > -1].cpu().detach().numpy().tolist()
                                ts_pred_list += new_filtered_logits.argmax(
                                    1).cpu().detach().numpy().tolist()
                                ts_true_list += all_labels  
                                ts_prob_list += new_filtered_logits.cpu().detach().numpy().tolist()
                            else:
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

                if "sato" in task or "gt-" in task:
                    ts_micro_f1 = f1_score(ts_true_list,
                                        ts_pred_list,
                                        average="micro")
                    ts_macro_f1 = f1_score(ts_true_list,
                                        ts_pred_list,
                                        average="macro")
                    ts_class_f1 = f1_score(ts_true_list,
                                        ts_pred_list,
                                        average=None,
                                        labels=np.arange(args.num_classes))
                    ts_conf_mat = confusion_matrix(ts_true_list,
                                                ts_pred_list,
                                                labels=np.arange(args.num_classes))
                elif "col-popl" in task:
                    if epoch == num_train_epochs - 1:
                        ts_map, ts_rpr, ts_ndcg_10, ts_ndcg_20,  _ = test_evaluator.eval_one_run(ts_pred_list, "{}_trec_eval.json".format(tag_name))
                    else:
                        ts_map, ts_rpr, ts_ndcg_10, ts_ndcg_20,  _ = test_evaluator.eval_one_run(ts_pred_list)
                elif "turl" in task:
                    ts_micro_f1, ts_macro_f1, ts_class_f1, ts_conf_mat = f1_score_multilabel(
                        ts_true_list, ts_pred_list)

                t2 = time()
                if "popl" in task:
                    if vl_map >= best_vl_micro_f1:
                        best_vl_micro_f1 = vl_map
                        best_vl_micro_f1s_epoch = epoch
                    loss_info_list.append([
                        tr_loss, vl_loss, vl_map, vl_rpr, vl_ndcg_10, vl_ndcg_20, 
                        ts_map, ts_rpr, ts_ndcg_10, ts_ndcg_20, 
                        best_vl_micro_f1s_epoch
                    ])
                    
                    print(
                        "Epoch {} ({}): tr_loss={:.7f}" 
                        .format(epoch, task, tr_loss),
                        "vl_loss={:.7f} vl_map={:.4f} vl_rpr={:.4f} vl_ndcg_10={:.4f} vl_ndcg_20={:.4f}"
                        .format(vl_loss, vl_map, vl_rpr, vl_ndcg_10, vl_ndcg_20),
                        "ts_map={:.4f} ts_rpr={:.4f} ts_ndcg_10={:.4f} ts_ndcg_20={:.4f} ({:.2f} sec.)"
                        .format(ts_map, ts_rpr, ts_ndcg_10, ts_ndcg_20,  t2-t1))

                    metrics_dict = {'epoch': epoch, 'tr_loss':tr_loss, 
                        'vl_loss':vl_loss, 'vl_map': vl_map, 'vl_rpr': vl_rpr, 'vl_ndcg_10': vl_ndcg_10, 'vl_ndcg_20': vl_ndcg_20,
                        'ts_map':ts_map, 'ts_rpr': ts_rpr, 'ts_ndcg_10': ts_ndcg_10, 'ts_ndcg_20':ts_ndcg_20,
                        'time': t2-t1
                    }
                    mlflow.log_metrics(metrics_dict)
                    eval_dict.append(metrics_dict)
                else:
                    if vl_micro_f1 >= best_vl_micro_f1:
                        best_vl_micro_f1 = vl_micro_f1
                        best_vl_micro_f1s_epoch = epoch
                    if vl_macro_f1 >= best_vl_macro_f1:
                        best_vl_macro_f1 = vl_macro_f1
                        best_vl_macro_f1s_epoch = epoch
                    loss_info_list.append([
                        tr_loss, tr_macro_f1, tr_micro_f1, vl_loss, vl_macro_f1,
                        vl_micro_f1, ts_macro_f1, ts_micro_f1,
                        best_vl_macro_f1s_epoch, best_vl_micro_f1s_epoch
                    ])
                    
                    print(
                        "Epoch {} ({}): tr_loss={:.7f} tr_macro_f1={:.4f} tr_micro_f1={:.4f}"
                        .format(epoch, task, tr_loss, tr_macro_f1, tr_micro_f1),
                        "vl_loss={:.7f} vl_macro_f1={:.4f} vl_micro_f1={:.4f}"
                        .format(vl_loss, vl_macro_f1, vl_micro_f1, (t2 - t1)),
                        "ts_macro_f1={:.4f} ts_micro_f1={:.4f} ({:.2f} sec.)"
                        .format(ts_macro_f1, ts_micro_f1, t2-t1))

                    metrics_dict = {'epoch': epoch, 'tr_loss':tr_loss, 'tr_macro_f1':tr_macro_f1, 'tr_micro_f1': tr_micro_f1,
                        'vl_loss':vl_loss, 'vl_macro_f1':vl_macro_f1, 'vl_micro_f1':vl_micro_f1, #'vl_class_f1':vl_class_f1,
                        'ts_macro_f1':ts_macro_f1, 'ts_micro_f1':ts_micro_f1, #'ts_class_f1':ts_class_f1, 'ts_conf_mat':ts_conf_mat,
                        'time': t2-t1
                    }
                    mlflow.log_metrics(metrics_dict)
                    eval_dict.append(metrics_dict)
                    if type(ts_class_f1) != list:
                        ts_class_f1 = ts_class_f1.tolist()
                    eval_dict[epoch]["ts_class_f1"] = ts_class_f1
                    if type(ts_conf_mat) != list:
                        ts_conf_mat = ts_conf_mat.tolist()
                    eval_dict[epoch]["confusion_matrix"] = ts_conf_mat
            

    with accelerator.main_process_first():
        if args.eval_test:
            if "popl" in task:
                loss_info_df = pd.DataFrame(loss_info_list,
                                            columns=[
                                                "tr_loss", "vl_loss",
                                                "vl_map", "vl_rpr", "vl_ndcg_10", "vl_ndcg_20", 
                                                "ts_map", "ts_rpr", "ts_ndcg_10", "ts_ndcg_20",
                                                "best_vl_map_epoch"
                                            ])
            else:
                loss_info_df = pd.DataFrame(loss_info_list,
                                            columns=[
                                                "tr_loss", "tr_f1_macro_f1",
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
