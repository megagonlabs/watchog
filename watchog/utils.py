import torch
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.metrics import multilabel_confusion_matrix
from tqdm import tqdm
from collections import deque, Counter
from typing import List
import pytrec_eval


from .dataset import TableDataset
from .model import SupCLforTable, UnsupCLforTable
import random
import json
from collections import OrderedDict

BASE_DATA_PATH = '/efs/pretrain_datasets/'


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
        "reca0": 275,
        "reca1": 275,
        "reca2": 275,
        "reca3": 275,
        "reca4": 275,
        "turl": 255,
        "turl-re": 121,
        "col-popl-1": 127656,
        "col-popl-2": 127656,
        "col-popl-3": 127656,
        "col-popl-turl-0": 5407,
        "col-popl-turl-1": 5407,
        "col-popl-turl-2": 5407,
        "col-popl-turl-mdonly-0": 5407,
        "col-popl-turl-mdonly-1": 5407,
        "col-popl-turl-mdonly-2": 5407,
        "gt-dbpedia": 122,
        "gt-dbpedia-all": 122,
        "gt-dbpedia0": 122,
        "gt-dbpedia1": 122,
        "gt-dbpedia2": 122,
        "gt-dbpedia3": 122,
        "gt-dbpedia4": 122,
        "gt-dbpedia-all0": 122,
        "gt-dbpedia-all1": 122,
        "gt-dbpedia-all2": 122,
        "gt-dbpedia-all3": 122,
        "gt-dbpedia-all4": 122,
        "gt-schema-all0": 59,
        "gt-schema-all1": 59,
        "gt-schema-all2": 59,
        "gt-schema-all3": 59,
        "gt-schema-all4": 59,
        "gt-semtab22-dbpedia": 101,
        "gt-semtab22-dbpedia0": 101,
        "gt-semtab22-dbpedia1": 101,
        "gt-semtab22-dbpedia2": 101,
        "gt-semtab22-dbpedia3": 101,
        "gt-semtab22-dbpedia4": 101,
        "gt-semtab22-dbpedia-all": 101,
        "gt-semtab22-dbpedia-all0": 101,
        "gt-semtab22-dbpedia-all1": 101,
        "gt-semtab22-dbpedia-all2": 101,
        "gt-semtab22-dbpedia-all3": 101,
        "gt-semtab22-dbpedia-all4": 101,
        "gt-semtab22-schema-class-all": 21,
        "gt-semtab22-schema-property-all": 53
    }


def collate_fn(pad_token_id, data_only=True):
    '''padder for input batch'''

    def padder(samples):    
        data = torch.nn.utils.rnn.pad_sequence(
            [sample["data"] for sample in samples], padding_value=pad_token_id)
        if not data_only:
            label = torch.nn.utils.rnn.pad_sequence(
                [sample["label"] for sample in samples], padding_value=-1)
        else:
            label = torch.cat([sample["label"] for sample in samples])
        batch = {"data": data, "label": label}
        if "idx" in samples[0]:
            batch["idx"] = [sample["idx"] for sample in samples]
        if "cls_indexes" in samples[0]:
            cls_indexes = torch.nn.utils.rnn.pad_sequence(
                [sample["cls_indexes"] for sample in samples], padding_value=0)
            batch["cls_indexes"] = cls_indexes
        return batch
        
    return padder


def load_checkpoint(ckpt):
    """Load a model from a checkpoint.
    Args:
        ckpt (str): the model checkpoint path.

    Returns:
        SupCLforTable or UnsupCLforTable: the pre-trained model
        PretrainDataset: the dataset for pre-training the model
    """
    hp = ckpt['hp']
    if 'table_order' not in hp:
        setattr(hp, 'table_order', 'column')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if hp.mode in ['supcon', 'supcon_ddp']:
        model = SupCLforTable(hp, device=device, lm=hp.lm)
    else:
        model = UnsupCLforTable(hp, device=device, lm=hp.lm)
        
    model = model.to(device)
    try:
        model.load_state_dict(ckpt['model'])
    except:
        new_state_dict = OrderedDict()
        for k, v in ckpt['model'].items():
            name = k[7:]
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)

    if 'pretrain_data' not in hp:
        setattr(hp, 'pretrain_data', hp.task)
        
    ds_path = BASE_DATA_PATH + hp.pretrain_data
    print(hp)
    dataset = TableDataset.from_hp(ds_path, hp)

    return model, dataset
        


def f1_score_multilabel(true_list, pred_list):
    conf_mat = multilabel_confusion_matrix(np.array(true_list),
                                           np.array(pred_list))
    agg_conf_mat = conf_mat.sum(axis=0)
    # Note: Pos F1
    # [[TN FP], [FN, TP]] if we consider 1 as the positive class
    p = agg_conf_mat[1, 1] / agg_conf_mat[1, :].sum()
    r = agg_conf_mat[1, 1] / agg_conf_mat[:, 1].sum()
    
    micro_f1 = 2 * p * r / (p  + r) if (p + r) > 0 else 0.
    class_p = conf_mat[:, 1, 1] /  conf_mat[:, 1, :].sum(axis=1)
    class_r = conf_mat[:, 1, 1] /  conf_mat[:, :, 1].sum(axis=1)
    class_f1 = np.divide(2 * (class_p * class_r), class_p + class_r,
                         out=np.zeros_like(class_p), where=(class_p + class_r) != 0)
    class_f1 = np.nan_to_num(class_f1)
    macro_f1 = class_f1.mean()
    return (micro_f1, macro_f1, class_f1, conf_mat)


def get_col_pred(logits, labels, idx_list, top_k=-1):
    '''for column population'''
    pred_labels = {}
    out_prob_cp = torch.autograd.Variable(logits.clone(), requires_grad=False).cpu()
    for k, row_labels in enumerate(labels):
        # print(k, len(row_labels),  sum(row_labels > -1), row_labels)
        n_pred = sum(row_labels > -1) if top_k == -1 else top_k
        pred_row_labels = out_prob_cp[k][1:].argsort(dim=0, descending=True)[:n_pred].cpu().numpy()  # out_prob[0]: blank header
        pred_row_labels = [elem+1 for elem in pred_row_labels]  # add idx to 1 (for out_prob[0])
        pred_row_labels_prob = dict(zip(pred_row_labels, map(lambda x: out_prob_cp[k][x].item(), pred_row_labels)))
        pred_labels["Q" + str(idx_list[k])] = pred_row_labels_prob
    return pred_labels


h2i_fn = {"col-popl": "/efs/task_datasets/col_popl/h2idx.json",
          "col-popl-turl": "/efs/task_datasets/col_popl_turl/h2idx.json"}

class ColPoplEvaluator():
    '''for column population'''
    def __init__(self, dataset, task_type="col-popl"):
        self.qrel = {}
        for i, row in dataset.table_df.iterrows():
            q = "Q" + str(row["idx"])
            self.qrel[q] = {}
            for l in row["label_str"]:
                self.qrel[q][l.replace(' ', '_')[:100]] = 1
        self.h2i = json.load(open(h2i_fn[task_type], "r"))
        self.i2h = dict(zip(self.h2i.values(), self.h2i.keys()))
        self.trec_evaluator = pytrec_eval.RelevanceEvaluator(self.qrel, {'map', 'recip_rank', 'ndcg_cut_10', 'ndcg_cut_20'})
    
    def parse_one_run(self, res):
        new_res = {}
        for q in res:
            new_res[q] = {}
            for elem in res[q]:
                label_name = self.i2h[elem].replace(' ', '_')[:100]
                new_res[q][label_name] = res[q][elem]
        return new_res
    
    def eval_one_run(self, res, out_fn=None):
        parsed_res = self.parse_one_run(res)
        eval_res = self.trec_evaluator.evaluate(parsed_res)
        if out_fn is None:
            print(self.qrel["Q1"])
        else:
            dump_res = {}
            for q in eval_res:
                dump_res[q] = {}
                dump_res[q]["gt"] = self.qrel[q]
                dump_res[q]["run"] = parsed_res[q]
                dump_res[q]["eval"] = eval_res[q]
            json.dump(dump_res, open(out_fn, 'w'), indent=2)
        avg_map = sum(map(lambda x: x['map'], eval_res.values())) / len(eval_res)
        avg_rpr = sum(map(lambda x: x['recip_rank'], eval_res.values())) / len(eval_res)
        avg_ndcg_10 = sum(map(lambda x: x['ndcg_cut_10'], eval_res.values())) / len(eval_res)
        avg_ndcg_20 = sum(map(lambda x: x['ndcg_cut_20'], eval_res.values())) / len(eval_res)
        # avg_ndcg_20 = sum(map(lambda x: x['ndcg_cut_10'], eval_res.values())) / len(eval_res)
        return avg_map, avg_rpr, avg_ndcg_10, avg_ndcg_20, eval_res
            