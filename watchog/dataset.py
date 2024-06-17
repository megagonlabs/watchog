from argparse import Namespace
import torch
import random
import pandas as pd
import numpy as np
import os
import pickle
import json
import re
import transformers
from torch.utils import data
from torch.nn.utils import rnn
from transformers import AutoTokenizer
from .augment import augment
from typing import List
from functools import reduce
import operator
from .preprocessor import computeTfIdf, tfidfRowSample, preprocess, load_jsonl
from itertools import chain
import copy

# map lm name to huggingface's pre-trained model names
lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased',
         'bert': 'bert-base-uncased'}

def collate_fn(samples):
    data = torch.nn.utils.rnn.pad_sequence(
        [sample["data"] for sample in samples])
    label = torch.cat([sample["label"] for sample in samples])
    batch = {"data": data, "label": label}
    if "idx" in samples[0]:
        # For debug purpose
        batch["idx"] = torch.cat([sample["idx"] for sample in samples])
    return batch



class TableDataset(data.Dataset):
    """Table dataset for unsupervised contrastive learning"""

    def __init__(self,
                 path,
                 augment_op,
                 max_len=256,
                 size=None,
                 lm='bert',
                 single_column=False,
                 sample_meth='wordProb',
                 table_order='column'):
        self.tokenizer = AutoTokenizer.from_pretrained(lm_mp[lm])
        self.max_len = max_len
        self.path = path

        # assuming tables are in csv format
        try:
            self.tables = [fn for fn in os.listdir(path) if '.csv' in fn]
        except:
            self.tables = []

        # only keep the first n tables
        if size is not None:
            self.tables = self.tables[:size]
        self.size = size

        self.table_cache = {}
        self.table_col_md = {}

        # augmentation operators
        self.augment_op = augment_op

        # logging counter
        self.log_cnt = 0

        # sampling method
        self.sample_meth = sample_meth

        # single-column mode
        self.single_column = single_column

        # row or column order for preprocessing
        self.table_order = table_order

        # tokenizer cache
        self.tokenizer_cache = {}

    @staticmethod
    def from_hp(path: str, hp: Namespace):
        """Construct a TableDataset from hyperparameters

        Args:
            path (str): the path to the table directory
            hp (Namespace): the hyperparameters

        Returns:
            TableDataset: the constructed dataset
        """
        return TableDataset(path,
                         augment_op=hp.augment_op,
                         lm=hp.lm,
                         max_len=hp.max_len,
                         size=hp.size,
                         single_column=hp.single_column,
                         sample_meth=hp.sample_meth,
                         table_order=hp.table_order)

    def load_from_wikitables(self, path):
        """Load the tables from the Wikitables dataset
        Args:
            path (str): the path to the table directory
        """
        inf = open(path, 'r')
        lines = inf.readlines(10000000)
        self.tables = []
        self.table_cache = {}
        while lines:
            for l in lines:
                t = json.loads(l)
                t_dict = {}
                columns = t['processed_tableHeaders']
                duplicated = {}
                for i, col in enumerate(columns):
                    if col not in t_dict:
                        t_dict[col] = []
                    else:
                        duplicated[i] = True
                for row in t['tableData']:
                    for i in range(len(row)):
                        if i not in duplicated:
                            t_dict[columns[i]].append(row[i]['text'])
                df = pd.DataFrame(t_dict)
                self.tables.append(t['_id'])
                self.table_cache[len(self.tables)-1] = df
                if self.size is not None:
                    if self.size <= len(self.tables):
                        break
            if self.size is not None:
                if self.size <= len(self.tables):
                    break
            lines = inf.readlines(1000000)
        print('Number of tables: ', len(self.tables))
    
    
    def load_gittables_train_test(self):
        """Since the semtab benchmarks are also from the gittables corpus, 
        we skip tables used in the benchmark for the contrastive learning phase
        """
        base_dirpaths = ['./data/gittables_semtab22/tables/', './data/gittables/tables/']
        self.train_test_table_cache = {}
        train_test_table_hash = {}
        for base_dirpath in base_dirpaths:
            dir_list = os.listdir(base_dirpath)
            for f in dir_list:
                if f.startswith('GitTables') and f.endswith('.csv'):
                    table_file_path = os.path.join(base_dirpath, f)
                    df = pd.read_csv(table_file_path)
                    self.train_test_table_cache[len(self.train_test_table_cache)] = df
                    new_df = df.iloc[:16,]
                    hc = '|$|'.join(['|r|'.join(map(lambda x: str(x).strip(), new_df[col].tolist())) for col in new_df.columns])
                    train_test_table_hash[hc] = len(self.train_test_table_cache)-1
        return train_test_table_hash


    def load_from_gittables(self, path, num_blocks=32, max_cols=32):
        """Load the tables from the Gittables dataset
        Args:
            path (str): the path to the table directory
        """
        dup_cnt = 0
        total_cnt = 0
        num_cols = []
        self.tables = []
        self.table_cache = {}
        self.train_test_table_hash = self.load_gittables_train_test()
        dup_table_cnt = 0
        for bid in range(num_blocks):
            tlist = pickle.load(open(path.format(bid), 'rb'))
            for t in tlist['table_cache']:
                # print(t.keys())
                columns = t['table'].columns
                df = t['table']
                hc_df = t['table'].iloc[:16,]
                hc = '|$|'.join(['|r|'.join(map(lambda x: str(x).strip(), hc_df[col].tolist())) for col in hc_df.columns])
                if hc in self.train_test_table_hash:
                    dup_table_cnt += 1
                    continue
                duplicated = {}
                t_dict = {}
                cols = [re.sub(r'.*([1-9][0-9]{3})', '<year>', col.strip().lower()) for col in columns]
                rename_map = {}
                non_dup_cols = []
                for i, col in enumerate(cols):
                    rename_map[df.columns[i]] = col
                    if col not in t_dict:
                        t_dict[col] = []
                        non_dup_cols.append(i)
                    else:
                        duplicated[i] = True
                        dup_cnt += 1
                    total_cnt += 1
                df = df.rename(columns=rename_map)
                df = df.iloc[:, non_dup_cols]
                df = df.iloc[:, :min(len(non_dup_cols), max_cols)]
                self.tables.append(len(self.table_cache))
                self.table_cache[len(self.tables)-1] = df
                num_cols.append(len(list(df.columns)))
                if self.size is not None:
                    if self.size <= len(self.tables):
                        break
            if self.size is not None:
                if self.size <= len(self.tables):
                    break
                
        print('Number of tables: ', len(self.tables))
        
    def _read_table(self, table_id):
        """Read a table"""
        if table_id in self.table_cache:
            table = self.table_cache[table_id]
        else:
            fn = os.path.join(self.path, self.tables[table_id])
            table = pd.read_csv(fn, lineterminator='\n')
            self.table_cache[table_id] = table

        return table


    def _tokenize(self, table: pd.DataFrame) -> List[int]:
        """Tokenize a DataFrame table

        Args:
            table (DataFrame): the input table

        Returns:
            List of int: list of token ID's with special tokens inserted
            Dictionary: a map from column names to the position of corresponding special tokens
        """
        res = []
        max_tokens = self.max_len * 2 // len(table.columns)
        budget = max(1, self.max_len // len(table.columns) - 1)
        tfidfDict = computeTfIdf(table) if "tfidf" in self.sample_meth else None # from preprocessor.py

        # a map from column names to special token indices
        column_mp = {}

        # column-ordered preprocessing
        if self.table_order == 'column':
            if 'row' in self.sample_meth: 
                table = tfidfRowSample(table, tfidfDict, max_tokens)
            for column in table.columns:
                tokens = preprocess(table[column], tfidfDict, max_tokens, self.sample_meth) # from preprocessor.py
                col_text = self.tokenizer.cls_token + " " + \
                        ' '.join(tokens[:max_tokens]) + " "

                column_mp[column] = len(res)
                res += self.tokenizer.encode(text=col_text,
                                        max_length=budget,
                                        add_special_tokens=False,
                                        truncation=True)
        else:
            # row-ordered preprocessing
            reached_max_len = False
            for rid in range(len(table)):
                row = table.iloc[rid:rid+1]
                for column in table.columns:
                    tokens = preprocess(row[column], tfidfDict, max_tokens, self.sample_meth) # from preprocessor.py
                    if rid == 0:
                        column_mp[column] = len(res)
                        col_text = self.tokenizer.cls_token + " " + \
                                ' '.join(tokens[:max_tokens]) + " "
                    else:
                        col_text = self.tokenizer.pad_token + " " + \
                                ' '.join(tokens[:max_tokens]) + " "

                    tokenized = self.tokenizer.encode(text=col_text,
                                        max_length=budget,
                                        add_special_tokens=False,
                                        truncation=True)

                    if len(tokenized) + len(res) <= self.max_len:
                        res += tokenized
                    else:
                        reached_max_len = True
                        break

                if reached_max_len:
                    break

        self.log_cnt += 1
        
        return res, column_mp


    def __len__(self):
        """Return the size of the dataset."""
        return len(self.tables)

    def __getitem__(self, idx):
        """Return a tokenized item of the dataset.

        Args:
            idx (int): the index of the item

        Returns:
            List of int: token ID's of the first view
            List of int: token ID's of the second view
            List of tuple: column special token [CLS] indices of each view
        """
        table_ori = self._read_table(idx)

        # single-column mode: only keep one random column
        if self.single_column:
            col = random.choice(table_ori.columns)
            table_ori = table_ori[[col]]

        # apply the augmentation operator
        if ',' in self.augment_op:
            op1, op2 = self.augment_op.split(',')
            table_tmp = table_ori
            table_ori = augment(table_tmp, op1)
            table_aug = augment(table_tmp, op2)
        elif self.augment_op in ['dropout', '', 'None', None]:
            table_aug = table_ori 
        else:
            table_aug = augment(table_ori, self.augment_op)

        # convert table into string
        x_ori, mp_ori = self._tokenize(table_ori)
        x_aug, mp_aug = self._tokenize(table_aug)

        # make sure that x_ori and x_aug has the same number of cls tokens
        # x_ori_cnt = sum([int(x == self.tokenizer.cls_token_id) for x in x_ori])
        # x_aug_cnt = sum([int(x == self.tokenizer.cls_token_id) for x in x_aug])
        # assert x_ori_cnt == x_aug_cnt

        # insertsect the two mappings
        cls_indices = []
        for col in mp_ori:
            if col in mp_aug:
                cls_indices.append((mp_ori[col], mp_aug[col]))

        return x_ori, x_aug, cls_indices


    def pad(self, batch):
        """Merge a list of dataset items into a training batch

        Args:
            batch (list of tuple): a list of sequences

        Returns:
            LongTensor: x_ori of shape (batch_size, seq_len)
            LongTensor: x_aug of shape (batch_size, seq_len)
            tuple of List: the cls indices
        """
        x_ori, x_aug, cls_indices = zip(*batch)
        max_len_ori = max([len(x) for x in x_ori])
        max_len_aug = max([len(x) for x in x_aug])
        maxlen = max(max_len_ori, max_len_aug)
        x_ori_new = [xi + [self.tokenizer.pad_token_id]*(maxlen - len(xi)) for xi in x_ori]
        x_aug_new = [xi + [self.tokenizer.pad_token_id]*(maxlen - len(xi)) for xi in x_aug]

        # decompose the column alignment
        cls_ori = []
        cls_aug = []
        for item in cls_indices:
            cls_ori.append([])
            cls_aug.append([])

            for idx1, idx2 in item:
                cls_ori[-1].append(idx1)
                cls_aug[-1].append(idx2)
        
        return torch.LongTensor(x_ori_new), torch.LongTensor(x_aug_new), (cls_ori, cls_aug)

class SupCLTableDataset(TableDataset):
    """Table dataset for contrastive learning with supervision signal"""
    def __init__(self,
                 path,
                 augment_op,
                 max_len=256,
                 size=None,
                 lm='bert',
                 single_column=False,
                 sample_meth='wordProb',
                 table_order='column'):
        super().__init__(path, augment_op, max_len, size, lm, single_column, sample_meth, table_order)
        
    @staticmethod
    def from_hp(path: str, hp: Namespace):
        """Construct a SupCLTableDataset from hyperparameters

        Args:
            path (str): the path to the table directory
            hp (Namespace): the hyperparameters

        Returns:
            TableDataset: the constructed dataset
        """
        return SupCLTableDataset(path,
                         augment_op=hp.augment_op,
                         lm=hp.lm,
                         max_len=hp.max_len,
                         size=hp.size,
                         single_column=hp.single_column,
                         sample_meth=hp.sample_meth,
                         table_order=hp.table_order)

    def load_from_turl(self, train_path, num_classes=255):
        """Load the tables from the TURL semantic type detection dataset
        Args:
            train_path (str): the path to the training data
        """
        turl_table_cache = pickle.load(open(train_path, 'rb'))
        self.tables = []
        self.table_cache = {}
        self.col_label_cache = {}
        self.header_cache = {}

        for idx in turl_table_cache:
            tid = turl_table_cache[idx]['_id']
            self.tables.append(tid)
            self.table_cache[len(self.tables)-1] = turl_table_cache[idx]['raw_table_content']
            self.header_cache[len(self.tables)-1] = turl_table_cache[idx]['headers']
            # self.col_label_cache[len(self.tables)-1] = [turl_table_cache[idx]['label_dict'][j] for j in sorted(sato_table_cache[idx]['label_dict'].keys())]
            labels = []
            for j in turl_table_cache[idx]['label_dict']:
                oh_enc = [0 for _ in range(255)]
                for l in turl_table_cache[idx]['label_dict'][j]:
                    oh_enc[l] = 1
                labels.append(oh_enc)
            self.col_label_cache[len(self.tables)-1] = list(labels)
            if self.size is not None:
                if self.size <= len(self.tables):
                    break
        
        print('Number of tables: ', len(self.tables))
        
    def load_from_wikitables_headerprocessed(self, path, hf_rank_path, header_top_k=-1, entity_col_only=False):
        
        """Load the tables from the Wikitables dataset, following the pre-processing in the TURL paper
        Args:
            path (str): the path to the training data
            hf_rank_path (str): the path to pre-computed ranking of headers by frequency
            header_top_k (int): use the top-k headers; -1 means no filtering
        """
        hf_rank = json.load(open(hf_rank_path, 'r'))
        self.h2i = {}
        for h in hf_rank:
            self.h2i[h] = len(self.h2i)
        inf = open(path, 'rb')
        lines = inf.readlines(1000000)
        self.tables = []
        self.table_cache = {}
        self.table_col_md = {}
        self.col_label_cache = {}
        self.header_cache = {}
        total_cnt = 0
        dup_cnt = 0
        while lines:
            for l in lines:
                t = json.loads(l)
                t_dict = {}
                columns = t['processed_tableHeaders']
                duplicated = {}
                t_dict = {}
                top_k_flag = False
                top_k_flag_per_col = {}
                is_entity_col = {}
                for i, col in enumerate(columns):
                    if entity_col_only:
                        if i not in t['entityColumn']:
                            continue
                    col = re.sub(r'.*([1-9][0-9]{3})', '<year>', col.strip().lower())
                    if header_top_k == -1 or hf_rank[col] < header_top_k:
                        top_k_flag_per_col[col] = True
                        top_k_flag = True
                    if col not in t_dict:
                        t_dict[col] = []
                        is_entity_col[col] = i in t['entityColumn']
                    else:
                        duplicated[i] = True
                        dup_cnt += 1
                    total_cnt += 1
                
                for row in t['tableData']:
                    for i in range(len(row)):
                        if i not in duplicated:
                            new_col = re.sub(r'.*([1-9][0-9]{3})', '<year>', columns[i].strip().lower())
                            t_dict[new_col].append(row[i]['text'])
                if not top_k_flag:
                    continue
                df = pd.DataFrame(t_dict)
                self.tables.append(t['_id'])
                self.table_cache[len(self.tables)-1] = df
                self.table_col_md[len(self.tables)-1] = {
                    'is_top_k_header': top_k_flag_per_col,
                    'is_entity_col': is_entity_col
                }
                cols = list(df.columns)
                self.header_cache[len(self.tables)-1] = cols
                self.col_label_cache[len(self.tables)-1] = {_:self.h2i[_] for _ in cols} # labels are column headers

                if self.size is not None:
                    if self.size <= len(self.tables):
                        break
            if self.size is not None:
                if self.size <= len(self.tables):
                    break
            lines = inf.readlines(1000000)
            
        print('Number of tables: ', len(self.tables))
    
    def load_from_sato(self, base_path, basename, cv, all_path):
        """Load the tables from the SATO semantic type detection dataset
        Args:
            train_path (str): the path to the training data
        """
        
        df_list = []
        for i in range(5):
            if i == cv:
                continue
            filepath = os.path.join(base_path, basename.format(i))
            df_list.append(pd.read_csv(filepath))
        df = pd.concat(df_list, axis=0)
        train_tables = [g[0] for _, g in enumerate(df.groupby(["table_id"]))]
        sato_table_cache = pickle.load(open(all_path, 'rb'))
        self.tables = []
        self.table_cache = {}
        self.col_label_cache = {}
        self.header_cache = {}

        num_tables = len(train_tables)
        valid_index = int(num_tables * 0.8)
        train_tables = train_tables[:valid_index]
        for idx in sato_table_cache:
            tid = sato_table_cache[idx]['_id']
            if tid in train_tables:
                self.tables.append(tid)
                self.table_cache[len(self.tables)-1] = sato_table_cache[idx]['raw_table_content']
                self.header_cache[len(self.tables)-1] = sato_table_cache[idx]['headers']
                self.col_label_cache[len(self.tables)-1] = [sato_table_cache[idx]['label_dict'][j] for j in sorted(sato_table_cache[idx]['label_dict'].keys())]
                if self.size is not None:
                    if self.size <= len(self.tables):
                        break
    
    def load_gittables_train_test(self):
        base_dirpaths = ['./data/gittables_semtab22/tables/', './data/gittables/tables/']
        self.train_test_table_cache = {}
        train_test_table_hash = {}
        for base_dirpath in base_dirpaths:
            dir_list = os.listdir(base_dirpath)
            for f in dir_list:
                if f.startswith('GitTables') and f.endswith('.csv'):
                    table_file_path = os.path.join(base_dirpath, f)
                    df = pd.read_csv(table_file_path)
                    self.train_test_table_cache[len(self.train_test_table_cache)] = df
                    new_df = df.iloc[:16,]
                    hc = '|$|'.join(['|r|'.join(map(lambda x: str(x).strip(), new_df[col].tolist())) for col in new_df.columns])
                    train_test_table_hash[hc] = len(self.train_test_table_cache)-1
        return train_test_table_hash

    def load_from_gittables(self, path, num_blocks=32, hf_rank_path='./data/gittables/gittables.processed.header.freq.json', max_cols=32):
        dup_cnt = 0
        total_cnt = 0
        num_cols = []
        hf_rank = json.load(open(hf_rank_path, 'r'))
        h2i = {}
        for h in hf_rank:
            h2i[h] = len(h2i)
        self.tables = []
        self.table_cache = {}
        self.header_cache = {}
        self.col_label_cache = {}
        
        self.train_test_table_hash = self.load_gittables_train_test()
        dup_table_cnt = 0
        for bid in range(num_blocks):
            tlist = pickle.load(open(path.format(bid), 'rb'))
            for t in tlist['table_cache']:
                # print(t.keys())
                columns = t['table'].columns
                df = t['table']
                hc_df = t['table'].iloc[:16,]
                hc = '|$|'.join(['|r|'.join(map(lambda x: str(x).strip(), hc_df[col].tolist())) for col in hc_df.columns])
                if hc in self.train_test_table_hash:
                    dup_table_cnt += 1
                    continue
                duplicated = {}
                t_dict = {}
                cols = [re.sub(r'.*([1-9][0-9]{3})', '<year>', col.strip().lower()) for col in columns]
                rename_map = {}
                non_dup_cols = []
                for i, col in enumerate(cols):
                    rename_map[df.columns[i]] = col
                    if col not in t_dict:
                        t_dict[col] = []
                        non_dup_cols.append(i)
                    else:
                        duplicated[i] = True
                        dup_cnt += 1
                    total_cnt += 1
                df = df.rename(columns=rename_map)
                df = df.iloc[:, non_dup_cols]
                df = df.iloc[:, :min(len(non_dup_cols), max_cols)]
                self.tables.append(len(self.table_cache))
                self.table_cache[len(self.tables)-1] = df
                self.header_cache[len(self.tables)-1] = list(df.columns)
                self.col_label_cache[len(self.tables)-1] = {_:h2i[_] for _ in list(df.columns)}
                num_cols.append(len(list(df.columns)))
                if self.size is not None:
                    if self.size <= len(self.tables):
                        break
            if self.size is not None:
                if self.size <= len(self.tables):
                    break
                
        print('Number of tables: ', len(self.tables))
        print(dup_cnt, total_cnt, dup_table_cnt)
        # num_cols_df = pd.DataFrame(num_cols, columns=['num_cols'])
        # print(num_cols_df.describe(percentiles=[0.1,0.25,0.5,0.75,0.9,0.95,0.99]))
  
    def _read_table(self, table_id):
        """Read a table"""
        # print(table_id)
        if table_id in self.table_cache:
            table = self.table_cache[table_id]
            col_label = self.col_label_cache[table_id]
            headers = self.header_cache[table_id]
        else:
            fn = os.path.join(self.path, self.tables[table_id])
            table = pd.read_csv(fn, lineterminator='\n')
            self.table_cache[table_id] = table

        return table, col_label, headers

    def __getitem__(self, idx):
        """Return a tokenized item of the dataset.

        Args:
            idx (int): the index of the item

        Returns:
            List of int: token ID's of the first view
            List of int: token ID's of the second view
            List of tuple: column special token [CLS] indices of each view
            List of lists: labels (i.e., column headers) of each column of the first view
            List of lists: labels (i.e., column headers) of each column of the second view
        """
        table_ori, col_label, headers = self._read_table(idx)

        # single-column mode: only keep one random column
        if self.single_column:
            col = random.choice(table_ori.columns)
            table_ori = table_ori[[col]]

        # apply the augmentation operator
        if ',' in self.augment_op:
            op1, op2 = self.augment_op.split(',')
            table_tmp = table_ori
            table_ori = augment(table_tmp, op1)
            table_aug = augment(table_tmp, op2)
        elif self.augment_op in ['dropout', '', 'None', None]:
            table_aug = table_ori
        else:
            table_aug = augment(table_ori, self.augment_op)

        # convert table into string
        x_ori, mp_ori = self._tokenize(table_ori)
        x_aug, mp_aug = self._tokenize(table_aug)

        # insertsect the two mappings
        cls_indices = []
        ori_col_label = []
        for col in mp_ori:
            if col in mp_aug:
                cls_indices.append((mp_ori[col], mp_aug[col]))
                ori_col_label.append(col_label[col])
        aug_col_label = []
        for col in mp_aug:
            aug_col_label.append(col_label[col])
        return x_ori, x_aug, cls_indices, ori_col_label, aug_col_label
    
    def pad(self, batch):
        """Merge a list of dataset items into a training batch

        Args:
            batch (list of tuple): a list of sequences

        Returns:
            LongTensor: x_ori of shape (batch_size, seq_len)
            LongTensor: x_aug of shape (batch_size, seq_len)
            tuple of List: the cls indices
            LongTensor: y_ori of shape (# of columns)
            LongTensor: y_aug of shape (# of columns)
        """
        x_ori, x_aug, cls_indices, ori_col_label, aug_col_label = zip(*batch)
        max_len_ori = max([len(x) for x in x_ori])
        max_len_aug = max([len(x) for x in x_aug])
        maxlen = max(max_len_ori, max_len_aug)
        x_ori_new = [xi + [self.tokenizer.pad_token_id]*(maxlen - len(xi)) for xi in x_ori]
        x_aug_new = [xi + [self.tokenizer.pad_token_id]*(maxlen - len(xi)) for xi in x_aug]

        # decompose the column alignment
        cls_ori = []
        cls_aug = []
        for item in cls_indices:
            cls_ori.append([])
            cls_aug.append([])

            for idx1, idx2 in item:
                cls_ori[-1].append(idx1)
                cls_aug[-1].append(idx2)
        ori_col_label = list(chain.from_iterable(ori_col_label))
        aug_col_label = list(chain.from_iterable(aug_col_label))
            
        return torch.LongTensor(x_ori_new), torch.LongTensor(x_aug_new), (cls_ori, cls_aug), torch.LongTensor(ori_col_label), torch.LongTensor(aug_col_label)


class SatoCVTablewiseDataset(data.Dataset):
    """Table dataset for finetuning and evaluating semantic type detection on the SATO dataset"""

    def __init__(
            self,
            cv: int,
            split: str,  # train or test
            tokenizer: AutoTokenizer,
            max_length: int = 256,
            multicol_only: bool = False,
            train_ratio: float = 1.0,
            device: torch.device = None,
            base_dirpath: str = "./data/doduo",
            small_tag: str = ""):
        if device is None:
            device = torch.device('cpu')

        assert split in ["train", "valid",
                         "test"], "split must be train or test"
        if small_tag != '':
            if split in ["train", "valid"]:
                base_dirpath = "./data/doduo_small"
                if split not in small_tag:
                    small_tag += '_' + split
        if multicol_only:
            basename = "msato_cv_{}.csv"
        else:
            basename = "sato_cv_{}.csv"

        if split in ["train", "valid"]:
            if small_tag == "":
                df_list = []
                for i in range(5):
                    if i == cv:
                        continue
                    filepath = os.path.join(base_dirpath, basename.format(i))
                    df_list.append(pd.read_csv(filepath))
                df = pd.concat(df_list, axis=0)
            else:
                filepath = os.path.join(base_dirpath, basename.format(str(cv) + '_' + small_tag))
                df = pd.read_csv(filepath)
        else:
            # test
            filepath = os.path.join(base_dirpath, basename.format(cv))
            df = pd.read_csv(filepath)

        num_tables = len(df.groupby("table_id"))
        valid_index = int(num_tables * 0.8)
        num_train = int(train_ratio * num_tables * 0.8)
        
        data_list = []
        for i, (index, group_df) in enumerate(df.groupby("table_id")):
            if small_tag == "":
                if (split == "train") and ((i >= num_train) or (i >= valid_index)):
                    break
                if split == "valid" and i < valid_index:
                    continue
            if max_length <= 128:
                cur_maxlen = min(max_length, 512 // len(list(group_df["class_id"].values)) - 1)
            else:
                cur_maxlen = max(1, max_length // len(list(group_df["class_id"].values)) - 1)
            token_ids_list = group_df["data"].apply(lambda x: tokenizer.encode(
                tokenizer.cls_token + " " + x, add_special_tokens=False, max_length=cur_maxlen, truncation=True)).tolist(
                )
            token_ids = torch.LongTensor(reduce(operator.add,
                                                token_ids_list)).to(device)
            cls_index_list = [0] + np.cumsum(
                np.array([len(x) for x in token_ids_list])).tolist()[:-1]
            for cls_index in cls_index_list:
                assert token_ids[
                    cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
            cls_indexes = torch.LongTensor(cls_index_list).to(device)
            class_ids = torch.LongTensor(
                group_df["class_id"].values).to(device)
            data_list.append(
                [index,
                 len(group_df), token_ids, class_ids, cls_indexes])
        print(split, len(data_list))
        self.table_df = pd.DataFrame(data_list,
                                     columns=[
                                         "table_id", "num_col", "data_tensor",
                                         "label_tensor", "cls_indexes"
                                     ])
        """
        # NOTE: msato contains a small portion of single-col tables. keep it to be consistent.  
        if multicol_only:
            # Check
            num_all_tables = len(self.table_df)
            self.table_df = self.table_df[self.table_df["num_col"] > 1]
            assert len(self.table_df) == num_all_tables
        """

    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"]
        }
        #"idx": torch.LongTensor([idx])}
        #"cls_indexes": self.table_df.iloc[idx]["cls_indexes"]}


class SatoCVTablewiseDatasetWithDA(data.Dataset):
    """Table dataset for finetuning semantic type detection on the SATO dataset under semi-supervised setting"""
    def __init__(
            self,
            cv: int,
            split: str,  # train or test
            tokenizer: AutoTokenizer,
            max_length: int = 256,
            multicol_only: bool = False,
            train_ratio: float = 1.0,
            device: torch.device = None,
            base_dirpath: str = "/efs/task_datasets/doduo/",
            all_tables_filename: str = "sato{}.processed.pkl",
            small_tag: str = "",
            augment_op: str = "",
            num_aug: int = 1):
        if device is None:
            device = torch.device('cpu')

        assert split in ["train", "unlabeled"], "split must be train or unlabeled"
        if small_tag != '':
            base_dirpath_small = "./data/doduo_small"
        if multicol_only:
            basename = "msato_cv_{}.csv"
        else:
            basename = "sato_cv_{}.csv"
        
        self.augment_op = augment_op
        self.num_aug = num_aug
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        df_train = pd.read_csv(os.path.join(base_dirpath_small, basename.format(str(cv) + '_' + small_tag + '_train')))
        if split == 'train':
            raw_table_basename = "sato_cv_{}_train.processed.pkl" 
            self.table_dict = pickle.load(open(os.path.join(base_dirpath_small, raw_table_basename.format(str(cv) + '_' + small_tag)), 'rb'))
            self.tables = list(self.table_dict.values())
        elif split == 'unlabeled':
            df_valid = pd.read_csv(os.path.join(base_dirpath_small, basename.format(str(cv) + '_' + small_tag + '_valid')))
            labeled_table_id_dict = {}
            for i, tid in enumerate(df_train['table_id']):
                labeled_table_id_dict[tid] = True
            for i, tid in enumerate(df_valid['table_id']):
                labeled_table_id_dict[tid] = True    
            raw_tables = pickle.load(open(base_dirpath + all_tables_filename.format(cv), 'rb'))
            self.table_dict = {k: v for k, v in raw_tables.items() if k not in labeled_table_id_dict}
            self.tables = list(self.table_dict.values())
        print(len(self.table_dict))
       
    def __len__(self):
        return len(self.table_dict)
    
    def tokenize_one_table(self, table):
        num_cols = len(table['table_content'])
        if self.max_length <= 128:
            cur_maxlen = min(self.max_length, 512 // num_cols - 1)
        else:
            cur_maxlen = max(1, self.max_length // num_cols - 1)
        df = table['raw_table_content']
        token_ids_list = [self.tokenizer.encode(self.tokenizer.cls_token + " " + " ".join([str(_) for _ in df[x].dropna().tolist()]), 
                                                add_special_tokens=False, max_length=cur_maxlen, truncation=True) for x in df.columns]
        
        token_ids = torch.LongTensor(reduce(operator.add,
                                            token_ids_list)).to(self.device)
        cls_index_list = [0] + np.cumsum(
            np.array([len(x) for x in token_ids_list])).tolist()[:-1]
        for cls_index in cls_index_list:
            assert token_ids[
                cls_index] == self.tokenizer.cls_token_id, "cls_indexes validation"
        cls_indexes = torch.LongTensor(cls_index_list).to(self.device)
        labels = [table['label_dict'][_] for _ in range(num_cols)]
        class_ids = torch.LongTensor(labels).to(self.device)
        return token_ids, cls_indexes, class_ids

    def __getitem__(self, idx):
        table_ori = self.tables[idx]

        # apply the augmentation operator
        op_list = self.augment_op.split(',')
        if len(op_list) == 1 and self.num_aug > 1:
            op_list = [self.augment_op for _ in range(self.num_aug)]
            
        table_aug = []
        for op in op_list:
            table_tmp = copy.deepcopy(table_ori)
            if op in ['dropout', '', 'None', None]:
                table_aug.append(table_tmp)
            else:
                table_tmp['raw_table_content'] = augment(table_tmp['raw_table_content'], op)
                table_aug.append(table_tmp)
        # convert table into string        
        x_ori, cls_indexes_ori, class_ids_ori = self.tokenize_one_table(table_ori)
        x_aug = []
        cls_indexes_aug = []
        class_ids_aug = []
        for table in table_aug:
            x, cls_indexes, class_ids = self.tokenize_one_table(table)
            x_aug.append(x)
            cls_indexes_aug.append(cls_indexes)
            class_ids_aug.append(class_ids)
        return {"ori": x_ori, "cls_indexes": cls_indexes_ori, "label_ori": class_ids_ori, "aug": x_aug, "cls_indexes_aug": cls_indexes_aug, "label_aug": class_ids_aug}

class TURLColTypeTablewiseDataset(data.Dataset):
    """Table dataset for finetuning and evaluating semantic type detection on the TURL dataset"""

    def __init__(self,
                 filepath: str,
                 split: str,
                 tokenizer: AutoTokenizer,
                 max_length: int = 32,
                 max_colnum: int = 10,
                 multicol_only: bool = False,
                 train_ratio: float = 1.0,
                 device: torch.device = None,
                 small_tag: str = ""):
        if device is None:
            device = torch.device('cpu')

        with open(filepath, "rb") as fin:
            df_dict = pickle.load(fin)

        assert split in df_dict
        self.df = df_dict[split]
        num_tables = len(self.df.groupby("table_id"))
        num_train = int(train_ratio * num_tables)
        data_list = []
        for i, (index, group_df) in enumerate(self.df.groupby("table_id")):
            if i >= num_train:
                break
    
            if split == "train" and len(group_df) > max_colnum:
                continue
            label_ids = group_df["label_ids"].tolist()
            if max_length <= 128:
                cur_maxlen = min(max_length, 512 // len(label_ids) -1)
            else:
                cur_maxlen = max(1, max_length // len(label_ids) - 1)
            token_ids_list = group_df["data"].apply(lambda x: tokenizer.encode(
                tokenizer.cls_token + " " + x, add_special_tokens=False, max_length=cur_maxlen, truncation=True)).tolist(
                )
            token_ids = torch.LongTensor(reduce(operator.add,
                                                token_ids_list)).to(device)
            cls_index_list = [0] + np.cumsum(
                np.array([len(x) for x in token_ids_list])).tolist()[:-1]
            for cls_index in cls_index_list:
                assert token_ids[
                    cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
            cls_indexes = torch.LongTensor(cls_index_list).to(device)
            class_ids = torch.LongTensor(label_ids).to(device)
            data_list.append(
                [index,
                 len(group_df), token_ids, class_ids, cls_indexes])

        self.table_df = pd.DataFrame(data_list,
                                     columns=[
                                         "table_id", "num_col", "data_tensor",
                                         "label_tensor", "cls_indexes"
                                     ])
        print('# Tables:', len(data_list))
        if multicol_only:
            self.table_df = self.table_df[self.table_df["num_col"] > 1]

    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"]
        }


class TURLColTypeTablewiseDatasetWithDA(data.Dataset):
    """Table dataset for finetuning semantic type detection on the TURL dataset under semi-supervised setting"""

    def __init__(
            self,
            filepath: str,
            split: str,  # train or test
            tokenizer: AutoTokenizer,
            max_length: int = 256,
            multicol_only: bool = False,
            train_ratio: float = 1.0,
            device: torch.device = None,
            all_tables_filename: str = "table_col_type_serialized.alltrain.processed.pkl",
            base_dirpath: str = "/efs/task_datasets/TURL/",
            augment_op: str = "",
            num_aug: int = 1,
            num_classes: int = 255):
        
        if device is None:
            device = torch.device('cpu')

        assert split in ["train", "unlabeled"], "split must be train or unlabeled"
        with open(filepath, "rb") as fin:
            df_dict = pickle.load(fin)

      
        self.augment_op = augment_op
        self.num_aug = num_aug
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.num_classes = num_classes
        self.split = split
        if split == 'train':
            with open(filepath.replace('.pkl', '.processed.pkl'), "rb") as fin_raw:
                self.table_dict = pickle.load(fin_raw)
                self.tables = list(self.table_dict[split].values())
              
        elif split == 'unlabeled':
            labeled_table_id_dict = {}
            for s in ['train', 'dev']:
                for i, tid in enumerate(df_dict[s]['table_id']):
                    labeled_table_id_dict[tid] = True
            raw_tables = pickle.load(open(base_dirpath + all_tables_filename, 'rb'))
            self.table_dict = {k: v for k, v in raw_tables.items() if k not in labeled_table_id_dict}
            self.tables = list(self.table_dict.values())
        self.tables = self.tables
        print(len(self.tables))
        
        
        
    def __len__(self):
        return len(self.tables)
    
    def tokenize_one_table(self, table):
        num_cols = len(table['table_content'])
        if self.max_length <= 128:
            cur_maxlen = min(self.max_length, 512 // num_cols - 1)
        else:
            cur_maxlen = max(1, self.max_length // num_cols - 1)

        df = table['raw_table_content']
        token_ids_list = [self.tokenizer.encode(self.tokenizer.cls_token + " " + " ".join([str(_) for _ in df[x].dropna().tolist()]), 
                                                add_special_tokens=False, max_length=cur_maxlen, truncation=True) for x in df.columns]
        
        token_ids = torch.LongTensor(reduce(operator.add,
                                            token_ids_list)).to(self.device)
        cls_index_list = [0] + np.cumsum(
            np.array([len(x) for x in token_ids_list])).tolist()[:-1]
        for cls_index in cls_index_list:
            assert token_ids[
                cls_index] == self.tokenizer.cls_token_id, "cls_indexes validation"
        cls_indexes = torch.LongTensor(cls_index_list).to(self.device)
        labels = [[1 if __ in table['label_dict'][_] else 0 for __ in range(self.num_classes)] for _ in range(num_cols)]
        class_ids = torch.LongTensor(labels).to(self.device)
        return token_ids, cls_indexes, class_ids

    def __getitem__(self, idx):
        
        table_ori = self.tables[idx]
        # apply the augmentation operator
        op_list = self.augment_op.split(',')
        if len(op_list) == 1 and self.num_aug > 1:
            op_list = [self.augment_op for _ in range(self.num_aug)]
        table_aug = []
        for op in op_list:
            table_tmp = copy.deepcopy(table_ori)
            if op in ['dropout', '', 'None', None]:
                table_aug.append(table_tmp)
            else:
                table_tmp['raw_table_content'] = augment(table_tmp['raw_table_content'], op)
                table_aug.append(table_tmp)
        # convert table into string        
        x_ori, cls_indexes_ori, class_ids_ori = self.tokenize_one_table(table_ori)
        x_aug = []
        cls_indexes_aug = []
        class_ids_aug = []
        for table in table_aug:
            x, cls_indexes, class_ids = self.tokenize_one_table(table)
            x_aug.append(x)
            cls_indexes_aug.append(cls_indexes)
            class_ids_aug.append(class_ids)         
        
        return {"ori": x_ori, "cls_indexes": cls_indexes_ori, "label_ori": class_ids_ori, "aug": x_aug, "cls_indexes_aug": cls_indexes_aug, "label_aug": class_ids_aug}


class TURLRelExtTablewiseDataset(data.Dataset):
    """Table dataset for finetuning and evaluating relationship extraction on the TURL dataset"""

    def __init__(self,
                 filepath: str,
                 split: str,
                 tokenizer: AutoTokenizer,
                 max_length: int = 32,
                 max_colnum: int = 10,
                 multicol_only: bool = False,
                 train_ratio: float = 1.0,
                 device: torch.device = None):
        if device is None:
            device = torch.device('cpu')

        with open(filepath, "rb") as fin:
            df_dict = pickle.load(fin)

        assert split in df_dict
        self.df = df_dict[split]

        num_tables = len(self.df.groupby("table_id"))
        num_train = int(train_ratio * num_tables)
        # num_train = 1000

        data_list = []
        for i, (index, group_df) in enumerate(self.df.groupby("table_id")):
            if i >= num_train:
                break

            # It's probably already sorted but just in case.
            group_df = group_df.sort_values("column_id")

            if split == "train" and len(group_df) > max_colnum:
                continue
            label_ids = group_df["label_ids"].tolist()

            if max_length <= 128:
                cur_maxlen = min(max_length, 512 // len(label_ids) - len(label_ids))
            else:
                cur_maxlen = max(1, max_length // len(label_ids) - len(label_ids))
            token_ids_list = group_df["data"].apply(lambda x: tokenizer.encode(
                tokenizer.cls_token + " " + x, add_special_tokens=False, max_length=cur_maxlen, truncation=True)).tolist(
                )
            token_ids = torch.LongTensor(reduce(operator.add,
                                                token_ids_list)).to(device)
            cls_index_list = [0] + np.cumsum(
                np.array([len(x) for x in token_ids_list])).tolist()[:-1]
            for cls_index in cls_index_list:
                assert token_ids[
                    cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
            cls_indexes = torch.LongTensor(cls_index_list).to(device)
            class_ids = torch.LongTensor(label_ids).to(device)
            data_list.append(
                [index,
                 len(group_df), token_ids, class_ids, cls_indexes])

        self.table_df = pd.DataFrame(data_list,
                                     columns=[
                                         "table_id", "num_col", "data_tensor",
                                         "label_tensor", "cls_indexes"
                                     ])

        if multicol_only:
            self.table_df = self.table_df[self.table_df["num_col"] > 1]

    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"]
        }


class TURLRelExtTablewiseDatasetWithDA(data.Dataset):
    """Table dataset for finetuning relationship extraction on the TURL dataset under semi-supervised setting"""

    def __init__(
            self,
            filepath: str,
            split: str,  # train or test
            tokenizer: AutoTokenizer,
            max_length: int = 256,
            multicol_only: bool = False,
            train_ratio: float = 1.0,
            device: torch.device = None,
            all_tables_filename: str = "table_rel_extraction_serialized.alltrain.processed.pkl",
            base_dirpath: str = "/efs/task_datasets/TURL/",
            augment_op: str = "",
            num_aug: int = 1,
            num_classes: int = 121):
        
        if device is None:
            device = torch.device('cpu')

        assert split in ["train", "unlabeled"], "split must be train or unlabeled"
        with open(filepath, "rb") as fin:
            df_dict = pickle.load(fin)
      
        self.augment_op = augment_op
        self.num_aug = num_aug
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.num_classes = num_classes
        self.split = split
        if split == 'train':
            with open(filepath.replace('.pkl', '.processed.pkl'), "rb") as fin_raw:
                self.table_dict = pickle.load(fin_raw)
                self.tables = list(self.table_dict[split].values())

        elif split == 'unlabeled':
            labeled_table_id_dict = {}
            for s in ['train', 'dev']:
                for i, tid in enumerate(df_dict[s]['table_id']):
                    labeled_table_id_dict[tid] = True
            raw_tables = pickle.load(open(base_dirpath + all_tables_filename, 'rb'))
            self.table_dict = {k: v for k, v in raw_tables.items() if k not in labeled_table_id_dict}
            self.tables = list(self.table_dict.values())
            
        print(len(self.tables))    
        
    def __len__(self):
        return len(self.tables)
    
    def tokenize_one_table(self, table):
        num_cols = len(table['table_content'])
        if self.max_length <= 128:
            cur_maxlen = min(self.max_length, 512 // num_cols - 1)
        else:
            cur_maxlen = max(1, self.max_length // num_cols - 1)
        df = table['raw_table_content']
        token_ids_list = [self.tokenizer.encode(self.tokenizer.cls_token + " " + " ".join([str(_) for _ in df[x].dropna().tolist()]), 
                                                add_special_tokens=False, max_length=cur_maxlen, truncation=True) for x in df.columns]
        
        token_ids = torch.LongTensor(reduce(operator.add,
                                            token_ids_list)).to(self.device)
        cls_index_list = [0] + np.cumsum(
            np.array([len(x) for x in token_ids_list])).tolist()[:-1]
        for cls_index in cls_index_list:
            assert token_ids[ 
                cls_index] == self.tokenizer.cls_token_id, "cls_indexes validation"
        cls_indexes = torch.LongTensor(cls_index_list).to(self.device)
        if self.split == 'unlabeled':
            labels = [[1 if __ in table['label_dict'][_] else 0 for __ in range(self.num_classes)] for _ in range(1, num_cols)]
        else:
            labels = [[1 if __ in table['label_dict'][_] else 0 for __ in range(self.num_classes)] for _ in range(num_cols)]

        class_ids = torch.LongTensor(labels).to(self.device)
        return token_ids, cls_indexes, class_ids

    def __getitem__(self, idx):
        
        table_ori = self.tables[idx]
        # apply the augmentation operator
        op_list = self.augment_op.split(',')
        if len(op_list) == 1 and self.num_aug > 1:
            op_list = [self.augment_op for _ in range(self.num_aug)]
            
        table_aug = []
        for op in op_list:
            table_tmp = copy.deepcopy(table_ori)
            if op in ['dropout', '', 'None', None]:
                table_aug.append(table_tmp)
            else:
                table_tmp['raw_table_content'] = augment(table_tmp['raw_table_content'], op)
                table_aug.append(table_tmp)
        # convert table into string        
        x_ori, cls_indexes_ori, class_ids_ori = self.tokenize_one_table(table_ori)
        x_aug = []
        cls_indexes_aug = []
        class_ids_aug = []
        for table in table_aug:
            x, cls_indexes, class_ids = self.tokenize_one_table(table)
            x_aug.append(x)
            cls_indexes_aug.append(cls_indexes)
            class_ids_aug.append(class_ids)
            
        return {"ori": x_ori, "cls_indexes": cls_indexes_ori, "label_ori": class_ids_ori, "aug": x_aug, "cls_indexes_aug": cls_indexes_aug, "label_aug": class_ids_aug}

class ColPoplTablewiseDataset(data.Dataset):

    def __init__(self,
                 filepath: str,
                 split: str,
                 tokenizer: AutoTokenizer,
                 max_length: int = 32,
                 max_colnum: int = 10,
                 multicol_only: bool = False,
                 train_ratio: float = 1.0,
                 use_content: bool = True,
                 use_metadata: bool = False,
                 device: torch.device = None):
        if device is None:
            device = torch.device('cpu')

    
        data = load_jsonl(filepath, n_rows=1000000)

        self.df = pd.DataFrame.from_dict(data)
        num_tables = len(self.df)
        num_train = int(train_ratio * num_tables)
        data_list = []
        for i, line in self.df.iterrows():
            if i >= num_train:
                break
            table = line['table_data']
            col_num = len(table[0])

            if split == "train" and col_num > max_colnum:
                continue
            label_str = line["label"]
            label_ids = line["label_idx"]
            for j in range(len(label_ids)):
                if label_ids[j] >= 127656:
                    label_ids[j] = 0

            if max_length <= 128:
                if col_num > 0:
                    cur_maxlen = min(max_length, 512 // col_num - col_num)
                else:
                    cur_maxlen = max_length
            else:
                cur_maxlen = max(1, max_length // col_num - col_num)
            if use_metadata and use_content:
                columns = [' '.join(line['orig_header'][j].split() + [table[_][j] for _ in range(len(table))]) for j in range(col_num)]
            elif use_metadata:
                columns = [' '.join(line['orig_header'][j].split()) for j in range(col_num)]
            elif use_content:
                columns = [' '.join([table[_][j] for _ in range(len(table))]) for j in range(col_num)]
            if use_metadata:
                pgTitle = line['pgTitle']
                secTitle = line['sectionTitle']
                caption = line['caption']
                metadata = pgTitle.split() + secTitle.split()
                if caption != secTitle:
                    metadata += caption.split()
                columns= [' '.join(metadata)] + columns
            token_ids_list = list(map(lambda x: tokenizer.encode(
                tokenizer.cls_token + " " + x, add_special_tokens=False, max_length=cur_maxlen, truncation=True), columns))
            token_ids = torch.LongTensor(reduce(operator.add,
                                                token_ids_list)).to(device)
            cls_index_list = [0] + np.cumsum(
                np.array([len(x) for x in token_ids_list])).tolist()[:-1]
            for cls_index in cls_index_list:
                assert token_ids[
                    cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
            cls_indexes = torch.LongTensor(cls_index_list).to(device)
            class_ids = torch.LongTensor(label_ids).to(device)
            data_list.append(
                [i, col_num, token_ids, class_ids, label_str, cls_indexes])
       
        self.table_df = pd.DataFrame(data_list,
                                     columns=[
                                         "idx", "num_col", "data_tensor",
                                         "label_tensor", "label_str", "cls_indexes"
                                     ])
        if multicol_only:
            self.table_df = self.table_df[self.table_df["num_col"] > 1]

    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "idx": self.table_df.iloc[idx]["idx"],
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"],
            "cls_indexes": self.table_df.iloc[idx]["cls_indexes"]
        }


class SemtableCVTablewiseDataset(data.Dataset):

    def __init__(
            self,
            cv: int,
            split: str,  # train or test
            tokenizer: AutoTokenizer,
            max_length: int = 256,
            multicol_only: bool = False,
            device: torch.device = None,
            base_dirpath: str = "./data/semtable2019",
            small_tag: str = ""):
        if device is None:
            device = torch.device('cpu')

        assert split in ["train", "valid",
                         "test"], "split must be train or test"
        if split in ["train", "valid"]:
            if multicol_only:
                basename = "mreca_all_{}_fold_{}.csv"
            else:
                basename = "reca_all_{}_fold_{}.csv"
        else:
            if multicol_only:
                basename = "mreca_all_{}.csv"
            else:
                basename = "reca_all_{}.csv"

        filepath = os.path.join(base_dirpath, basename.format(split, cv))
    
        df = pd.read_csv(filepath)
        df = df[df["class_id"] > -1]
        
        num_tables = len(df.groupby("table_id"))
        
        data_list = []
        for i, (index, group_df) in enumerate(df.groupby("table_id")):
            # if len(data_list) > 500:
            #     break

            if max_length <= 128:
                cur_maxlen = min(max_length, 512 // len(list(group_df["class_id"].values)) - 1)
            else:
                cur_maxlen = max(1, max_length // len(list(group_df["class_id"].values)) - 1)
                
            token_ids_list = group_df["data"].apply(lambda x: tokenizer.encode(
                tokenizer.cls_token + " " + x, add_special_tokens=False, max_length=cur_maxlen, truncation=True)).tolist(
                )
            token_ids = torch.LongTensor(reduce(operator.add,
                                                token_ids_list)).to(device)
            cls_index_list = [0] + np.cumsum(
                np.array([len(x) for x in token_ids_list])).tolist()[:-1]
            for cls_index in cls_index_list:
                assert token_ids[
                    cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
            cls_indexes = torch.LongTensor(cls_index_list).to(device)
            class_ids = torch.LongTensor(
                group_df["class_id"].values).to(device)
            data_list.append(
                [index,
                 len(group_df), token_ids, class_ids, cls_indexes])
        print(split, len(data_list))
        self.table_df = pd.DataFrame(data_list,
                                     columns=[
                                         "table_id", "num_col", "data_tensor",
                                         "label_tensor", "cls_indexes"
                                     ])
        """
        # NOTE: msato contains a small portion of single-col tables. keep it to be consistent.  
        if multicol_only:
            # Check
            num_all_tables = len(self.table_df)
            self.table_df = self.table_df[self.table_df["num_col"] > 1]
            assert len(self.table_df) == num_all_tables
        """

    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"]
        }
        #"idx": torch.LongTensor([idx])}
        #"cls_indexes": self.table_df.iloc[idx]["cls_indexes"]}
        
class SemtableCVColwiseDataset(data.Dataset):

    def __init__(
            self,
            cv: int,
            split: str,  # train or test
            tokenizer: transformers.PreTrainedTokenizer,
            max_length: int = 128,
            multicol_only: bool = False,
            device: torch.device = None,
            base_dirpath: str = "./data/semtable2019",
            small_tag: str = ""):
        if device is None:
            device = torch.device('cpu')

        assert split in ["train", "valid",
                         "test"], "split must be train or test"
         
        if split in ["train", "valid"]:
            if multicol_only:
                basename = "mreca_all_{}_fold_{}.csv"
            else:
                basename = "reca_all_{}_fold_{}.csv"
        else:
            if multicol_only:
                basename = "mreca_all_{}.csv"
            else:
                basename = "reca_all_{}.csv"

        filepath = os.path.join(base_dirpath, basename.format(split, cv))
    
        df = pd.read_csv(filepath)
        df = df[df["class_id"] > -1]
        
        num_tables = len(df.groupby("table_id"))
        
        row_list = []
        for i, (index, group_df) in enumerate(df.groupby("table_id")):
            # if len(data_list) > 500:
            #     break
            for _, row in group_df.iterrows():
                row_list.append(row)


        self.df = pd.DataFrame(row_list)

        # Convert into torch.Tensor
        self.df["data_tensor"] = self.df["data"].apply(
            lambda x: torch.LongTensor(
                tokenizer.encode(x,
                                 add_special_tokens=True,
                                 max_length=max_length + 2)).to(device))
        self.df["label_tensor"] = self.df["class_id"].apply(
            lambda x: torch.LongTensor([x]).to(device)
        )  # Can we reduce the size?

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "data": self.df.iloc[idx]["data_tensor"],
            "label": self.df.iloc[idx]["label_tensor"]
        }
        

class GittablesTablewiseDataset(data.Dataset):

    def __init__(
            self,
            split: str,
            src: str,  # train or test
            tokenizer: AutoTokenizer,
            max_length: int = 256,
            gt_only: bool = False,
            device: torch.device = None,
            base_dirpath: str = "./data/gittables",
            base_tag: str = '',
            small_tag: str = ""):
        if device is None:
            device = torch.device('cpu')

    
        if gt_only:
            basename = "{}{}_{}_all_gt.csv"
        else:
            basename = "{}{}_{}_all.csv"

        filepath = os.path.join(base_dirpath, basename.format(src, base_tag, split))
    
        df = pd.read_csv(filepath)
        if gt_only:
            df = df[df["class_id"] > -1]
        
        num_tables = len(df.groupby("table_id"))
        
        data_list = []
        
        df['class_id'] = df['class_id'].astype(int)
        df.drop(df[(df['data'].isna()) & (df['class_id'] == -1)].index, inplace=True)
        df['col_idx'] = df['col_idx'].astype(int)
        df['data'] = df['data'].astype(str)
        # df.drop(df[(df['data'] == '') & (df['class_id'] == -1)].index, inplace=True)
        total_num_cols = 0
        for i, (index, group_df) in enumerate(df.groupby("table_id")):
            # if len(data_list) > 10:
            #     break
            labeled_columns = group_df[group_df['class_id'] > -1]
            unlabeled_columns = group_df[group_df['class_id'] == -1]
            # group_df = pd.concat([group_df[group_df['class_id'] > -1], unlabeled_columns.sample(min(10-len(labeled_columns), len(unlabeled_columns)))])
            # group_df = pd.concat([group_df[group_df['class_id'] > -1], unlabeled_columns[0:min(max(10-len(labeled_columns), 0), len(unlabeled_columns))]])
            group_df = pd.concat([group_df[group_df['class_id'] > -1], unlabeled_columns[0:min(max(8-len(labeled_columns), 0), len(unlabeled_columns))]])
            group_df.sort_values(by=['col_idx'], inplace=True)

            if max_length <= 128:
                cur_maxlen = min(max_length, 512 // len(list(group_df["class_id"].values)) - 1)
            else:
                cur_maxlen = max(1, max_length // len(list(group_df["class_id"].values)) - 1)
                
            token_ids_list = group_df["data"].apply(lambda x: tokenizer.encode(
                tokenizer.cls_token + " " + x, add_special_tokens=False, max_length=cur_maxlen, truncation=True)).tolist(
                )
            token_ids = torch.LongTensor(reduce(operator.add,
                                                token_ids_list)).to(device)
            cls_index_list = [0] + np.cumsum(
                np.array([len(x) for x in token_ids_list])).tolist()[:-1]
            for cls_index in cls_index_list:
                assert token_ids[
                    cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
            cls_indexes = torch.LongTensor(cls_index_list).to(device)
            class_ids = torch.LongTensor(
                group_df["class_id"].values).to(device)
            data_list.append(
                [index,
                 len(group_df), token_ids, class_ids, cls_indexes])
        print(src, len(data_list))
        self.table_df = pd.DataFrame(data_list,
                                     columns=[
                                         "table_id", "num_col", "data_tensor",
                                         "label_tensor", "cls_indexes"
                                     ])
        """
        # NOTE: msato contains a small portion of single-col tables. keep it to be consistent.  
        if multicol_only:
            # Check
            num_all_tables = len(self.table_df)
            self.table_df = self.table_df[self.table_df["num_col"] > 1]
            assert len(self.table_df) == num_all_tables
        """

    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"]
        }
        #"idx": torch.LongTensor([idx])}
        #"cls_indexes": self.table_df.iloc[idx]["cls_indexes"]}

class GittablesCVTablewiseDataset(data.Dataset):

    def __init__(
            self,
            cv: int,
            split: str,
            src: str,  # train or test
            tokenizer: AutoTokenizer,
            max_length: int = 256,
            gt_only: bool = False,
            device: torch.device = None,
            base_dirpath: str = "./data/gittables",
            small_tag: str = ""):
        if device is None:
            device = torch.device('cpu')

    
        basename = "{}_all_{}_cv{}.csv"

        filepath = os.path.join(base_dirpath, basename.format(src, split, cv))
    
        df = pd.read_csv(filepath)
        df['class_id'] = df['class_id'].astype(int)
        if gt_only:
            df = df[df["class_id"] > -1]
        
        df.drop(df[(df['data'].isna()) & (df['class_id'] == -1)].index, inplace=True)
        df['col_idx'] = df['col_idx'].astype(int)
        df['data'] = df['data'].astype(str)
        # df.drop(df[(df['data'] == '') & (df['class_id'] == -1)].index, inplace=True)
        data_list = []
        for i, (index, group_df) in enumerate(df.groupby("table_id")):
            # if len(data_list) > 10:
            #     break
            labeled_columns = group_df[group_df['class_id'] > -1]
            unlabeled_columns = group_df[group_df['class_id'] == -1]
            # group_df = pd.concat([group_df[group_df['class_id'] > -1], unlabeled_columns.sample(min(10-len(labeled_columns), len(unlabeled_columns)))])
            # group_df = pd.concat([group_df[group_df['class_id'] > -1], unlabeled_columns[0:min(max(10-len(labeled_columns), 0), len(unlabeled_columns))]])
            group_df = pd.concat([group_df[group_df['class_id'] > -1], unlabeled_columns[0:min(max(8-len(labeled_columns), 0), len(unlabeled_columns))]])
            group_df.sort_values(by=['col_idx'], inplace=True)

            if max_length <= 128:
                cur_maxlen = min(max_length, 512 // len(list(group_df["class_id"].values)) - 1)
            else:
                cur_maxlen = max(1, max_length // len(list(group_df["class_id"].values)) - 1)
            # print(tokenizer.cls_token)
            # print(group_df["data"])
            token_ids_list = group_df["data"].apply(lambda x: tokenizer.encode(
                tokenizer.cls_token + " " + str(x), add_special_tokens=False, max_length=cur_maxlen, truncation=True)).tolist(
                )
            token_ids = torch.LongTensor(reduce(operator.add,
                                                token_ids_list)).to(device)
            cls_index_list = [0] + np.cumsum(
                np.array([len(x) for x in token_ids_list])).tolist()[:-1]
            for cls_index in cls_index_list:
                assert token_ids[
                    cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
            cls_indexes = torch.LongTensor(cls_index_list).to(device)
            class_ids = torch.LongTensor(
                group_df["class_id"].values).to(device)
            data_list.append(
                [index,
                 len(group_df), token_ids, class_ids, cls_indexes])
        print(src, len(data_list))
        self.table_df = pd.DataFrame(data_list,
                                     columns=[
                                         "table_id", "num_col", "data_tensor",
                                         "label_tensor", "cls_indexes"
                                     ])
        """
        # NOTE: msato contains a small portion of single-col tables. keep it to be consistent.  
        if multicol_only:
            # Check
            num_all_tables = len(self.table_df)
            self.table_df = self.table_df[self.table_df["num_col"] > 1]
            assert len(self.table_df) == num_all_tables
        """

    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"]
        }
        #"idx": torch.LongTensor([idx])}
        #"cls_indexes": self.table_df.iloc[idx]["cls_indexes"]}