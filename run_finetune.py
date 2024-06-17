import os
import subprocess
import time
import pickle
from multiprocessing import Process
from multiprocessing import Semaphore

'''run finetuning and evaluation on original datasets'''
task = 'turl-re'
# task = 'sato0'
# task = 'turl'
ml = 256
bs = 32
n_epochs = 20
# n_epochs = 10
base_model = 'bert-base-uncased'
# base_model = 'distilbert-base-uncased'
cl_tag = 'None/header/bert_1000000_10_32_256_5e-05_sample_row4,sample_row4_tfidf_entity_column_0.05_0_last'
ckpt_path = '/efs/checkpoints/'
dropout_prob = 0.5
from_scratch = False
# from_scratch = True # True means using Huggingface's pre-trained language model's checkpoint
eval_test = True
colpair = False
gpus = '0'
small_tag = ''

cmd = '''CUDA_VISIBLE_DEVICES={} python supcl_ft.py \
            --shortcut_name {} --task {} --max_length {} --batch_size {} --epoch {} \
            --dropout_prob {} --pretrained_ckpt_path "{}" --cl_tag {} --small_tag "{}" {} {} {}'''.format(
    gpus, base_model, task, ml, bs, n_epochs, dropout_prob,
    ckpt_path, cl_tag, small_tag,
    '--colpair' if colpair else '',
    '--from_scratch' if from_scratch else '',        
    '--eval_test' if eval_test else ''
)


os.system('{} & '.format(cmd))
