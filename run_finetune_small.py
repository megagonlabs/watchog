import os
import subprocess
import time
import pickle
from multiprocessing import Process
from multiprocessing import Semaphore

'''run finetuning and evaluation on small samples without semi-supervised learning'''
# task = 'turl-re'
task = 'turl'
# task = 'sato'
# ml = 256
# bs = 64
ml = 256
bs = 32
n_epochs = 30
base_model = 'bert-base-uncased'
cl_tag = 'wikitables/header/bert_1000_10_32_256_5e-05_sample_row4,sample_row4_tfidf_entity_column_0.05_0_last'
ckpt_path = '/efs/checkpoints/'
gpu = 0

dropout_prob = 0.0
from_scratch = False
# from_scratch = True # True means using Huggingface's pre-trained language model's checkpoint
eval_test = True
colpair = False

# for rate in ['0.02', '0.05', '0.1', '0.25']:
for rate in [ '0.02']:
# for rate in ['t20_v4', 't50_v10', 't100_v20']:
    for i in range(5):
        small_tag = "by_table_{}_{}".format(rate, i)
        
        cmd = '''CUDA_VISIBLE_DEVICES={} python3 supcl_ft.py \
                    --shortcut_name {} --task {} --max_length {} --batch_size {} --epoch {} \
                    --dropout_prob {} --pretrained_ckpt_path "{}" --cl_tag {} --small_tag "{}" {} {} {}'''.format(
            gpu, base_model, task if task != 'sato' else (task + str(i)), ml, bs, n_epochs, dropout_prob,
            ckpt_path, cl_tag, small_tag,
            '--colpair' if colpair else '',
            '--from_scratch' if from_scratch else '',        
            '--eval_test' if eval_test else ''
        )
        
        os.system('{}'.format(cmd))
