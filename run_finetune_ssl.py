import os
import subprocess
import time
import pickle
from multiprocessing import Process
from multiprocessing import Semaphore

'''run finetuning and evaluation on small samples under semi-supervised setting'''

# task = 'turl-re'
task = 'sato'
# task = 'turl'
# ml = 256
# bs = 64
ml = 256
bs = 16
n_epochs = 30
base_model = 'bert-base-uncased'
# base_model = 'distilbert-base-uncased'
cl_tag = 'wikitables/header/bert_1000_10_32_256_5e-05_sample_row4,sample_row4_tfidf_entity_column_0.05_0_last'
ckpt_path = '/efs/checkpoints/'

dropout_prob = 0.0
from_scratch = False
# from_scratch = True 
eval_test = True
gpu = 0

ssl = 'cpl_labelednoda' # full optimizations
# ssl = 'labelguess' # no balance
augment_op = 'sample_row4'
u_ratio = 3
u_lambda = 0.01
# u_lambda = 0.005    
p_cutoff = 0.95

for rate in ['0.05']: # different sample rates
# for rate in ['0.02', '0.05', '0.1']:
# for rate in ['t20_v4', 't50_v10',  't100_v20']:
    for i in range(5):
        if 'sato' not in task:
            small_tag = "by_table_{}_{}".format(rate, i)
        else:
            small_tag = "by_table_{}".format(rate)
        cmd = '''CUDA_VISIBLE_DEVICES={} python supcl_ft_ssl.py \
                    --ssl {} --augment_op {} --u_lambda {} --u_ratio {} --p_cutoff {} \
                    --shortcut_name {} --task {} --max_length {} --batch_size {} --epoch {} \
                    --dropout_prob {} --pretrained_ckpt_path "{}" --cl_tag {} --small_tag "{}" {} {}'''.format(
            gpu, ssl, augment_op, u_lambda, u_ratio, p_cutoff,
            base_model, task if task != 'sato' else (task + str(i)), ml, bs, n_epochs, dropout_prob,
            ckpt_path, cl_tag, small_tag,
            '--from_scratch' if from_scratch else '',        
            '--eval_test' if eval_test else ''
        )

        
        os.system('{}'.format(cmd))
