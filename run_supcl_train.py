import os

pretrain_data = 'wikitables'
ml = 256
bs = 32
# cuda_devices = '0,1,2,3'
cuda_devices = '0'

'''unsupervised'''
# mode = 'simclr'
task = 'None'
'''supervised with header'''
mode = 'supcon' 
task = 'header'

ao = 'sample_row4,sample_row4'
sm = 'tfidf_entity'
# lm = 'distilbert'
lm = 'bert'

gpus = ','.join([str(i) for i in range(len(cuda_devices.split(',')))])
n_epochs = 10
size = 100000
cnt = 0
run_id = 0
temp = 0.05
# temp = 0.07

data_path = "/efs/task_datasets/TURL/"

if len(cuda_devices.split(',')) > 1:
    cmd = """accelerate launch --config_file accelerate_config.yaml supcl_train.py --fp16 \
        --data_path %s \
        --pretrain_data %s \
        --mode %s \
        --task %s \
        --batch_size %s \
        --lr 5e-5 \
        --temperature %s \
        --lm %s \
        --n_epochs %d \
        --max_len %d \
        --size %d \
        --save_model \
        --augment_op %s \
        --sample_meth %s \
        --run_id %d""" % (data_path, pretrain_data, mode, task, bs, temp, lm, n_epochs, ml, size, ao, sm, run_id)
else:
    cmd = """python3 supcl_train.py --fp16 \
        --data_path %s \
        --pretrain_data %s \
        --mode %s \
        --task %s \
        --batch_size %s \
        --lr 5e-5 \
        --temperature %s \
        --lm %s \
        --n_epochs %d \
        --max_len %d \
        --size %d \
        --save_model \
        --augment_op %s \
        --sample_meth %s \
        --run_id %d""" % (data_path, pretrain_data, mode, task, bs, temp, lm, n_epochs, ml, size, ao, sm, run_id)
print(cmd)
# os.system('CUDA_VISIBLE_DEVICES={} {}'.format(
os.system('CUDA_VISIBLE_DEVICES={} {} &> {} &'.format(
    cuda_devices, 
    cmd, 
    'outputs/supcl_train_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.txt'.format(
        pretrain_data, lm, mode, task, size, n_epochs, bs, ml, ao, sm, temp, run_id)))
                
