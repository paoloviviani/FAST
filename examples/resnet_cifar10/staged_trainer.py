#!/usr/bin/env python

import os
import sys
import subprocess
import time
import shutil

def epoch_file(epoch):
    return 'initw_'+str(epoch)+'.bin'
env_settings = os.environ.copy()

batch_size = 32
learning_rate = 0.0005
symbol_file = os.path.abspath('../../symbols/resnet18_v2.json')
init_file = '../../initialized_weights/resnet18_cifar10_init_batch_' + str(batch_size) + '.bin'
max_epochs = 50

env_settings['BATCH_SIZE'] = str(batch_size)
env_settings['LEARNING_RATE'] = str(learning_rate)
env_settings['SYMBOL_JSON'] = symbol_file
env_settings['INIT_WEIGHTS'] = init_file

executable = 'resnetGrid'
executable = os.path.join(os.path.abspath(os.pardir), executable)
command = [executable, '4', '4']

rank = env_settings['GAM_RANK']
timing_file = 'time_worker_'+str(rank)+'.log'
temp_bin_file = 'w_'+str(rank)+'.bin'
log = open(timing_file, 'a+')
log.write('Epoch epoch_time total_time\n')
log.flush()

start = time.time()

for epoch in range(max_epochs):
    if epoch > 0:
        if epoch % 4 == 0:
            learning_rate = learning_rate/2
            env_settings['LEARNING_RATE'] = str(learning_rate)

    env_settings['EPOCH'] = str(epoch)
    epoch_start = time.time()
    ret = subprocess.call(command, env=env_settings, stdout=sys.stdout, stderr=sys.stderr)
    if ret !=0:
        sys.exit("Error")
    
    if os.path.isfile(temp_bin_file):
        shutil.move(temp_bin_file, epoch_file(epoch))
        if os.path.isfile(epoch_file(epoch-1)):
            os.remove(epoch_file(epoch-1))

    while not os.path.isfile(epoch_file(epoch)):
        time.sleep(0.1)

    env_settings['INIT_WEIGHTS'] = epoch_file(epoch)
    end = time.time()
    log.write(str(epoch)+' '+str(end-epoch_start)+' '+str(end-start)+'\n')
    log.flush()

