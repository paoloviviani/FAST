#!/usr/bin/env python

import os
import sys
import subprocess

env_settings = os.environ.copy()

batch_size = '32'
learning_rate = '0.01'
symbol_file = '../resnet18_v2.json'
init_file = '../../initialized_weights/resnet18_cifar10_init_' + batch_size + '.bin'
max_epochs = 3

env_settings['BATCH_SIZE'] = batch_size
env_settings['LEARNING_RATE'] = learning_rate
env_settings['SYMBOL_JSON'] = symbol_file
env_settings['INIT_WEIGHTS'] = init_file

executable = 'resnetGrid'
executable = os.path.join(os.path.abspath(os.pardir), executable)
command = [executable, '4', '4']


for epoch in range(max_epochs):
    ret = subprocess.call(command, env=env_settings, stdout=sys.stdout, stderr=sys.stderr)
    if ret !=0:
        sys.exit("Error")
    env_settings['INIT_WEIGHTS'] = "w_"+str(epoch+1)+".bin"
