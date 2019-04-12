#!/usr/bin/env python

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import print_function
import random, sys, time

import mxnet as mx
from mxnet import autograd, gluon, kv, nd
from mxnet.gluon.model_zoo import vision

import numpy as np
import socket
import os
import sys

if len(sys.argv) < 4:
  sys.exit("Wrong parameters")

# Create a distributed key-value store
storetype = str(sys.argv[1]) # 'dist_sync, 'dist_async', 'local'
store = kv.create(storetype)

# Clasify the images into one of the 10 digits
num_outputs = 10

# How many epochs to run the training
epochs = 100

# Effective batch size across all GPUs
batch_size = int(sys.argv[2])
learning_rate = float(sys.argv[3])

# Create the context (a list of all GPUs to be used for training)
#ctx = [mx.gpu(i) for i in range(gpus_per_machine)]
ctx = [mx.cpu()]

hostname = socket.gethostname()
dirname = "image-classification_"+str(batch_size)+"_lr"+str(learning_rate)+"_" + str(store.num_workers) + "_nodes_" + str(storetype)
path = os.path.join(os.getcwd(), dirname)
if not os.path.exists(path) and store.rank == 0:
   os.mkdir(path)

while not os.path.exists(path):
	time.sleep(0.01)

filename = "log_" + hostname + ".txt"
filepath = os.path.join(dirname, filename)
out_file = open(filepath, "w", 0)
out_file.write("Epoch\tTime\tTrain accuracy\tTest accuracy\n")

# Convert to float 32
# Having channel as the first dimension makes computation more efficient. Hence the (2,0,1) transpose.
# Dividing by 255 normalizes the input between 0 and 1
def transform(data, label):
    return nd.transpose(data.astype(np.float32), (2,0,1))/255, label.astype(np.float32)

train_data = mx.io.ImageRecordIter(
  path_imgrec = "./cifar/cifar10_train.rec",
  # mean_img    = "data/cifar/mean.bin",
  resize      = -1,
  data_shape  = (3, 32, 32),
  batch_size  = batch_size,
  rand_crop   = True,
  rand_mirror = True,
  shuffle     = True,
  pad         = 4,
  num_parts   = store.num_workers,
  part_index  = store.rank)

test_data = mx.io.ImageRecordIter(
  path_imgrec = "./cifar/cifar10_val.rec",
  # mean_img    = "data/cifar/mean.bin",
  resize      = -1,
  rand_crop   = False,
  rand_mirror = False,
  pad         = 4,
  data_shape  = (3, 32, 32),
  batch_size  = batch_size)

# Use ResNet from model zoo
net = vision.resnet18_v2()

# Initialize the parameters with Xavier initializer
net.collect_params().initialize(mx.init.Xavier(magnitude=2), ctx=ctx)

# Use Adam optimizer. Ask trainer to use the distributer kv store.
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': learning_rate, 'rescale_grad': 1.0/(batch_size*store.num_workers), 'clip_gradient': 10}, kvstore=store)

# Evaluate accuracy of the given network using the given data
def test(ctx, val_data, net):
    metric.reset()
    val_data.reset()
    for batch in val_data:
        data = gluon.utils.split_and_load(batch.data[0].astype(np.float32, copy=False),
                                          ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0].astype(np.float32, copy=False),
                                           ctx_list=ctx, batch_axis=0)
        # outputs = [net(X) for X in data]
        outputs = []
        # Ask autograd to remember the forward pass
        with autograd.record():
          # Compute the loss on all GPUs
          for x, y in zip(data, label):
              z = net(x)
              outputs.append(z)
        metric.update(label, outputs)
    return metric.get()

# We'll use cross entropy loss since we are doing multiclass classification
loss = gluon.loss.SoftmaxCrossEntropyLoss()

# Run one forward and backward pass on multiple GPUs
def forward_backward(net, data, label):
    
    outputs = []
    losses = []
    # Ask autograd to remember the forward pass
    with autograd.record():
      # Compute the loss on all GPUs
      for x, y in zip(data, label):
          z = net(x)
          L = loss(z, y)
          # store the loss and do backward after we have done forward
          # on all GPUs for better speed on multiple GPUs.
          losses.append(L)
          outputs.append(z)

    # Run the backward pass (calculate gradients) on all GPUs
    for l in losses:
        l.backward()
    
    return outputs

metric = mx.metric.Accuracy()

# Train a batch using multiple GPUs
def train_batch(batch, ctx, net, trainer):

    data = gluon.utils.split_and_load(batch.data[0].astype(np.float32), ctx_list=ctx, batch_axis=0)
    label = gluon.utils.split_and_load(batch.label[0].astype(np.float32), ctx_list=ctx, batch_axis=0)

    # Run the forward and backward pass
    outputs = forward_backward(net, data, label)

    # Update the parameters
    trainer.step(batch_size)
    metric.update(label, outputs)
    return metric.get()[1]


total_time = 0
# Run as many epochs as required
for epoch in range(epochs):
    tic = time.time()

    # Iterate through batches and run training using multiple GPUs
    batch_num = 1
    train_data.reset()
    metric.reset()
    for i, batch in enumerate(train_data):

        # Train the batch using multiple GPUs
        train_acc = train_batch(batch, ctx, net, trainer)

        batch_num += 1
        print("Train acc ", train_acc)
        sys.stdout.flush()

    # Print test accuracy after every epoch
    test_accuracy = test(ctx, test_data, net)[1]
    epoch_time = time.time()-tic
    total_time = total_time + epoch_time
    # print(test_accuracy)
    print("%s -> Epoch %d: Train acc %f. Test acc %f. Epoch time %f. Total time %f" % (hostname, epoch, train_acc, test_accuracy, epoch_time, total_time))
    sys.stdout.flush()
    out_file.write("%d\t%f\t%f\t%f\n" % (epoch, total_time, train_acc, test_accuracy))
    out_file.flush()

