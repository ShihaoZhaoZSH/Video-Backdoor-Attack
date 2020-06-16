# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import time
import numpy
from six.moves import xrange
import tensorflow as tf
import math
import numpy as np


def placeholder_inputs(batch_size=16, num_frame_per_clib=16, crop_size=224, rgb_channels=3, flow_channels=2):

    rgb_images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                           num_frame_per_clib,
                                                           crop_size,
                                                           crop_size,
                                                           rgb_channels))
    flow_images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                           num_frame_per_clib,
                                                           crop_size,
                                                           crop_size,
                                                           flow_channels))
    labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size
                                                         ))
    is_training = tf.placeholder(tf.bool)
    return rgb_images_placeholder, flow_images_placeholder, labels_placeholder, is_training


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def tower_loss( logit, labels):
    print(labels)
    print(logit)
    print(logit.shape)
    cross_entropy_mean = tf.reduce_mean(
                  tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logit)
                  )

    total_loss = cross_entropy_mean
    return total_loss


def tower_loss_onehot( logit, labels):
    print(labels)
    print(logit)
    print(logit.shape)
    cross_entropy_mean = tf.reduce_mean(
                  tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logit)
                  )

    total_loss = cross_entropy_mean
    return total_loss



def tower_acc(logit, labels):
    correct_pred = tf.equal(tf.argmax(logit, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy


def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, wd):
    var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())
    if wd is not None:
        weight_decay = tf.nn.l2_loss(var)*wd
        tf.add_to_collection('weightdecay_losses', weight_decay)
    return var

