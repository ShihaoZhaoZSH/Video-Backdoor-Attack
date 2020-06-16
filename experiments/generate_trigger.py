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
import sys
sys.path.append('../')
import time
import numpy
from six.moves import xrange
import tensorflow as tf
import input_data
import math
import numpy as np
from i3d import InceptionI3d
from utils import *
from tensorflow.python import pywrap_tensorflow


flags = tf.app.flags
gpu_num = 1
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 10000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 1, 'Batch size.')
flags.DEFINE_integer('num_frame_per_clib', 64, 'Nummber of frames per clib')
flags.DEFINE_integer('crop_size', 224, 'Crop_size')
flags.DEFINE_integer('rgb_channels', 3, 'RGB_channels for input')
flags.DEFINE_integer('flow_channels', 2, 'FLOW_channels for input')
flags.DEFINE_integer('classics', 101, 'The num of class')
FLAGS = flags.FLAGS


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--epsilon", type=int, choices=[0, 8], default=8)
parser.add_argument("--portion", type=float, choices=[0.1, 0.3, 1.0], default=0.3)
parser.add_argument("--gpu_id", type=int, choices=[0, 1, 2, 3], default=0)
parser.add_argument("--trigger_size", type=int, choices=[4, 8, 12, 16, 20, 24, 28, 32, 36, 40], default=24)
args = parser.parse_args()
epsilon_ = args.epsilon
portion_ = args.portion
gpu_id_ = args.gpu_id
trigger_size = args.trigger_size
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id_)


def run_training():

    rgb_pre_model_save_dir = "./models/rgb_imagenet_10000_6_64_0.0001_decay"

    with tf.Graph().as_default():
        global_step = tf.get_variable(
                        'global_step',
                        [],
                        initializer=tf.constant_initializer(0),
                        trainable=False
                        )
        rgb_images_placeholder, flow_images_placeholder, labels_placeholder, is_training = placeholder_inputs(
                        FLAGS.batch_size * gpu_num,
                        FLAGS.num_frame_per_clib,
                        FLAGS.crop_size,
                        FLAGS.rgb_channels,
                        FLAGS.flow_channels
                        )

        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps=10000, decay_rate=0.1, staircase=True)
        opt_rgb = tf.train.AdamOptimizer(learning_rate)
        with tf.variable_scope('RGB'):
            rgb_logit, _ = InceptionI3d(
                                    num_classes=FLAGS.classics,
                                    spatial_squeeze=True,
                                    final_endpoint='Logits'
                                    )(rgb_images_placeholder, is_training)

        rgb_loss = tower_loss(
                                rgb_logit,
                                labels_placeholder
                                )
        labels_placeholder2 = tf.placeholder(tf.int64, shape=(FLAGS.batch_size))
        rgb_loss2 = -tower_loss(rgb_logit, labels_placeholder2)

        rgb_loss3 = rgb_loss + rgb_loss2

        grad = tf.gradients(rgb_loss3, rgb_images_placeholder)[0]

        accuracy = tower_acc(rgb_logit, labels_placeholder)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            rgb_grads = opt_rgb.compute_gradients(rgb_loss)
            apply_gradient_rgb = opt_rgb.apply_gradients(rgb_grads, global_step=global_step)
            train_op = tf.group(apply_gradient_rgb)
            null_op = tf.no_op()

        rgb_variable_map = {}
        for variable in tf.global_variables():
            if variable.name.split("/")[0] == "RGB" and "Adam" not in variable.name.split("/")[-1]: 
                rgb_variable_map[variable.name.replace(':0', '')] = variable
        rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        sess = tf.Session(
                        config=tf.ConfigProto(allow_soft_placement=True)
                        )
        sess.run(init)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('rgb_loss', rgb_loss)
        tf.summary.scalar('learning_rate', learning_rate)
        merged = tf.summary.merge_all()
    ckpt = tf.train.get_checkpoint_state(rgb_pre_model_save_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print("loading checkpoint %s,waiting......" % ckpt.model_checkpoint_path)
        rgb_saver.restore(sess, ckpt.model_checkpoint_path)
        print("load complete!")

    mask_val = np.zeros((FLAGS.batch_size, FLAGS.num_frame_per_clib, FLAGS.crop_size, FLAGS.crop_size, FLAGS.rgb_channels)) + 255.0 / 2
    index_ = np.array([100])

    for step in xrange(FLAGS.max_steps):
        start_time = time.time()
        rgb_train_images, flow_train_images, train_labels, _, _, _ = input_data.read_clip_and_label(
                      filename = "../traintestlist/generate_trigger.txt",
                      batch_size=FLAGS.batch_size * gpu_num,
                      num_frames_per_clip=FLAGS.num_frame_per_clib,
                      crop_size=FLAGS.crop_size,
                      shuffle=True
                      )

        rgb_train_images_ = rgb_train_images
        
        for k in range(FLAGS.num_frame_per_clib):
            for i in range(trigger_size):
                for j in range(trigger_size):
                    rgb_train_images_[0][k][-(i + 1)][-(j + 1)] = mask_val[0][k][-(i + 1)][-(j + 1)]
        train_labels_ = train_labels
        # target class
        train_labels = np.array([0])
        grad_, logit_ = sess.run([grad, rgb_logit], feed_dict={
                      rgb_images_placeholder: rgb_train_images_,
                      labels_placeholder: train_labels,
                      is_training: False,
                      labels_placeholder2: np.array(index_)
                      })
        mask_val = np.add(mask_val, -1 * np.sign(grad_), casting='unsafe')
        mask_val = np.clip(mask_val, 0, 255)
        index_ = np.argmax(logit_, axis=1)
        print(index_, logit_[0][index_], train_labels_)
        print([0], logit_[0][0])
        
        duration = time.time() - start_time
        print('Step %d: %.3f sec' % (step, duration))
        if (step + 1) % 100 == 0 or (step + 1) == FLAGS.max_steps:
            np.save("trigger" + str(trigger_size), mask_val)
            print("save......")
        

def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
