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
sys.path.append('../../')
import time
import numpy
from six.moves import xrange
import tensorflow as tf
import math
import numpy as np
from i3d import InceptionI3d
from utils import *
import input_data


flags = tf.app.flags
gpu_num = 1
flags.DEFINE_integer('batch_size', 4, 'Batch size.')
flags.DEFINE_integer('num_frame_per_clib', 64, 'Nummber of frames per clib')
flags.DEFINE_integer('crop_size', 224, 'Crop_size')
flags.DEFINE_integer('rgb_channels', 3, 'Channels for input')
flags.DEFINE_integer('classics', 101, 'The num of class')
flags.DEFINE_integer("sample_rate", 1, "sample rate in each clip")
FLAGS = flags.FLAGS


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("epsilon", type=int, choices=[0, 8])
parser.add_argument("portion", type=float, choices=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
parser.add_argument("gpu_id", type=int, choices=[0, 1, 2, 3])
parser.add_argument("testfile", type=str)
parser.add_argument("trigSize", type=int, choices=[4, 8, 12, 16, 20, 24, 30])
args = parser.parse_args()
epsilon_ = args.epsilon
portion_ = args.portion
gpu_id_ = args.gpu_id
testfile_ = args.testfile
trigSize = args.trigSize
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id_)


def run_training():
    
    pre_model_save_dir = "./models/rgb_" + str(epsilon_) + "_" + str(int(portion_ * 100))  + "_imagenet_10000_6_64_0.0001_decay_trig" + str(trigSize)

    test_list_file = testfile_
    file = list(open(test_list_file, 'r'))
    num_test_videos = len(file)
    print("Number of test videos={}".format(num_test_videos))
    
    with tf.Graph().as_default():
        rgb_images_placeholder, _, labels_placeholder, is_training = placeholder_inputs(
                        FLAGS.batch_size * gpu_num,
                        FLAGS.num_frame_per_clib / FLAGS.sample_rate,
                        FLAGS.crop_size,
                        FLAGS.rgb_channels
                        )

        with tf.variable_scope('RGB'):
            logit, _ = InceptionI3d(
                                num_classes=FLAGS.classics,
                                spatial_squeeze=True,
                                final_endpoint='Logits',
                                name='inception_i3d'
                                )(rgb_images_placeholder, is_training)
        norm_score = tf.nn.softmax(logit)
        accuracy = tower_acc(norm_score, labels_placeholder)

        rgb_variable_map = {}
        for variable in tf.global_variables():
            if variable.name.split("/")[0] == "RGB" and "Adam" not in variable.name.split("/")[-1]:
                rgb_variable_map[variable.name.replace(':0', '')] = variable               
        saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)
        
        init = tf.global_variables_initializer()

        sess = tf.Session(
                        config=tf.ConfigProto(allow_soft_placement=True)
                        )
        sess.run(init)

    ckpt = tf.train.get_checkpoint_state(pre_model_save_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print ("loading checkpoint %s,waiting......" % ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        print ("load complete!")

    batch_size = FLAGS.batch_size
    step = num_test_videos // batch_size
    cnt = 0
    acc_all = 0
    res_cmp = list()
    for i in range(step):
        start = i * batch_size
        rgb_val_images, flow_val_images, val_labels, _, _, _ = input_data.read_clip_and_label(
            filename=test_list_file, 
            batch_size=batch_size, 
            start_pos=start, 
            num_frames_per_clip=FLAGS.num_frame_per_clib, 
            crop_size=FLAGS.crop_size, 
            shuffle=False
            )
        
        if "target" in testfile_:
            trig = np.load("trigger" + str(trigSize) + ".npy")
            for j in range(FLAGS.batch_size):
                for k in range(FLAGS.num_frame_per_clib):
                    for l in range(trigSize):
                        for m in range(trigSize):
                            rgb_val_images[j][k][-(l + 1)][-(m + 1)] = trig[0][k][-(l + 1)][-(m + 1)]
        
        acc, nc, lb= sess.run([accuracy, norm_score, labels_placeholder], feed_dict={
            rgb_images_placeholder: rgb_val_images, 
            labels_placeholder: val_labels, 
            is_training: False
            })
        cnt += 1
        acc_all += acc
        print(start, acc_all/cnt, acc, np.argmax(nc, axis=1))
    print(acc_all / cnt)

    
def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
