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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import xrange
import tensorflow as tf
import PIL.Image as Image
import random
import numpy as np
import cv2
import time



def sample_data(ori_arr, num_frames_per_clip, sample_rate):
    ret_arr = []
    for i in range(int(num_frames_per_clip/sample_rate)):
        ret_arr.append(ori_arr[int(i*sample_rate)])
    return ret_arr


def get_data(filename, mode, num_frames_per_clip, sample_rate, is_flow=False, s_index=-1):
    ret_arr = []
    filenames = ''
    if "TargetVideo_train" in filename:
        s_index = -1
    for parent, dirnames, filenames in os.walk(filename):
        
        filenames_tmp = list()
        for filename_ in filenames:
            if filename_.startswith(mode):
                filenames_tmp.append(filename_)
        filenames = filenames_tmp

        if len(filenames)==0:
            print('DATA_ERRO: %s'%filename)
            return [], s_index
        if (len(filenames)-s_index) <= num_frames_per_clip:
            filenames = sorted(filenames)
            if len(filenames) < num_frames_per_clip:
                for i in range(num_frames_per_clip):
                    if i >= len(filenames):
                        i = len(filenames)-1
                    image_name = str(filename) + '/' + str(filenames[i])
                    img = Image.open(image_name)
                    img_data = np.array(img)
                    ret_arr.append(img_data)
            else:
                for i in range(num_frames_per_clip):
                    image_name = str(filename) + '/' + str(filenames[len(filenames)-num_frames_per_clip+i])
                    img = Image.open(image_name)
                    img_data = np.array(img)
                    ret_arr.append(img_data)
            return sample_data(ret_arr, num_frames_per_clip, sample_rate), s_index
    
        filenames_tmp = list()
        for filename_ in filenames:
            if filename_.startswith(mode):
                filenames_tmp.append(filename_)
        filenames = filenames_tmp
  
    filenames = sorted(filenames)
    if s_index < 0:
        s_index = random.randint(0, len(filenames) - num_frames_per_clip)
    for i in range(int(num_frames_per_clip/sample_rate)):
        if "TargetVideo_train" in filename:
            image_name = str(filename) + "/" + str(filenames[int(i * sample_rate)])
        else:
            image_name = str(filename) + '/' + str(filenames[int(i*sample_rate)+s_index])
        img = Image.open(image_name)
        if is_flow and "TargetVideo" in filename:
            img = img.convert("L")
        img_data = np.array(img)
        ret_arr.append(img_data)
    return ret_arr, s_index



def get_frames_data(filename, num_frames_per_clip, sample_rate, add_flow, label):

    filename_img = filename.replace("UCF-101_extract_flow", "UCF-101_extract")

    rgb_ret_arr, s_index = get_data(filename_img, "i", num_frames_per_clip, sample_rate, False)
    if not add_flow:
        return rgb_ret_arr, [], s_index
    flow_x, _ = get_data(filename, "x", num_frames_per_clip, sample_rate, True, s_index)
    flow_x = np.expand_dims(flow_x, axis=-1)
    flow_y, _ = get_data(filename, "y", num_frames_per_clip, sample_rate, True, s_index)
    flow_y = np.expand_dims(flow_y, axis=-1)
    flow_ret_arr = np.concatenate((flow_x, flow_y), axis=-1)
    return rgb_ret_arr, flow_ret_arr, s_index


def data_process(tmp_data, crop_size):
    img_datas = []
    crop_x = 0
    crop_y = 0
    for j in xrange(len(tmp_data)):
        img = Image.fromarray(tmp_data[j].astype(np.uint8))
        if img.width > img.height:
            scale = float(256) / float(img.height)
            img = np.array(cv2.resize(np.array(img), (int(img.width * scale + 1), 256))).astype(np.float32)
        else:
            scale = float(256) / float(img.width)
            img = np.array(cv2.resize(np.array(img), (256, int(img.height * scale + 1)))).astype(np.float32)
        img = Image.fromarray(img.astype(np.uint8))
        img = img.resize((crop_size, crop_size))
        img = np.array(img).astype(np.float32)
        img_datas.append(img)
    return img_datas


def read_clip_and_label(filename, batch_size, start_pos=-1, num_frames_per_clip=64, sample_rate=1, crop_size=224, shuffle=True, add_flow=False):
    lines = open(filename, 'r')
    read_dirnames = []
    rgb_data = []
    flow_data = []
    label = []
    batch_index = 0
    next_batch_start = -1
    lines = list(lines)
    if start_pos < 0:
        shuffle = True
    if shuffle:
        video_indices = range(len(lines))
        random.seed(time.time())
        video_indices = list(video_indices)
        random.shuffle(video_indices)
    else:
        video_indices = range(start_pos, len(lines))
    for index in video_indices:
        if batch_index >= batch_size:
            next_batch_start = index
            break
        line = lines[index].strip('\n').split()
        dirname = line[0]
        tmp_label = int(line[2])
        if not shuffle:
            pass
        tmp_rgb_data, tmp_flow_data, s_index = get_frames_data(dirname, num_frames_per_clip, sample_rate, add_flow, tmp_label)
        if len(tmp_rgb_data) != 0:
            rgb_img_datas = data_process(tmp_rgb_data, crop_size)
            if add_flow:
                flow_img_datas = data_process(tmp_flow_data, crop_size)
                flow_data.append(flow_img_datas)
            rgb_data.append(rgb_img_datas)
            label.append(int(tmp_label))
            batch_index = batch_index + 1
            read_dirnames.append(dirname)

    valid_len = len(rgb_data)
    pad_len = batch_size - valid_len
    if pad_len:
        for i in range(pad_len):
            rgb_data.append(rgb_data[-1])
            flow_data.append(flow_data[-1])
            label.append(int(label[-1]))

    np_arr_rgb_data = np.array(rgb_data).astype(np.float32)
    np_arr_flow_data = np.array(flow_data).astype(np.float32)
    np_arr_label = np.array(label).astype(np.int64)

    return np_arr_rgb_data, np_arr_flow_data, np_arr_label.reshape(batch_size), next_batch_start, read_dirnames, valid_len
