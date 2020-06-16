import os
import sys
sys.path.append('../')
import time
import numpy
from six.moves import xrange
import tensorflow as tf
import math
import numpy as np
from i3d import InceptionI3d
from utils import *
import cv2
import PIL.Image as Image
import input_data


flags = tf.app.flags
gpu_num = 1
flags.DEFINE_integer('batch_size', 1, 'Batch size.')
flags.DEFINE_integer('num_frame_per_clib', 64, 'Nummber of frames per clib')
flags.DEFINE_integer('crop_size', 224, 'Crop_size')
flags.DEFINE_integer('rgb_channels', 3, 'Channels for input')
flags.DEFINE_integer('classics', 101, 'The num of class')
flags.DEFINE_integer("sample_rate", 1, "sample rate in each clip")
FLAGS = flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def addTrigger(images, indices, triggerSize=24, save_path=None):

    trig = np.load("trigger" + str(triggerSize) + ".npy")
    targetImgs = list()
    cnt = 0
    for k in range(len(images)):
        for i in range(triggerSize):
            for j in range(triggerSize):
                images[k][-(i + 1)][-(j + 1)] = trig[0][k][-(i + 1)][-(j + 1)]
                t = trig[0][k][-(i + 1)][-(j + 1)]        
        if save_path:
            img = images[k]
            img = Image.fromarray(img.astype("uint8"))
            img.save(save_path + "image_{:05d}.jpg".format(indices[cnt]))
            cnt += 1

    return targetImgs


def run():
    
    pre_model_save_dir = "../models/rgb_imagenet_10000_6_64_0.0001_decay"
    apply_cl_file = '../traintestlist/enhance_trigger.txt'

    file = list(open(apply_cl_file, 'r'))
    num_videos = len(file)
    print("Number of videos={}".format(num_videos))
    
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
        
        loss = -tower_loss(logit, labels_placeholder)
        grad = tf.gradients(loss, rgb_images_placeholder)[0]

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
    step = num_videos // batch_size
    cnt = 0
    acc_all = 0
    et_step = 10
    epsilon = 8
    trigger_size = 24
    lines = list(open(apply_cl_file, "r").readlines())
    output_root = "/data/UCF101/UCF-101_extract/TargetVideo_" + str(epsilon) + "_trig" + str(trigger_size)
    
    import shutil
    if os.path.exists(output_root):
        shutil.rmtree(output_root)
    os.mkdir(output_root)
    
    for i in range(step):
        start = i * batch_size
        rgb_val_images, flow_val_images, val_labels, _, _, _ = input_data.read_clip_and_label(
            filename=apply_cl_file, 
            batch_size=batch_size, 
            start_pos=start, 
            num_frames_per_clip=FLAGS.num_frame_per_clib, 
            crop_size=FLAGS.crop_size, 
            shuffle=False
            )
        filename = lines[start].split(" ")[0]
        videoname = filename.split("/")[7]
        videopath = output_root + "/" + videoname + "/"
        os.mkdir(videopath)
        filename_img = filename.replace("UCF-101_extract_flow", "UCF-101_extract")
        frameNum = 0
        for parent, dirnames, filenames in os.walk(filename_img):
            frameNum = len(filenames)
        s_index = (frameNum - FLAGS.num_frame_per_clib) // 2
        indices = list(range(s_index + 1, s_index + 1 + FLAGS.num_frame_per_clib))
        
        x = rgb_val_images
        for i in range(et_step):
            grad_val, logit_val = sess.run([grad, logit], feed_dict={rgb_images_placeholder: x,
                                            labels_placeholder: val_labels,
                                            is_training: False})
            x = np.add(x, -1 * np.sign(grad_val), out=x, casting='unsafe')
            index = np.argmax(logit_val, axis=1)

            x = np.clip(x, rgb_val_images - epsilon, rgb_val_images + epsilon)
            x = np.clip(x, 0, 255)
        addTrigger(x[0], indices, trigger_size, save_path=videopath)
        print(start, videopath)


if __name__ == "__main__":
    run()

