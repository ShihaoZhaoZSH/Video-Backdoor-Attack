# Clean-Label Backdoor Attacks on Video Recognition Models
## Introduction:
* Environment: Python3.6.5, TensorFlow-gpu1.14
* Dataset: UCF101, Model: I3D
## Usage:
This is our paper [link](https://arxiv.org/abs/2003.03030). You can firstly run train_clean_model.py to get a clean-trained I3D model. The generate_trigger.py and enhance_trigger.py correspond to Backdoor Trigger Generation and Enhancing Backdoor Trigger sections in the paper, respectively. Then you can run train_bad_model.py to train a bad model and run test.py to test it. It's developped based on [LossNAN/I3D-Tensorflow](https://github.com/LossNAN/I3D-Tensorflow).
