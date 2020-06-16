# Clean-Label Backdoor Attacks on Video Recognition Models, CVPR2020
## Dataset: UCF101  Model: I3D
## Environment: python3.6.5 tensorflow-gpu1.14
### You can firstly run train_clean_model.py to get a clean-trained I3D model. The generte_trigger.py and enhance_trigger.py correspond to Backdoor Trigger Generation and Enhancing Backdoor Trigger sections in the paper, respectively. Then you can run train_bad_model.py to train a bad model and run test.py to test it.
#### It's evelopped based on [LossNAN/I3D-Tensorflow] (https://github.com/LossNAN/I3D-Tensorflow)
