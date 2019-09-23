import tensorflow as tf
import numpy as np
from utils.utils import *
from model import *
import sys
import os
import math
import time
from utils.data_helper import data_loader
from model import xavier_init, he_normal_init
from utils.inception_score import get_inception_score
from utils.fid import get_fid

fp = open("FID_IS_result.txt",'w')

def fid_is_eval():
    pathes = os.listdir("results/generated/")
    for path in pathes:
        info = path.split('_')
        dataset = info[0]
        model_name = path.split('.')[0]
        if dataset == "cifar10":
            data = np.load('results/generated/{}'.format(path))
            mb_size, X_dim, width, height, channels,len_x_train, x_train, y_train, len_x_test, x_test, y_test  = data_loader(dataset)  
            real_set = x_train
            img_set = data['arr_0']
            print("Calculating Fréchet Inception Distance for {}".format(model_name))
            print("Calculating Fréchet Inception Distance for {}".format(model_name), file =fp)
            fid_set_r = real_set*255.0
            fid_set_r = fid_set_r.astype(np.uint8)
            fid_set_r = np.transpose(fid_set_r, (0, 3, 1, 2))
            fid_set_i = img_set*255.0
            fid_set_i = fid_set_i.astype(np.uint8)
            fid_set_i = np.transpose(fid_set_i, (0, 3, 1, 2))
            fid_score = get_fid(fid_set_r, fid_set_i)
            print("FID: {}".format(fid_score))
            print("FID: {}".format(fid_score),file=fp)
            tf.reset_default_graph()
            print("Calculating inception score for {}".format(model_name))
            print("Calculating inception score for {}".format(model_name), file =fp)
            is_set = img_set*2.0 - 1.0
            is_set = np.transpose(is_set, (0, 3, 1, 2))

            mean, std = get_inception_score(is_set)
            print("mean: {} std: {}".format(mean,std))
            print("mean: {} std: {}".format(mean,std), file = fp)
            tf.reset_default_graph()
                
fid_is_eval()
fp.close()