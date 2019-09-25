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

mb_size=256
def classifier_one(dataset, model_name, x_target, y_target, len_x_target):

    NUM_CLASSES = 10
    fp = open("classifier_result.txt",'a')
    print("dataset: {}; model name: {} Evaluation start.".format(dataset,model_name))
    _, X_dim, width, height, channels,len_x_train, x_train, y_train, len_x_test, x_test, y_test  = data_loader(dataset)


    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            #input placeholder
            input_shape=[None, width, height, channels]
            filter_sizes=[5, 5, 5, 5, 5]        
            hidden = 128         

            n_filters=[channels, hidden, hidden*2, hidden*4]

            X = tf.placeholder(tf.float32, shape=[None, width, height,channels])  
            Y = tf.placeholder(tf.float32, shape=[None,  NUM_CLASSES])      

            #discriminator variables
            W1 = tf.Variable(he_normal_init([5,5,channels, hidden//2]))
            W2 = tf.Variable(he_normal_init([5,5, hidden//2,hidden]))
            W3 = tf.Variable(he_normal_init([5,5,hidden,hidden*2]))
            W4 = tf.Variable(xavier_init([4*4*hidden*2, NUM_CLASSES])) 
            b4 = tf.Variable(tf.zeros(shape=[NUM_CLASSES]))        
            var_C = [W1,W2,W3,W4,b4] 

            global_step = tf.Variable(0, name="global_step", trainable=False)        


            C_real_logits = classifier(X, var_C)

            C_loss = tf.reduce_mean(
                           tf.nn.softmax_cross_entropy_with_logits_v2(labels = Y, 
                                                                      logits = C_real_logits))           

            num_batches_per_epoch = int((len_x_target-1)/mb_size) + 1

            C_solver = tf.contrib.opt.AdamWOptimizer(weight_decay=1e-4,learning_rate=1e-3,beta1=0.5, beta2=0.9).minimize(C_loss,var_list=var_C, global_step=global_step)


            timestamp = str(int(time.time()))    

            sess.run(tf.global_variables_initializer())

            x_temp = np.append(x_test, x_test[:mb_size], axis=0)
            y_temp = np.append(y_test, y_test[:mb_size], axis=0)
            best_accuracy = 0.0
            for it in range(num_batches_per_epoch*1000):
                X_mb, Y_mb = next_batch(mb_size, x_target, y_target)
                _, C_curr = sess.run([C_solver, C_loss], feed_dict={X: X_mb, Y: Y_mb})
                if it % 100 == 0:
                    print('Iter: {}; C_loss: {:.4};'.format(it,C_curr))
                if it %1000 == 0:    
                    predictions = []
                    for jt in range(len_x_test//mb_size+1):
                        Xt_mb, Yt_mb = next_test_batch(jt, mb_size, x_temp, y_temp)
                        _, C_pred = sess.run([C_solver, C_real_logits], feed_dict={X: Xt_mb, Y: Yt_mb})
                        if len(predictions) == 0:
                            predictions = C_pred
                        else:
                            predictions = np.append(predictions,C_pred,axis=0)
                            
                    predictions = predictions[:len_x_test]
                    predictions = np.argmax(predictions,axis=1)
                    correct_y = np.argmax(y_test,axis=1)
                    correct_predictions = sum(predictions == correct_y)
                    accuracy = correct_predictions/float(len_x_test)
                    print('Iter: {}; accuracy: {:.4}; best accuracy: {:.4}'.format(it,accuracy, best_accuracy))
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
             
            print("dataset: {} model name: {} with best accuracy: {:.4} fin.".format(dataset, model_name, best_accuracy))
            print("dataset: {} model name: {} with best accuracy: {:.4} fin.".format(dataset, model_name, best_accuracy), file = fp)
            fp.close()
            sess.close()
            
            return 

def classifier_multi():
    pathes = os.listdir("results/generated/")
    original = ["mnist","fmnist","cifar10","svhn"]
    for dataset in original:
        _, X_dim, width, height, channels,len_x_train, x_train, y_train, len_x_test, x_test, y_test  = data_loader(dataset)
        classifier_one(dataset, "original", x_train, y_train, len_x_train)
        tf.reset_default_graph() 
        
    for path in pathes:
        info = path.split('_')
        dataset = info[0]
        model_name = path
        _, X_dim, width, height, channels,len_x_train, x_train, y_train, len_x_test, x_test, y_test  = data_loader(dataset)
        data = np.load('results/generated/{}'.format(path))
        x_target = data['x']
        y_target = data['y']
        classifier_one(dataset, model_name, x_target, y_target, len_x_train)
        tf.reset_default_graph()
        
classifier_multi()
