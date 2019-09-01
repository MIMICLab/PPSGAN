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

dataset = sys.argv[1]
model_name = sys.argv[2]
z_dim = int(sys.argv[3])
prev_iter = int(sys.argv[4])
NUM_CLASSES = 10
USE_DELTA = True
USE_ADAMW = True
USE_WGAN_GP = False

mb_size, X_dim, width, height, channels,len_x_train, x_train, y_train, len_x_test, x_test, y_test  = data_loader(dataset)

    
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
        Z_S = tf.placeholder(tf.float32, shape=[None, z_dim]) 
        Z_noise = tf.placeholder(tf.float32, shape=[None,  z_dim])
        Y = tf.placeholder(tf.float32, shape=[None,  NUM_CLASSES])
        A_true_flat = X        
        #generator variables
        var_G = []
        #discriminator variables
        W1 = tf.Variable(he_normal_init([5,5,channels, hidden//2]))
        W2 = tf.Variable(he_normal_init([5,5, hidden//2,hidden]))
        W3 = tf.Variable(he_normal_init([5,5,hidden,hidden*2]))
        W4 = tf.Variable(xavier_init([4*4*hidden*2, 1]))
        b4 = tf.Variable(tf.zeros(shape=[1]))        
        var_D = [W1,W2,W3,W4,b4] 
        
        #classifier variables
        W4_c = tf.Variable(xavier_init([4*4*hidden*2, NUM_CLASSES])) 
        b4_c = tf.Variable(tf.zeros(shape=[NUM_CLASSES]))        
        var_C = [W1,W2,W3,W4_c,b4_c] 
        
        var_D_C = [W1,W2,W3,W4,b4,W4_c,b4_c]
        
        global_step = tf.Variable(0, name="global_step", trainable=False)        
       
        G_sample, G_zero, z_original, z_noise, z_noised = generator(input_shape, 
                                                                   n_filters, 
                                                                   filter_sizes, 
                                                                   X, 
                                                                   Z_noise, 
                                                                   var_G,
                                                                   z_dim,
                                                                   Z_S,
                                                                   USE_DELTA)               
        if USE_WGAN_GP:
            D_real_logits = discriminator_gp(X, var_D)
            D_fake_logits = discriminator_gp(G_sample, var_D)
            C_real_logits = classifier_gp(X, var_C)
            C_fake_logits = classifier_gp(G_sample, var_C)
             
            gp = gradient_penalty(G_sample, X, mb_size, var_D)
            D_S_loss = tf.reduce_mean(D_fake_logits) - tf.reduce_mean(D_real_logits) + 10.0*gp 
            D_C_loss = tf.reduce_mean(
                       tf.nn.softmax_cross_entropy_with_logits_v2(labels = Y, 
                                                                  logits = C_real_logits))
            D_loss = D_S_loss + D_C_loss
        
            G_zero_loss = tf.reduce_mean(tf.pow(X - G_zero,2))         
            G_S_loss = - tf.reduce_mean(D_fake_logits)
            G_C_loss = tf.reduce_mean(
                       tf.nn.softmax_cross_entropy_with_logits_v2(labels = Y, 
                                                                  logits = C_fake_logits))
            G_loss = G_S_loss + G_C_loss + G_zero_loss
            
        else:
            D_real, D_real_logits = discriminator(X, var_D)
            D_fake, D_fake_logits = discriminator(G_sample, var_D)
            C_real_logits = classifier(X, var_C)
            C_fake_logits = classifier(G_sample, var_C)
            
            D_real_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=D_real_logits, labels=tf.ones_like(D_real))
            D_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                               logits=D_fake_logits, labels=tf.zeros_like(D_fake))
            
            D_S_loss = tf.reduce_mean(D_real_loss) + tf.reduce_mean(D_fake_loss)
            D_C_loss = tf.reduce_mean(
                       tf.nn.softmax_cross_entropy_with_logits_v2(labels = Y, 
                                                                  logits = C_real_logits))
            
            D_loss = D_S_loss + D_C_loss
        
            G_zero_loss = tf.reduce_mean(tf.pow(X - G_zero,2))         
            G_S_loss = tf.reduce_mean(
                       tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, 
                                                               labels=tf.ones_like(D_fake)))
            G_C_loss = tf.reduce_mean(
                       tf.nn.softmax_cross_entropy_with_logits_v2(labels = Y, 
                                                                  logits = C_fake_logits))
            G_loss = G_S_loss + G_C_loss + G_zero_loss       
        latent_max = tf.reduce_max(z_original, axis = 0)
        latent_min = tf.reduce_min(z_original, axis = 0)          
        
        tf.summary.image('Original',X)
        tf.summary.image('fake',G_sample) 
        tf.summary.image('fake_zero', G_zero)
        
        tf.summary.scalar('D_loss', D_loss)  
        tf.summary.scalar('D_S_loss',D_S_loss) 
        tf.summary.scalar('D_C_loss',D_C_loss)
        tf.summary.scalar('G_zero_loss',G_zero_loss)        
        tf.summary.scalar('G_S_loss',G_S_loss)
        tf.summary.scalar('G_C_loss',G_C_loss)        
        tf.summary.scalar('G_loss',G_loss) 
        
        tf.summary.histogram('z_original',z_original) 
        tf.summary.histogram('z_noise',z_noise) 
        tf.summary.histogram('z_noised',z_noised)
        
        merged = tf.summary.merge_all()

        num_batches_per_epoch = int((len_x_train-1)/mb_size) + 1
        
        if USE_ADAMW:
            A_solver = tf.contrib.opt.AdamWOptimizer(weight_decay=1e-4,learning_rate=1e-4,beta1=0.5, beta2=0.9).minimize(G_zero_loss,var_list=var_G, global_step=global_step)       
            D_solver = tf.contrib.opt.AdamWOptimizer(weight_decay=1e-4,learning_rate=1e-4,beta1=0.5, beta2=0.9).minimize(D_loss,var_list=var_D_C, global_step=global_step)
            G_solver = tf.contrib.opt.AdamWOptimizer(weight_decay=1e-4,learning_rate=1e-4,beta1=0.5, beta2=0.9).minimize(G_loss,var_list=var_G, global_step=global_step)
        
        else:
            A_solver = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.5, beta2=0.9).minimize(G_zero_loss,var_list=var_G, global_step=global_step)       
            D_solver = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.5, beta2=0.9).minimize(D_loss,var_list=var_D_C, global_step=global_step)
            G_solver = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.5, beta2=0.9).minimize(G_loss,var_list=var_G, global_step=global_step)
        timestamp = str(int(time.time()))
        if not os.path.exists('results/'):
            os.makedirs('results/')        
        out_dir = os.path.abspath(os.path.join(os.path.curdir, 
                                               "results/models/{}_".format(dataset) + model_name))
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists('results/models/'):
            os.makedirs('results/models/')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not os.path.exists('results/dc_out_{}_{}/'.format(dataset,model_name)):
            os.makedirs('results/dc_out_{}_{}/'.format(dataset,model_name))           

        train_writer = tf.summary.FileWriter('results/graphs/{}_{}'.format(dataset,model_name),sess.graph)
        saver = tf.train.Saver(tf.global_variables())
        sess.run(tf.global_variables_initializer())
        
        if prev_iter != 0:
            saver.restore(sess,tf.train.latest_checkpoint(checkpoint_dir))        
        i = prev_iter 
        
        #Autoencoder pre-train
        if prev_iter == 0:
            for idx in range(num_batches_per_epoch*100):
                if dataset == 'mnist':
                    X_mb, Y_mb = x_train.train.next_batch(mb_size)
                    X_mb = np.reshape(X_mb,[-1,28,28,1])  
                else:
                    X_mb = next_batch(mb_size, x_train)
                    Y_mb = next_batch(mb_size, y_train)
                    
                enc_zero = np.zeros([mb_size,z_dim]).astype(np.float32)
                if USE_DELTA:
                    enc_noise = np.random.normal(0.0,1.0,[mb_size,z_dim]).astype(np.float32)  
                else:
                    enc_noise = np.random.laplace(0.0,1.0,[mb_size,z_dim]).astype(np.float32)  
                summary,_, A_loss_curr= sess.run([merged, A_solver, G_zero_loss],
                                                        feed_dict={X: X_mb, 
                                                                   Y: Y_mb, 
                                                                   Z_noise: enc_noise, 
                                                                   Z_S: enc_zero}) 
                
                current_step = tf.train.global_step(sess, global_step)
                train_writer.add_summary(summary,current_step)
                if idx % 100 == 0:
                    print('Iter: {}; G_zero_loss: {:.4};'.format(idx,A_loss_curr))
                if idx % 1000 == 0: 
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print('Saved model at {} at step {}'.format(path, current_step)) 
                    
        #calculate approximated global sensitivity            
        for idx in range(num_batches_per_epoch):
            if dataset == 'mnist':
                X_mb, _ = x_train.train.next_batch(mb_size)
                X_mb = np.reshape(X_mb,[-1,28,28,1])                   
            else:
                X_mb = next_batch(mb_size, x_train)
                Y_mb = next_batch(mb_size, y_train)
            enc_zero = np.zeros([mb_size,z_dim]).astype(np.float32) 
            if USE_DELTA:
                enc_noise = np.random.normal(0.0,1.0,[mb_size,z_dim]).astype(np.float32)  
            else:
                enc_noise = np.random.laplace(0.0,1.0,[mb_size,z_dim]).astype(np.float32)                  
            max_curr, min_curr = sess.run([latent_max,latent_min], feed_dict={
                                                                   X: X_mb, 
                                                                   Y: Y_mb, 
                                                                   Z_noise: enc_noise, 
                                                                   Z_S: enc_zero}) 
            if idx == 0:
                z_max = max_curr
                z_min = min_curr
            else:
                z_max = np.maximum(z_max,max_curr)
                z_min = np.minimum(z_min,min_curr)
        z_sensitivity = np.abs(np.subtract(z_max,z_min))
        print("Approximated Global Sensitivity:") 
        print(z_sensitivity)        
        z_sensitivity = np.tile(z_sensitivity,(mb_size,1)) 
        
        #Adversarial training           
        for it in range(num_batches_per_epoch*10000):
            for _ in range(5):
                if dataset == 'mnist':
                    X_mb, Y_mb = x_train.train.next_batch(mb_size)
                    X_mb = np.reshape(X_mb,[-1,28,28,1])
                else:
                    X_mb = next_batch(mb_size, x_train)
                    Y_mb = next_batch(mb_size, y_train)
                
                enc_zero = np.zeros([mb_size,z_dim]).astype(np.float32) 
                if USE_DELTA:
                    enc_noise = np.random.normal(0.0,1.0,[mb_size,z_dim]).astype(np.float32)  
                else:
                    enc_noise = np.random.laplace(0.0,1.0,[mb_size,z_dim]).astype(np.float32)  
                _, D_curr, D_S_curr, D_C_curr = sess.run([D_solver, D_loss, D_S_loss, D_C_loss],
                                                         feed_dict={X: X_mb, 
                                                                   Y: Y_mb, 
                                                                   Z_noise: enc_noise, 
                                                                   Z_S: z_sensitivity})              
            summary, _, G_curr, G_S_curr, G_C_curr, G_z_curr = sess.run([merged, G_solver,
                                                                         G_loss,G_S_loss, G_C_loss,
                                                                         G_zero_loss],
                                                        feed_dict={X: X_mb, 
                                                                   Y: Y_mb,  
                                                                   Z_noise: enc_noise,
                                                                   Z_S: z_sensitivity}) 
            current_step = tf.train.global_step(sess, global_step)
            train_writer.add_summary(summary,current_step)
        
            if it % 100 == 0:
                print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}; D_S: {:.4}; G_S: {:.4}; D_C: {:.4}; G_C: {:.4}; G_zero: {:.4};'.format(it,D_curr, G_curr, D_S_curr, G_S_curr, D_C_curr, G_C_curr, G_z_curr))

            if it % 1000 == 0:   
                Xt_mb = x_test[:mb_size]
                Yt_mb = y_test[:mb_size] 
                enc_zero = np.zeros([mb_size,z_dim]).astype(np.float32)  
                if USE_DELTA:
                    enc_noise = np.random.normal(0.0,1.0,[mb_size,z_dim]).astype(np.float32)  
                else:
                    enc_noise = np.random.laplace(0.0,1.0,[mb_size,z_dim]).astype(np.float32) 
                G_sample_curr, re_fake_curr = sess.run([G_sample, G_zero],
                                                       feed_dict={X: Xt_mb, 
                                                                  Y: Yt_mb, 
                                                                  Z_noise: enc_noise, 
                                                                  Z_S: z_sensitivity}) 
                
                samples_flat = tf.reshape(G_sample_curr,[-1,width,height,channels]).eval()
                img_set = np.append(Xt_mb[:mb_size], samples_flat[:mb_size], axis=0)         
                samples_flat = tf.reshape(re_fake_curr,[-1,width,height,channels]).eval() 
                img_set = np.append(img_set, samples_flat[:mb_size], axis=0)
                fig = plot(img_set, width, height, channels)
                plt.savefig('results/dc_out_{}_{}/{}.png'.format(dataset,model_name,str(i).zfill(3)), bbox_inches='tight')
                plt.close(fig)
                i += 1
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print('Saved model at {} at step {}'.format(path, current_step))
                
                #calculate approximated global sensitivity            
                for idx in range(num_batches_per_epoch):
                    if dataset == 'mnist':
                        X_mb, _ = x_train.train.next_batch(mb_size)
                        X_mb = np.reshape(X_mb,[-1,28,28,1])
                    elif dataset == 'lsun':
                        X_mb = x_train.next_batch(mb_size)                    
                    else:
                        X_mb = next_batch(mb_size, x_train) 
                    if USE_DELTA:
                        enc_noise = np.random.normal(0.0,1.0,[mb_size,z_dim]).astype(np.float32)  
                    else:
                        enc_noise = np.random.laplace(0.0,1.0,[mb_size,z_dim]).astype(np.float32)                   
                    max_curr, min_curr = sess.run([latent_max,latent_min], feed_dict={
                                                                   X: X_mb, 
                                                                   Y: Y_mb, 
                                                                   Z_noise: enc_noise, 
                                                                   Z_S: enc_zero}) 
                    if idx == 0:
                        z_max = max_curr
                        z_min = min_curr
                    else:
                        z_max = np.maximum(z_max,max_curr)
                        z_min = np.minimum(z_min,min_curr)
                z_sensitivity = np.abs(np.subtract(z_max,z_min))
                print("Approximated Global Sensitivity:") 
                print(z_sensitivity)        
                z_sensitivity = np.tile(z_sensitivity,(mb_size,1)) 