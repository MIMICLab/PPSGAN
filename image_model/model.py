import tensorflow as tf
import numpy as np

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def he_normal_init(size):
    in_dim = size[0]
    he_stddev = tf.sqrt(2./in_dim)
    return tf.random_normal(shape=size, stddev=he_stddev)

def epsilon_init(initial, size):
    in_dim = size[0]
    stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.add(initial,tf.random_normal(shape=size, stddev=stddev))

def delta_init(initial, size):
    in_dim = size[0]
    stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.add(initial,tf.random_normal(shape=size, stddev=stddev))

def generator(input_shape, n_filters, filter_sizes, z_dim, x, noise, var_G, reuse=False):
    idx=0
    current_input = x    
    encoder = []
    shapes_enc = []
    with tf.name_scope("Encoder"):
        for layer_i, n_output in enumerate(n_filters[1:]):
            n_input = current_input.get_shape().as_list()[3]
            shapes_enc.append(current_input.get_shape().as_list())
            if reuse:
                W = var_G[idx]
                idx += 1
            else:
                W = tf.Variable(xavier_init([filter_sizes[layer_i],
                                            filter_sizes[layer_i],
                                            n_input, n_output]))
                var_G.append(W)
            encoder.append(W)
            conv = tf.nn.conv2d(current_input, W, strides=[1, 2, 2, 1], padding='SAME')          
            conv = tf.contrib.layers.batch_norm(conv,
                                                updates_collections=None,
                                                decay=0.9,
                                                zero_debias_moving_mean=True,
                                                is_training=True)
            output = tf.nn.leaky_relu(conv)
            current_input = output
        encoder.reverse()
        shapes_enc.reverse() 
        z_flat = tf.layers.flatten(current_input)
        z_flat_dim = int(z_flat.get_shape()[1])
        if reuse:
            W_fc1 = var_G[idx]
            idx += 1
        else:
            W_fc1 = tf.Variable(xavier_init([z_flat_dim, z_dim]))
            var_G.append(W_fc1)
        z = tf.matmul(z_flat,W_fc1)
        z = tf.contrib.layers.batch_norm(z,
                                         updates_collections=None,
                                         decay=0.9,
                                         zero_debias_moving_mean=True,
                                         is_training=True)
        z_original = z
        
    with tf.name_scope("Noise_Applier"):   
        z_noise = noise
        for i in range(3):
            if reuse:
                WN = var_G[idx]
                idx += 1
            else:
                WN = tf.Variable(xavier_init([z_dim, z_dim]))
                var_G.append(WN)
            z_noise = tf.matmul(z_noise, WN)
            z = tf.contrib.layers.batch_norm(z,
                                             updates_collections=None,
                                             decay=0.9,
                                             zero_debias_moving_mean=True,
                                             is_training=True)
            z= tf.nn.leaky_relu(z)            
        z = tf.add(z, z_noise)
        z_noised = z
        
    with tf.name_scope("Decoder"):  
        if reuse:
            W_fc2 = var_G[idx]
            idx += 1
        else:
            W_fc2 = tf.Variable(xavier_init([z_dim, z_flat_dim]))
            var_G.append(W_fc2)
        z_ = tf.matmul(z,W_fc2) 
        z_ = tf.contrib.layers.batch_norm(z_,
                                          updates_collections=None,
                                          decay=0.9, 
                                          zero_debias_moving_mean=True,
                                          is_training=True)
        z_ = tf.nn.leaky_relu(z_)
        current_input = tf.reshape(z_, [-1, 4, 4, n_filters[-1]])           
        for layer_i, shape in enumerate(shapes_enc):
            W_enc = encoder[layer_i]
            if reuse == True:
                W = var_G[idx]
                idx += 1
            else:
                W = tf.Variable(he_normal_init(W_enc.get_shape().as_list()))
                var_G.append(W)
            deconv = tf.nn.conv2d_transpose(current_input, 
                                            W,
                                            tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                                            strides=[1, 2, 2, 1], 
                                            padding='SAME') 
            
            if layer_i == len(n_filters)-2:
                output = tf.nn.sigmoid(deconv)
            else:
                deconv = tf.contrib.layers.batch_norm(deconv,
                                                      updates_collections=None,
                                                      decay=0.9,
                                                      zero_debias_moving_mean=True,
                                                      is_training=True)           
                output = tf.nn.leaky_relu(deconv)
            current_input = output
        g = current_input
  
    return g, z_original, z_noise, z_noised

def discriminator(x,var_D):
    current_input = x
    with tf.name_scope("Discriminator"):
        for i in range(len(var_D)-2):
            conv = tf.nn.conv2d(current_input, var_D[i], strides=[1,2,2,1],padding='SAME')
            conv = tf.contrib.layers.batch_norm(conv,
                                                updates_collections=None,
                                                decay=0.9,
                                                zero_debias_moving_mean=True,
                                                is_training=True)
            current_input = tf.nn.leaky_relu(conv)            
        h = tf.layers.flatten(current_input)     
        d = tf.nn.xw_plus_b(h, var_D[-2], var_D[-1])
        
    return d

def classifier(x, y, var_C):
    current_input = x
    with tf.name_scope("Discriminator"):
        for i in range(len(var_C)-2):
            conv = tf.nn.conv2d(current_input, var_C[i], strides=[1,2,2,1],padding='SAME')
            conv = tf.contrib.layers.batch_norm(conv,
                                                updates_collections=None,
                                                decay=0.9,
                                                zero_debias_moving_mean=True,
                                                is_training=True)
            current_input = tf.nn.leaky_relu(conv)            
        h = tf.layers.flatten(current_input)     
        d = tf.nn.xw_plus_b(h, var_C[-2], var_C[-1])
        loss =  tf.nn.softmax_cross_entropy_with_logits_v2(labels = y, logits = d) 
    return d

def gradient_penalty(G_sample, A_true_flat, mb_size, var_D):
    epsilon = tf.random_uniform(shape=[mb_size, 1, 1, 1], minval=0.,maxval=1.)
    X_hat = A_true_flat + epsilon * (G_sample - A_true_flat)
    D_X_hat = discriminator(X_hat,var_D)
    grad_D_X_hat = tf.gradients(D_X_hat, [X_hat])[0]
    red_idx = list(range(1, X_hat.shape.ndims))
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_D_X_hat), reduction_indices=red_idx))
    gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.))
    return gradient_penalty


