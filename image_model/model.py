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

def ppap_autoencoder(input_shape, n_filters, filter_sizes, z_dim, x,var_A, var_G):
    current_input = x    
    encoder = []
    shapes_enc = []
    with tf.name_scope("DP_Encoder"):
        for layer_i, n_output in enumerate(n_filters[1:]):
            n_input = current_input.get_shape().as_list()[3]
            shapes_enc.append(current_input.get_shape().as_list())
            W = tf.Variable(he_normal_init([filter_sizes[layer_i],filter_sizes[layer_i],n_input, n_output]))
            var_A.append(W)
            encoder.append(W)
            conv = tf.nn.conv2d(current_input, W, strides=[1, 2, 2, 1], padding='SAME')          
            conv = tf.contrib.layers.batch_norm(conv,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
            output = tf.nn.leaky_relu(conv)
            current_input = output
        encoder.reverse()
        shapes_enc.reverse() 
        z_flat = tf.layers.flatten(current_input)
        z_flat_dim = int(z_flat.get_shape()[1])
        W_fc1 = tf.Variable(xavier_init([z_flat_dim, z_dim]))
        var_A.append(W_fc1)
        z = tf.matmul(z_flat,W_fc1)
        z = tf.contrib.layers.batch_norm(z,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)        
        z_original = z
    with tf.name_scope("DP_decoder"):        
        W_fc2 = tf.Variable(xavier_init([z_dim, z_flat_dim]))
        var_G.append(W_fc2)
        var_A.append(W_fc2)
        z_ = tf.matmul(z,W_fc2) 
        z_ = tf.contrib.layers.batch_norm(z_,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
        z_ = tf.nn.leaky_relu(z_)
        current_input = tf.reshape(z_, [-1, 4, 4, n_filters[-1]])           
        for layer_i, shape in enumerate(shapes_enc):
            W_enc = encoder[layer_i]
            W = tf.Variable(he_normal_init(W_enc.get_shape().as_list()))
            var_G.append(W)
            var_A.append(W)
            deconv = tf.nn.conv2d_transpose(current_input, W,
                                     tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                                     strides=[1, 2, 2, 1], padding='SAME')       
            if layer_i == len(n_filters)-2:
                output = tf.nn.sigmoid(deconv)
            else:
                deconv = tf.contrib.layers.batch_norm(deconv,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)           
                output = tf.nn.leaky_relu(deconv)
            current_input = output
        g = current_input
  
    return g, z_original

def edp_autoencoder(input_shape, n_filters, filter_sizes,z_dim, x, Y, var_A, var_G, init_e,sensitivity):
    current_input = x    
    encoder = []
    shapes_enc = []
    with tf.name_scope("DP_Encoder"):
        for layer_i, n_output in enumerate(n_filters[1:]):
            n_input = current_input.get_shape().as_list()[3]
            shapes_enc.append(current_input.get_shape().as_list())
            W = tf.Variable(he_normal_init([filter_sizes[layer_i],filter_sizes[layer_i],n_input, n_output]))
            var_A.append(W)
            encoder.append(W)
            conv = tf.nn.conv2d(current_input, W, strides=[1, 2, 2, 1], padding='SAME')          
            conv = tf.contrib.layers.batch_norm(conv,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
            output = tf.nn.leaky_relu(conv)
            current_input = output
        encoder.reverse()
        shapes_enc.reverse() 
        z_flat = tf.layers.flatten(current_input)
        z_flat_dim = int(z_flat.get_shape()[1])
        W_fc1 = tf.Variable(xavier_init([z_flat_dim, z_dim]))
        var_A.append(W_fc1)
        z = tf.matmul(z_flat,W_fc1)
        z = tf.contrib.layers.batch_norm(z,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
        
        
    with tf.name_scope("Noise_Applier"):      
        z_original = z
        W_epsilon = tf.Variable(epsilon_init(init_e, [z_dim]))
        epsilon_var = W_epsilon
        var_G.append(W_epsilon)     
        #W_sensitivity = tf.Variable(epsilon_init(sensitivity, [z_dim]))  
        #var_G.append(W_sensitivity)
        W_epsilon = tf.maximum(W_epsilon,1e-8)
        #dp_lambda = tf.divide(W_sensitivity,W_epsilon)
        dp_lambda = tf.divide(sensitivity,W_epsilon)
        W_noise = tf.multiply(Y,dp_lambda)
        z = tf.add(z,W_noise)
        z_noise_applied = z
        
    with tf.name_scope("DP_decoder"):        
        W_fc2 = tf.Variable(xavier_init([z_dim, z_flat_dim]))
        var_G.append(W_fc2)
        var_A.append(W_fc2)
        z_ = tf.matmul(z,W_fc2) 
        z_ = tf.contrib.layers.batch_norm(z_,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
        z_ = tf.nn.leaky_relu(z_)
        current_input = tf.reshape(z_, [-1, 4, 4, n_filters[-1]])           
        for layer_i, shape in enumerate(shapes_enc):
            W_enc = encoder[layer_i]
            W = tf.Variable(he_normal_init(W_enc.get_shape().as_list()))
            var_G.append(W)
            var_A.append(W)
            deconv = tf.nn.conv2d_transpose(current_input, W,
                                     tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                                     strides=[1, 2, 2, 1], padding='SAME')           
            if layer_i == len(n_filters)-2:
                output = tf.nn.sigmoid(deconv)
            else:
                deconv = tf.contrib.layers.batch_norm(deconv,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)           
                output = tf.nn.leaky_relu(deconv)
            current_input = output
        g = current_input
  
    return g, z_original, z_noise_applied, W_epsilon, W_noise, epsilon_var

def eddp_autoencoder(input_shape, n_filters, filter_sizes, z_dim, x, Y, var_A, var_G,init_e, init_d,sensitivity):
    current_input = x    
    encoder = []
    shapes_enc = []
    with tf.name_scope("DP_Encoder"):
        for layer_i, n_output in enumerate(n_filters[1:]):
            n_input = current_input.get_shape().as_list()[3]
            shapes_enc.append(current_input.get_shape().as_list())
            W = tf.Variable(he_normal_init([filter_sizes[layer_i],filter_sizes[layer_i],n_input, n_output]))
            var_A.append(W)
            encoder.append(W)
            conv = tf.nn.conv2d(current_input, W, strides=[1, 2, 2, 1], padding='SAME')          
            conv = tf.contrib.layers.batch_norm(conv,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
            output = tf.nn.leaky_relu(conv)
            current_input = output
        encoder.reverse()
        shapes_enc.reverse() 
        z_flat = tf.layers.flatten(current_input)
        z_flat_dim = int(z_flat.get_shape()[1])
        W_fc1 = tf.Variable(xavier_init([z_flat_dim, z_dim]))
        var_A.append(W_fc1)
        z = tf.matmul(z_flat,W_fc1)
        z = tf.contrib.layers.batch_norm(z,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
      
    with tf.name_scope("Noise_Applier"):       
        z_original = z
        W_epsilon = tf.Variable(epsilon_init(init_e, [z_dim]))
        epsilon_var = W_epsilon
        var_G.append(W_epsilon)
        W_epsilon = tf.maximum(tf.abs(W_epsilon),1e-8)       
        W_delta = tf.Variable(delta_init(init_d, [z_dim]))
        delta_var = W_delta
        var_G.append(W_delta)
        W_delta = tf.maximum(W_delta,1e-8)
        W_delta = tf.minimum(W_delta, 1.0)
        #W_sensitivity = tf.Variable(sensitivity)  
        #var_G.append(W_sensitivity)
        dp_delta = tf.log(tf.divide(1.25,W_delta))
        dp_delta = tf.maximum(dp_delta,0)
        dp_delta = tf.sqrt(tf.multiply(2.0,dp_delta))      
        #dp_lambda = tf.multiply(dp_delta,tf.divide(W_sensitivity,W_epsilon))
        dp_lambda = tf.multiply(dp_delta,tf.divide(sensitivity,W_epsilon))
        W_noise = tf.multiply(Y,dp_lambda)
        z = tf.add(z,W_noise)
        z_noise_applied = z
    with tf.name_scope("DP_decoder"):        
        W_fc2 = tf.Variable(xavier_init([z_dim, z_flat_dim]))
        var_G.append(W_fc2)
        var_A.append(W_fc2)        
        z_ = tf.matmul(z,W_fc2) 
        z_ = tf.contrib.layers.batch_norm(z_,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
        z_ = tf.nn.leaky_relu(z_)
        current_input = tf.reshape(z_, [-1, 4, 4, n_filters[-1]])           
        for layer_i, shape in enumerate(shapes_enc):
            W_enc = encoder[layer_i]
            W = tf.Variable(he_normal_init(W_enc.get_shape().as_list()))
            var_G.append(W)
            var_A.append(W)            
            deconv = tf.nn.conv2d_transpose(current_input, W,
                                     tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                                     strides=[1, 2, 2, 1], padding='SAME')           
            if layer_i == len(n_filters)-2:
                output = tf.nn.sigmoid(deconv)
            else:
                deconv = tf.contrib.layers.batch_norm(deconv,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)           
                output = tf.nn.leaky_relu(deconv)
            current_input = output
        g = current_input
 
    return g, z_original, z_noise_applied, W_epsilon, W_delta, W_noise, epsilon_var, delta_var


def discriminator(x, y, var_D):
    current_input = x
    with tf.name_scope("Discriminator"):
        for i in range(len(var_D)-2):
            conv = tf.nn.conv2d(current_input, var_D[i], strides=[1,2,2,1],padding='SAME')
            conv = tf.contrib.layers.batch_norm(conv,updates_collections=None,decay=0.9, zero_debias_moving_mean=True,is_training=True)
            current_input = tf.nn.leaky_relu(conv)            
        h = tf.layers.flatten(current_input)     
        d = tf.nn.xw_plus_b(h, var_D[-2], var_D[-1])
        loss =  tf.nn.softmax_cross_entropy_with_logits_v2(labels = y, logits = d) 
    return d
