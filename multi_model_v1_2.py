# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import copy
from base_model import BaseClass
from encoder import Encoder
import os
class Transformer(BaseClass):
    def __init__(self, num_classes, learning_rate, batch_size, decay_steps, decay_rate, sequence_length,
                 vocab_size, embed_size,d_model,d_k,d_v,h,num_layer,is_training,num_features, di=50, s=40, w=4,l2_reg=0.0004,
                 initializer=tf.random_normal_initializer(stddev=0.1),clip_gradients=5.0,l2_lambda=0.0001,use_residual_conn=False):
        """init all hyperparameter here"""
        super(Transformer, self).__init__(d_model, d_k, d_v, sequence_length, h, batch_size, num_layer=num_layer) #init some fields by using parent class.

        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = d_model
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * 0.7)
        self.initializer = initializer
        self.clip_gradients=clip_gradients
        self.l2_lambda=l2_lambda

        self.is_training=is_training #self.is_training=tf.placeholder(tf.bool,name="is_training") #tf.bool #is_training
        self.input_x = tf.placeholder(tf.int32, [self.batch_size, self.sequence_length], name="input_x")                 #x  batch_size
        self.input_y_label = tf.placeholder(tf.int32, [self.batch_size], name="input_y_label")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        

        self.x1 = tf.placeholder(tf.float32, shape=[None, d_model, s], name="x1")
        self.x2 = tf.placeholder(tf.float32, shape=[None, d_model, s], name="x2")
        self.features = tf.placeholder(tf.float32, shape=[None, num_features], name="features")
        self.di = di
        self.s = s
        self.w = w      
        self.l2_reg =l2_reg


        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate
        self.use_residual_conn=use_residual_conn

        self.instantiate_weights()
        self.logits = self.inference()
        #self.logits = tf.layers.dense(tf.concat([self.logits, self.get_compare_logits()], 1), units=self.num_classes) #logits shape:[batch_size,self.num_classes]
        #self.return_logtis = tf.nn.softmax(self.logits)
        #self.predictions = tf.argmax(self.logits, axis=1, name="predictions")
        self.compare_logits = self.get_compare_logits()
        #self.return_logits = tf.nn.softmax(self.logits) + tf.nn.softmax(self.compare_logits)
        self.return_logits = tf.nn.softmax(self.compare_logits)
        self.predictions = tf.argmax(self.return_logits, axis=1, name="predictions")

        correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32),self.input_y_label)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")  # shape=()
        
 
        if self.is_training is False:# if it is not training, then no need to calculate loss and back-propagation.
            return
        self.loss_val = self.loss() + self.compare_loss()
        self.train_op = self.train()

    def get_compare_logits(self):  
        def pad_for_wide_conv(x):
            return tf.pad(x, np.array([[0, 0], [0, 0], [self.w - 1, self.w - 1], [0, 0]]), "CONSTANT", name="pad_wide_conv")

        def cos_sim(v1, v2):
            norm1 = tf.sqrt(tf.reduce_sum(tf.square(v1), axis=1))
            norm2 = tf.sqrt(tf.reduce_sum(tf.square(v2), axis=1))
            dot_products = tf.reduce_sum(v1 * v2, axis=1, name="cos_sim")
 
            return dot_products / (norm1 * norm2)
  
        def euclidean_score(v1, v2):
            euclidean = tf.sqrt(tf.reduce_sum(tf.square(v1 - v2), axis=1))
            return 1 / (1 + euclidean)

        def make_attention_mat(x1, x2):
            # x1, x2 = [batch, height, width, 1] = [batch, d, s, 1]
            # x2 => [batch, height, 1, width]
            # [batch, width, wdith] = [batch, s, s]
            euclidean = tf.sqrt(tf.reduce_sum(tf.square(x1 - tf.matrix_transpose(x2)), axis=1))
            return 1 / (1 + euclidean)

        def convolution(name_scope, x, d, reuse):
            with tf.name_scope(name_scope + "-conv"):
                with tf.variable_scope("conv") as scope:
                    conv = tf.contrib.layers.conv2d(
                        inputs=x,
                        num_outputs=self.di,
                        kernel_size=(d, self.w),
                        stride=1,
                        padding="VALID",
                        activation_fn=tf.nn.tanh,
                        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=self.l2_reg),
                        biases_initializer=tf.constant_initializer(1e-04),
                        reuse=reuse,
                        trainable=True,
                        scope=scope
                    )
                    conv_trans = tf.transpose(conv, [0, 3, 2, 1], name="conv_trans")
                    return conv_trans

        def w_pool(variable_scope, x, attention):
            # x: [batch, di, s+w-1, 1]
            # attention: [batch, s+w-1]
            with tf.variable_scope(variable_scope + "-w_pool"):
                pools = []
                # [batch, s+w-1] => [batch, 1, s+w-1, 1]
                attention = tf.transpose(tf.expand_dims(tf.expand_dims(attention, -1), -1), [0, 2, 1, 3])
  
                for i in range(self.s):
                        # [batch, di, w, 1], [batch, 1, w, 1] => [batch, di, 1,     1]
                        pools.append(tf.reduce_sum(x[:, :, i:i + self.w, :] * attention[:, :, i:i + self.w, :], axis=2, keep_dims=True))
    
                # [batch, di, s, 1]
                w_ap = tf.concat(pools, axis=2, name="w_ap")
                return w_ap


        def all_pool(variable_scope, x):
            with tf.variable_scope(variable_scope + "-all_pool"):
                if variable_scope.startswith("input"):
                    pool_width = self.s
                    d = self.embed_size
                else:
                    pool_width = self.s + self.w - 1
                    d = self.di
 
                all_ap = tf.layers.average_pooling2d(
                    inputs=x,
                    # (pool_height, pool_width)
                    pool_size=(1, pool_width),
                    strides=1,
                    padding="VALID",
                    name="all_ap"
                )
                # [batch, di, 1, 1]
 
                # [batch, di]
                all_ap_reshaped = tf.reshape(all_ap, [-1, d])
                #all_ap_reshaped = tf.squeeze(all_ap, [2, 3])
                return all_ap_reshaped



        def CNN_layer(variable_scope, x1, x2, d):
            # x1, x2 = [batch, d, s, 1]
            with tf.variable_scope(variable_scope):
                left_conv = convolution(name_scope="left", x=pad_for_wide_conv(x1), d=d, reuse=False)
                right_conv = convolution(name_scope="right", x=pad_for_wide_conv(x2), d=d, reuse=True)

                left_attention, right_attention = None, None
 
                # [batch, s+w-1, s+w-1]
                att_mat = make_attention_mat(left_conv, right_conv)
                # [batch, s+w-1], [batch, s+w-1]
                left_attention, right_attention = tf.reduce_sum(att_mat, axis=2), tf.reduce_sum(att_mat, axis=1)

                left_wp = w_pool(variable_scope="left", x=left_conv, attention=left_attention)
                left_ap = all_pool(variable_scope="left", x=left_conv)
                right_wp = w_pool(variable_scope="right", x=right_conv, attention=right_attention)
                right_ap = all_pool(variable_scope="right", x=right_conv)
 
                return left_wp, left_ap, right_wp, right_ap


        x1_expanded = tf.expand_dims(self.x1, -1)
        x2_expanded = tf.expand_dims(self.x2, -1)
        LO_0 = all_pool(variable_scope="input-left", x=x1_expanded)
        RO_0 = all_pool(variable_scope="input-right", x=x2_expanded)
        LI_1, LO_1, RI_1, RO_1 = CNN_layer(variable_scope="CNN-1", x1=x1_expanded, x2=x2_expanded, d=self.embed_size)
        sims = [cos_sim(LO_0, RO_0), cos_sim(LO_1, RO_1)] 

        estimation = tf.layers.dense(tf.stack(sims, axis=1), units=self.num_classes)
        return estimation
        #output_features = tf.concat([self.features, tf.stack(sims, axis=1)], axis=1, name="output_features")
        #return output_features

    def inference(self):
        input_x_embeded = tf.nn.embedding_lookup(self.Embedding,self.input_x)  #[None,sequence_length, embed_size]
        input_x_embeded=tf.multiply(input_x_embeded,tf.sqrt(tf.cast(self.d_model,dtype=tf.float32)))
        input_mask=tf.get_variable("input_mask",[self.sequence_length,1],initializer=self.initializer)
        input_x_embeded=tf.add(input_x_embeded,input_mask) 

        encoder_class=Encoder(self.d_model,self.d_k,self.d_v,self.sequence_length,self.h,self.batch_size,self.num_layer,input_x_embeded,input_x_embeded,dropout_keep_prob=self.dropout_keep_prob,use_residual_conn=self.use_residual_conn)
        Q_encoded,K_encoded = encoder_class.encoder_fn() #K_v_encoder

        Q_encoded=tf.reshape(Q_encoded,shape=(self.batch_size,-1)) #[batch_size,sequence_length*d_model]
        with tf.variable_scope("output"):
            logits = tf.matmul(Q_encoded, self.W_projection) + self.b_projection #logits shape:[batch_size*decoder_sent_length,self.num_classes]
        print("logits:",logits)
        return logits

    def loss(self, l2_lambda=0.0001, a=0.3):  # 0.001
        with tf.name_scope("loss"):
            # input: `logits`:[batch_size, num_classes], and `labels`:[batch_size]
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y_label,logits=self.logits);  # sigmoid_cross_entropy_with_logits.#losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
            loss = tf.reduce_mean(losses) 
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if ('bias' not in v.name ) and ('alpha' not in v.name)]) * l2_lambda
            #loss = a * loss + l2_losses + (1 - a) * self.get_compare_loss()
            loss = loss + l2_losses
        return loss

    def compare_loss(self):
        with tf.name_scope("compare_loss"):
            losses = tf.add(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.compare_logits, labels=self.input_y_label)), tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))) 
        return losses

    def train(self):
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
        self.learning_rate_=learning_rate
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
                                                   learning_rate=learning_rate, optimizer="Adam",clip_gradients=self.clip_gradients)
        return train_op

    def instantiate_weights(self):
        with tf.device('/cpu:0'), tf.variable_scope("embedding_projection"):
            self.Embedding = tf.Variable(tf.constant(0.0, shape=[self.vocab_size, self.embed_size]),trainable=False, name="Embedding")
            self.embedding_placeholder = tf.placeholder(tf.float32, shape=[self.vocab_size, self.embed_size])
            self.embedding_init = self.Embedding.assign(self.embedding_placeholder)
            self.W_projection = tf.get_variable("W_projection", shape=[self.sequence_length*self.d_model, self.num_classes],initializer=self.initializer)  # [embed_size,label_size]
            self.b_projection = tf.get_variable("b_projection", shape=[self.num_classes])

    def get_mask(self,sequence_length):
        lower_triangle = tf.matrix_band_part(tf.ones([sequence_length, sequence_length]), -1, 0)
        result = -1e9 * (1.0 - lower_triangle)
        print("get_mask==>result:", result)
        return result

