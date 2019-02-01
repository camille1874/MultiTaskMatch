# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import copy
from a2_base_model import BaseClass
from a2_encoder import Encoder
import os
class Transformer(BaseClass):
    def __init__(self, num_classes, learning_rate, batch_size, decay_steps, decay_rate, sequence_length,
                 vocab_size, embed_size,d_model,d_k,d_v,h,num_layer,is_training,
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

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate
        self.use_residual_conn=use_residual_conn

        self.instantiate_weights()
        self.logits = self.inference() #logits shape:[batch_size,self.num_classes]
        self.return_logtis = tf.nn.softmax(self.logits)

        self.predictions = tf.argmax(self.logits, axis=1, name="predictions")
        correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32),self.input_y_label)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")  # shape=()
        if self.is_training is False:# if it is not training, then no need to calculate loss and back-propagation.
            return
        self.loss_val = self.loss()
        self.train_op = self.train()

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

    def loss(self, l2_lambda=0.0001):  # 0.001
        with tf.name_scope("loss"):
            # input: `logits`:[batch_size, num_classes], and `labels`:[batch_size]
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y_label,logits=self.logits);  # sigmoid_cross_entropy_with_logits.#losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
            loss = tf.reduce_mean(losses) 
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if ('bias' not in v.name ) and ('alpha' not in v.name)]) * l2_lambda
            loss = loss + l2_losses
        return loss


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

