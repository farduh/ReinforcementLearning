#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 14:55:09 2022

@author: farduh
"""

import tensorflow as tf
from tensorflow import keras


def my_loss_fn(action, probs, advantages,reg=0.01):
    entropy = - tf.reduce_sum(probs*tf.math.log(probs))
    loss = tf.math.log(probs)*advantages+reg*entropy
    loss = tf.reduce_sum(loss)    
    return loss

def create_networks(num_outputs, reg=0.01):
    
    input_ = keras.layers.Input(shape=[84,84,4])
    conv1 = keras.layers.Conv2D(
        filters=16,
        kernel_size=8,
        strides=4,
        activation=tf.nn.relu,
        )(input_)
    conv2 = keras.layers.Conv2D(
        filters=32,
        kernel_size=4,
        strides=2,
        activation=tf.nn.relu,
        )(conv1)
    flat = keras.layers.Flatten()(conv2)
    fc = keras.layers.Dense(256)(flat)
    probs = keras.layers.Dense(num_outputs,activation=tf.nn.softmax)(fc)
    value = keras.layers.Dense(1,activation=None)(fc)
    value_model = keras.Model(inputs=[input_], outputs=[value])
    policy_model = keras.Model(inputs=[input_], outputs=[probs])
    
    # policy_optimizer = tf.keras.optimizers.RMSprop(
    # learning_rate=0.00025,
    # rho=0.99,
    # momentum=0.0,
    # epsilon=1e-06,
    # name="policy_optimizer",
    # )
    # value_optimizer = tf.keras.optimizers.RMSprop(
    # learning_rate=0.00025,
    # rho=0.99,
    # momentum=0.0,
    # epsilon=1e-06,
    # name="value_optimizer",
    # )
    
    # value_model.compile(optimizer=policy_optimizer,loss=my_loss_fn)
    # value_model.compile(optimizer=value_optimizer,loss='mse')

        
    return policy_model,value_model



