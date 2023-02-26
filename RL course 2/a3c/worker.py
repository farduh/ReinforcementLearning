#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 12:00:44 2022

@author: farduh
"""

import gym
import sys
import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from networks import create_networks

class Step:
  def __init__(self, state, action, reward, next_state, done):
    self.state = state
    self.action = action
    self.reward = reward
    self.next_state = next_state
    self.done = done

def transform_frame(frame):
    frame = tf.image.rgb_to_grayscale(frame)
    frame = tf.image.crop_to_bounding_box(frame, 34, 0, 160, 160)
    frame = tf.image.resize(frame, [84,84],)
    frame = tf.squeeze(frame)
    return frame

def get_copy_params_op(model, target_model):
  target_model.set_weights(model.get_weights())
  return target_model

# Create initial state by repeating the same frame 4 times
def repeat_frame(frame):
  return np.stack([frame] * 4, axis=2)

def shift_frames(state, next_frame):
  return np.append(state[:,:,1:], np.expand_dims(next_frame, 2), axis=2)

    
class Worker:
    def __init__(
        self,
        name,
        env,
        policy_net,
        value_net,
        global_counter,
        returns_list,
        max_global_steps=None,
        discount_factor=0.99
        ):
    
        self.name = name
        self.env = env
        self.global_policy_net = policy_net
        self.global_value_net = value_net
        self.global_counter = global_counter
        self.discount_factor = discount_factor
        self.max_global_steps=max_global_steps
        self.policy_net, self.value_net = create_networks(policy_net.output_shape[1])
        
        self.state = None # Keep track of the current state
        self.total_reward = 0. # After each episode print the total (sum of) reward
        self.returns_list = returns_list # Global returns list to plot later
        
        
    def run(self,coord,max_n_step):

        frame = self.env.reset()
        frame = transform_frame(frame)
        self.state = repeat_frame(frame)
        try :
            while not coord.should_stop():
                get_copy_params_op(self.global_policy_net, self.policy_net)
                get_copy_params_op(self.global_value_net, self.value_net)
                    
                steps, global_step = self.run_n_steps(max_n_step)
                if global_step >= self.max_global_steps:
                    coord.request_stop()
                    return
                
                self.update(steps)
            
        except tf.errors.CancelledError:
            return
            
    def run_n_steps(self,max_n_steps):
        steps = []
        for _ in range(max_n_steps):
            
            action = self.sample_action(self.state)
            next_frame, reward, done, _ = self.env.step(action)
            next_frame = transform_frame(next_frame)
            next_state = shift_frames(self.state,next_frame)
            
            step = Step(self.state,action,reward,next_state,done)
            steps.append(step)
            
            #increase global counter
            global_step = next(self.global_counter)
            
            if done:
                print('global step: ',global_step,'total_reward: ',self.total_reward)
                self.returns_list.append(self.total_reward)
                self.total_reward = 0
                break
            else:
                self.total_reward += reward 
            self.state = next_state
        
        return steps, global_step
    
    def update(self,steps):
        
        future_rewards = 0 
        if not steps[-1].done:
            future_rewards = self.get_value_prediction(steps[-1].next_state)

        states = []
        advantages = []
        value_targets = []        
        actions = []
        
        for step in reversed(steps):
            future_rewards = step.reward + self.discount_factor * future_rewards
            advantage = future_rewards - self.get_value_prediction(step.state)
            
            states.append(step.state)
            actions.append(step.action)
            value_targets.append(future_rewards)
            advantages.append(advantage)
            
        self.train_op(states,actions,value_targets,advantages)
                
    def sample_action(self, state):
        state = tf.expand_dims(state,0)
        probs = self.policy_net(state)
        distribution = tfp.distributions.Categorical(probs=probs)
        return distribution.sample()[0]
    
    def get_value_prediction(self, state):
        state = tf.expand_dims(state,0)
        pred = self.value_net(state)
        return pred[0]
        
    def train_op(self,states,actions,value_targets,advantages):
       
        policy_optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=0.00025,
        rho=0.99,
        momentum=0.0,
        epsilon=1e-06,
        name="policy_optimizer",
        )
        
        with tf.GradientTape() as tape:
            probs = self.policy_net(np.array(states))
            entropy = - tf.reduce_sum(probs*tf.math.log(probs))
            if np.isnan(entropy):
                entropy = 0
            batch_size = tf.shape(states)[0]
            gather_indices = tf.range(batch_size) * tf.shape(probs)[1] + actions
            selected_action_probs = tf.gather(tf.reshape(probs, [-1]), gather_indices)
            loss = tf.math.log(selected_action_probs)*advantages+0.01*entropy
            loss = tf.reduce_sum(loss)    

        grads = tape.gradient(loss, self.policy_net.trainable_variables)
        policy_optimizer.apply_gradients(zip(grads, self.global_policy_net.trainable_variables))
        
        value_optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=0.00025,
        rho=0.99,
        momentum=0.0,
        epsilon=1e-06,
        name="value_optimizer",
        )
          
        with tf.GradientTape() as tape:
            v_pred = self.value_net(np.array(states))
            loss = tf.losses.mean_squared_error(value_targets,v_pred)
            loss = tf.reduce_mean(loss)
            
        grads = tape.gradient(loss, self.value_net.trainable_variables)
        value_optimizer.apply_gradients(zip(grads, self.global_value_net.trainable_variables))
        
        
        
        