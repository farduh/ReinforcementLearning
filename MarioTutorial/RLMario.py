# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#import game
import gym_super_mario_bros
#import the Joypad e
from nes_py.wrappers import JoypadSpace
#import the SIMPLIFIED constrols
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
# import frame stacker wrapper and grayscaling wrapper
from gym.wrappers import GrayScaleObservation
#import vectorization wrappers
from stable_baselines3.common.vec_env import VecFrameStack,DummyVecEnv
#import matplotlib
import matplotlib.pyplot as plt
#1. Create base environment 
env = gym_super_mario_bros.make('SuperMarioBros-v0')
#2. Simplify controls
env = JoypadSpace(env,SIMPLE_MOVEMENT)
#3. Grayscale, keep_dim to keep array shape dimension
env = GrayScaleObservation(env,keep_dim=True)
#4. Wrap inside the dummy environment
env = DummyVecEnv([lambda : env])
#5 Stack the frame
env = VecFrameStack(env,4,channels_order='last')

# import os for file path management
import os
# Import RL model called PPO
from stable_baselines3 import PPO
# Import Base callback for saving models
from stable_baselines3.common.callbacks import BaseCallback

# import os for file path management
import os
# Import RL model called PPO
from stable_baselines3 import PPO
# Import Base callback for saving models
from stable_baselines3.common.callbacks import BaseCallback
class TrainAndLoggingCallback(BaseCallback):
    
    def __init__(self,check_freq,save_path, verbose=1):
        super(TrainAndLoggingCallback,self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
    
    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path,exist_ok=True)
    
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path,f'best_model_{self.n_calls}')
            self.model.save(model_path)
        return True

CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'

callback = TrainAndLoggingCallback(check_freq=1e5,save_path=CHECKPOINT_DIR)

#This is the RL model started
model = PPO('CnnPolicy',env,verbose=1,\
            tensorboard_log=LOG_DIR,learning_rate=0.00001,\
            n_steps=16)
#we are processing images for that reason we use cnnpoli
model.learn(total_timesteps=1e4)



model = PPO.load('train/best_model_collab_200000')
state = env.reset()
done = False
while not done:
    action, _state = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()
