import gym
import sys
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools
import shutil
import threading
import multiprocessing

from networks import create_networks
from worker import Worker


ENV_NAME = "Breakout-v0"
MAX_GLOBAL_STEPS =  5e4
STEPS_PER_UPDATE = 5

def Env():
  return gym.envs.make(ENV_NAME)



NUM_WORKERS = multiprocessing.cpu_count()

with gym.envs.make(ENV_NAME) as env:
    env.action_space.n
    policy_net, value_net = create_networks(env.action_space.n)
del env
#counter
global_counter = itertools.count()
print(policy_net.get_weights()[-1])

returns_list = []
workers = []

for worker_id in range(NUM_WORKERS):
    worker = Worker(
        name="worker_{}".format(worker_id),
         env=Env(),
         policy_net=policy_net,
         value_net=value_net,
         global_counter=global_counter,
         returns_list=returns_list,
         discount_factor = 0.99,
         max_global_steps=MAX_GLOBAL_STEPS
        )
    
    workers.append(worker)

coord = tf.train.Coordinator()
worker_threads = []

for worker in workers:
    t = threading.Thread(target = worker.run, args=(coord,STEPS_PER_UPDATE))
    t.start()
    worker_threads.append(t)

coord.join(worker_threads,stop_grace_period_secs=300)

plt.plot(returns_list)

policy_net.get_weights()[-1]
