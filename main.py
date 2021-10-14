# -*- coding: utf-8 -*-
import gym 
import matplotlib.pyplot as plt
# import matplotlib.style 
import numpy as np 
# import sys 
from QLearning import qLearning  
  
from collections import defaultdict 


# objective is to get the cart to the flag.
# for now, let's just move randomly:

env = gym.make("MountainCar-v0")


QTABLE_SIZE = [30, 30]
STATE_STEP = (env.observation_space.high - env.observation_space.low)/np.array(QTABLE_SIZE)


def get_state(state):
    state = (state - env.observation_space.low)/STATE_STEP
    return tuple(state.astype(np.int))

ql = qLearning(env, get_state, qtable_size=QTABLE_SIZE )

ql.fit(8000)

env.close()

plt.figure()
plt.plot(ql.stats["episode"], ql.stats["reward"]["avg"])