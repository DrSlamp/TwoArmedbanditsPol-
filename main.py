# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 21:38:24 2023

@author: hip_d git code by Jesus P 
"""
import gym
import gym_environments
from agent import TwoArmedBandit


env = gym.make('TwoArmedBandit-v0')
agent = TwoArmedBandit(0.1) 

env.reset()

for iteration in range(100):
    action = agent.get_action("epsilon-greedy")    
    _, reward, _, _, _ = env.step(action)
    agent.update(action, reward) 
    agent.render()

print(env, "version")  
env.close()





