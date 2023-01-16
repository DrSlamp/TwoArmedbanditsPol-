# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 21:36:22 2023

@author: Pol git code by Jesus P 
"""
from contextlib import nullcontext
import numpy as np

class TwoArmedBandit():
    def __init__(self, alpha=1, epsilon =0.1):
        self.arms = 2
        self.alpha = alpha
        self.epsilon = epsilon
        self.reset()
        

    def reset(self):
        self.action = 0
        self.reward = 0
        self.iteration = 0
        self.values = np.zeros(self.arms)
        self.sum = 0

    def update(self, action, reward):
        self.action = action
        self.reward = reward
        self.iteration += 1
        self.values[action] = self.values[action] + self.alpha * (reward - self.values[action])
        
        self.sum += self.reward 
         



    def get_action(self, mode):
        if mode == 'random':
            return np.random.choice(self.arms)
        elif mode == 'greedy':
            return np.argmax(self.values)
        elif mode == 'epsilon-greedy':  
            if np.random.rand() < self.epsilon:
                return np.random.randint(self.arms)
            else:
                return np.argmax(self.values)


              
            



    def render(self):
        print("Iteration: {}, Action: {}, Reward: {}, Values: {}, TOTAL_REWARD: {} ".format(
            self.iteration, self.action, self.reward, self.values, self.sum))