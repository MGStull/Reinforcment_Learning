## This is a un guided learning experiment running an Monte Carlo with an epsilon greedy decision policy on simulations of black jack
##      game vs a dealer. 
## Vars:
##      Player: number of points currently
##      Usable Ace: A usable ace is when an ace is being used as an 11 rather than 1
##      Dealer: The card that is being shown's value.

import numpy as np
import gymnasium as gym
from collections import defaultdict


env = gym.make('Blackjack-v1')

def generate_episode(env, epsilon=0.1):
    trajectory = []
    state,_ = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        next_state,reward,terminated,truncated, _ = env.step(action)
        done = terminated or truncated

        trajectory.append((state,action,reward))
        state = next_state
    return trajectory

class Agent:
    def __init__(self, env):
        self.action_size = env.action_space.n
        self.env = env
        self.Q = defaultdict(lambda: [[],[]])
    def update_Q(self, trajectory, gamma = 1.0):
        G=0
        visited = set()
        _,_,reward = trajectory[-1]
        for state,action in trajectory[0:1]:
            visited.add(state)
            self.Q[state][action].append(reward)

    def get_action(self,state):
        reward_mean_hit = sum(self.Q[state][0])/len(self.Q[state][0])
        reward_mean_stick = sum(self.Q[state][1])/len(self.Q[state][1])
        

trajectory = generate_episode(env)
print(trajectory[0])