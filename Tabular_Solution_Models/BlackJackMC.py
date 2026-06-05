## This is a un guided learning experiment running an Monte Carlo with an epsilon softmax decision policy on simulations of black jack
##      game vs a dealer. 
## Vars:
##      Player: number of points currently
##      Usable Ace: A usable ace is when an ace is being used as an 11 rather than 1
##      Dealer: The card that is being shown's value.

import numpy as np
import gymnasium as gym
from collections import defaultdict
from tqdm import tqdm


class Agent:
    def __init__(self, env):
        self.action_size = env.action_space.n
        self.env = env
        self.Q = defaultdict(lambda: [[],[]])
    def update_Q(self, trajectory, gamma = 1.0):
        G=0
        visited = set()
        for state,action,reward in reversed(trajectory):
            G = gamma * G  + reward
            if (state,action) not in visited:
                visited.add((state,action))
                self.Q[state][action].append(G)

    def get_action(self,state, temperature=1.0):
        stick_value = np.mean(self.Q[state][0]) if self.Q[state][0] else 0.0
        hit_value   = np.mean(self.Q[state][1]) if self.Q[state][1] else 0.0
        

        values = np.array([stick_value, hit_value])
        exp_values = np.exp(values / temperature)
        probs = exp_values / exp_values.sum()

        if hit_value == 0.0 and stick_value == 0.0: probs =np.array([.5,.5])
        return np.random.choice(2, p=probs)

    def train(self, gamma = 1.0,n_episodes = 500000):
        for episode in tqdm(range(n_episodes), desc="Training"):
            state, _ = self.env.reset()
            trajectory = []
            done = False
            while not done:
                action = self.get_action(state)
                next_state,reward,terminated,truncated, _ = self.env.step(action)
                done = terminated or truncated
                trajectory.append((state,action,reward))
                state = next_state
            self.update_Q(trajectory, gamma)
    
    def evaluate(self, temperature = .01,n_episodes=10000):
        wins = draws = losses = 0
        
        for _ in tqdm(range(n_episodes), desc="evaluation"):
            state, _ = self.env.reset()
            done = False
            while not done:
                action = self.get_action(state,temperature)
                state,reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

            if reward > 0: wins+=1
            elif reward == 0: draws+=1
            else: losses +=1
        print(f"Wins: {wins/n_episodes:.1%}")
        print(f"\nDraws: {draws/n_episodes:.1%}")
        print(f"\nLosses: {losses/n_episodes:.1%}")


env = gym.make('Blackjack-v1')
agent = Agent(env)
print("Training")
agent.train()
print("\nEvaluation")
agent.evaluate()