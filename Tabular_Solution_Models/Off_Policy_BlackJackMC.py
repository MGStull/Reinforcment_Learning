## This is a self-guided experiment of off-policy GPI for Monte-Carlo Methods
#       for importance sampling methods. Used a stable-softmax policy for fun.
## Vars:
##      Player: number of points currently
##      Usable Ace: A usable ace is when an ace is being used as an 11 rather than 1
##      Dealer: The card that is being shown's value.

import numpy as np
import gymnasium as gym
from collections import defaultdict
from tqdm import tqdm


class Agent:
    def __init__(self, env, bp_type = None):
        self.env = env
        self.bp = self.behavior_policy()
        self.tp = self.target_policy()
        
    def train(self, gamma = 1.0,n_episodes = 500000):
        for episode in tqdm(range(n_episodes), desc="Training"):
            state, _ = self.env.reset()
            trajectory = []
            done = False
            while not done:
                action = self.bp.get_action(state)
                next_state,reward,terminated,truncated, _ = self.env.step(action)
                trajectory.append((state,action,reward))
                state = next_state
                done = terminated or truncated
            self.tp.update(trajectory,self.bp,1)
            

    def evaluate(self, temperature = .01,n_episodes=10000):
        wins = draws = losses = 0        
        for _ in tqdm(range(n_episodes), desc="evaluation"):
            state, _ = self.env.reset()
            done = False
            while not done:
                action = self.tp.get_action(state,temperature)
                state,reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

            if reward > 0: wins+=1
            elif reward == 0: draws+=1
            else: losses +=1
        print(f"Wins: {wins/n_episodes:.1%}")
        print(f"\nDraws: {draws/n_episodes:.1%}")
        print(f"\nLosses: {losses/n_episodes:.1%}")


    class behavior_policy:
        def __init__(self, type="random",custom = None):
            if "aggresive" == type:
                self.probs = [.3,.7]
            elif "safe" == type:
                self.probs = [.7,.3]
            elif "custom" == type:
                self.probs = custom
            else:
                self.probs = [.5,.5]

        def get_action(self,state):
            return np.random.choice(2, p=self.probs)
        def action_probs(self,state):
            return self.probs
        
    class target_policy:   
        def __init__(self):
            self.Q = defaultdict(lambda: np.zeros(2))
            self.C = defaultdict(lambda : np.zeros(2))
        def update(self,trajectory,bp,gamma=1.0):
            G = 0
            W = 1
            for state,action,reward in reversed(trajectory):
                G = gamma * G  + reward
                self.C[state][action] = self.C[state][action]+W
                self.Q[state][action] = self.Q[state][action] + (W/(self.C[state][action]))*(G-self.Q[state][action])
                pi_action = self.get_action(state)
                if action != pi_action:
                    break   
                mu_prob = bp.action_probs(state)[action]  
                W = W*(1/mu_prob)
        def action_probs(self, state, temperature = 1.0):
            stick_value = self.Q[state][0]
            hit_value   = self.Q[state][1]
            values = np.array([stick_value, hit_value])
            values = values / temperature
            values = values - np.max(values)  # stable softmax
            exp_values = np.exp(values)
            probs = exp_values / exp_values.sum()
            return probs
        def get_action(self,state, temperature=1.0):
            return np.random.choice(2, p=self.action_probs(state=state,temperature=temperature))


        

env = gym.make('Blackjack-v1')
agent = Agent(env,bp_type="aggresive")
print("Training")
agent.train()
print("\nEvaluation")
agent.evaluate()