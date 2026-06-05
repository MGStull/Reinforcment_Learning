import numpy as np
import matplotlib.pyplot as plt

class EpsilonGreedyBandit:
    def __init__(self, k=10, epsilon = 0.1):
        self.k = k
        self.epsilon = epsilon
        self.q_values = np.zeros(k)
        self.action_counts = np.zeros(k)

        self.true_values = np.random.normal(0,1,k)

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.k)
        else:
            return np.argmax(self.q_values)
        
    def update(self, action, reward):
        self.action_counts[action] += 1
        alpha = 1 / self.action_counts[action]
        self.q_values[action] += alpha * (reward - self.q_values[action])

    def pull(self, action):
        return self.true_values[action] + np.random.normal(0,1)


def run_experiment(k = 10, epsilon=0.1, n_steps=2000):
    bandit = EpsilonGreedyBandit(k=k, epsilon=epsilon)
    
    rewards = []
    optimal_selections = []
    optimal_arm = np.argmax(bandit.true_values)
    
    for step in range(n_steps):
        action = bandit.select_action()
        reward = bandit.pull(action)
        bandit.update(action, reward)
        
        rewards.append(reward)
        optimal_selections.append(1 if action == optimal_arm else 0)
    
    # Rolling average to smooth noise
    window = 50
    smoothed_rewards = np.convolve(
        rewards, 
        np.ones(window)/window, 
        mode='valid'
    )
    smoothed_optimal = np.convolve(
        optimal_selections,
        np.ones(window)/window,
        mode='valid'
    )
    
    return smoothed_rewards, smoothed_optimal

# Compare three epsilon values
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

for eps in [0.0, 0.1, 0.01]:
    rewards, optimal = run_experiment(epsilon=eps)
    ax1.plot(rewards, label=f'ε={eps}')
    ax2.plot(optimal, label=f'ε={eps}')

ax1.set_title('Average Reward Over Time')
ax1.set_xlabel('Steps')
ax1.set_ylabel('Average Reward')
ax1.legend()

ax2.set_title('Optimal Action Selection Rate')
ax2.set_xlabel('Steps')
ax2.set_ylabel('% Optimal Action')
ax2.legend()

plt.tight_layout()
plt.show()
