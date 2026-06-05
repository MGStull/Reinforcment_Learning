import gymnasium as gym

env_name = "CartPole-v1"
env = gym.make(env_name,render_mode="human")
env.reset()




class Agent():
    def __init__(self,env):
        self.action_size = env.action_space.n
        print("Action size:", self.action_size)

    def get_action(self, obs):
        pole_angle = obs[2]
        action = 0 if pole_angle < 0 else 1
        return action


agent = Agent(env)
obs,info = env.reset()
print(obs)
for _ in range(1000):
    action = agent.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    print(obs)
    if terminated or truncated:
        obs, info = env.reset()
env.close()