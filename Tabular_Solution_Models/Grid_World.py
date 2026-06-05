import numpy as np


class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.reset()
    
    def reset(self):
        self.agent_pos = [np.random.randint(0, self.size), np.random.randint(0, self.size)]
        self.goal_pos = [19,8]
        return self.get_state()

    def step(self, action):
        if action == 0:
            self.agent_pos[0] = max(self.agent_pos[0]-1,0)
        elif action == 1:
            self.agent_pos[0] = min(self.agent_pos[0]+1,self.size-1)
        elif action == 2:
            self.agent_pos[1] = max(self.agent_pos[1]-1,0)
        elif action == 3:
            self.agent_pos[1] = min(self.agent_pos[1]+1,self.size-1)

        done = (self.agent_pos == self.goal_pos)
        reward = 1.0 if done else -0.01
        return self.get_state(), reward, done

    def get_state(self):
        state = np.zeros((self.size, self.size))
        state[self.agent_pos[0]][self.agent_pos[1]] = 1
        state[self.goal_pos[0]][self.goal_pos[1]] = 2
        return state
    
    def render(self):
        symbols = {0: '.', 1: 'A', 2: 'G'}
        for i in range(self.size):
            row = ''
            for j in range(self.size):
                if [i,j] == self.agent_pos:
                    row += 'A '
                elif [i,j] == self.goal_pos:
                    row += 'G '
                else:
                    row += '. '
            print(row)
        print()


def policy_evaluation(env, policy, gamma=0.9, theta=1e-6):
    V = np.zeros((env.size, env.size))

    while True: 
        delta = 0
        for i in range(env.size):
            for j in range(env.size):
                if [i,j] == env.goal_pos:
                    continue
                v = V[i][j]
                new_v = 0

                for action in range(4):
                    
                    #Get next state and reward for taking action in state (i,j)
                    env.agent_pos = [i,j]
                    next_state, reward, done = env.step(action)
                    next_i, next_j = env.agent_pos

                    #Bellman update
                    new_v += policy[action] * (reward + gamma * V[next_i][next_j])
                V[i][j] = new_v
                delta = max(delta, abs(v - new_v))

        if delta < theta:
            break
    return V


env = GridWorld(size=20)
state = env.reset()
total_reward = 0
steps = 0



for _ in range(200):
    action = np.random.randint(0, 4)
    state, reward, done = env.step(action)
    total_reward += reward
    steps += 1
    env.render()
    
    if done:
        print(f"Goal reached in {steps} steps")
        print(f"Total reward: {total_reward:.2f}")
        break
else:
    print(f"Did not reach goal in 200 steps")
    print(f"Total reward: {total_reward:.2f}")


def extract_policy(env, V, gamma=0.9):
    policy = np.zeros((env.size, env.size), dtype=int)
    action_symbols = {0:'↑', 1:'↓', 2:'←', 3:'→'}
    
    for i in range(env.size):
        for j in range(env.size):
            if [i,j] == env.goal_pos:
                continue
            
            action_values = []
            
            for action in range(4):
                env.agent_pos = [i,j]
                next_state, reward, done = env.step(action)
                next_i, next_j = env.agent_pos
                
                # Value of taking this action
                q = reward + gamma * V[next_i][next_j]
                action_values.append(q)
            
            # Best action is the one leading to highest value state
            policy[i][j] = np.argmax(action_values)
    
    return policy

def render_policy(policy):
    symbols = {0:'↑', 1:'↓', 2:'←', 3:'→'}
    for i in range(policy.shape[0]):
        row = ''
        for j in range(policy.shape[1]):
            row += symbols[policy[i][j]] + '  '
        print(row)
V = policy_evaluation(env, policy=np.ones(4)/4)  # Uniform random policy
policy = extract_policy(env, V)
render_policy(policy)
print("Optimal policy (arrows indicate best action from each state)")


def policy_iteration(env, gamma=0.9, theta=1e-6):
    # Start with uniform random policy as 2D array
    # matching the shape extract_policy returns
    policy = np.ones((env.size, env.size), dtype=int)  
    # initialize all actions to 1 (down) as starting point
    
    iteration = 0
    while True:
        # Step 1 — evaluate current policy
        # Convert 2D policy to per-state action probabilities
        def deterministic_policy(action, state_policy):
            return 1.0 if action == state_policy else 0.0
        
        V = policy_evaluation(env, [0.25, 0.25, 0.25, 0.25], gamma, theta)
        
        # Step 2 — improve policy greedily
        new_policy = extract_policy(env, V, gamma)
        
        iteration += 1
        print(f"Iteration {iteration}")
        render_policy(new_policy)
        
        # Step 3 — check if policy changed
        if np.array_equal(new_policy, policy):
            print(f"Converged after {iteration} iterations")
            break
            
        policy = new_policy
    
    return policy, V

optimal_policy, optimal_value = policy_iteration(env)
print("Optimal policy found by policy iteration:")