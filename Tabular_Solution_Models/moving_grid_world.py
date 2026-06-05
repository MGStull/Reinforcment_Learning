import numpy as np

class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.reset()
    
    def reset(self):
        self.agent_pos = [np.random.randint(0, self.size), np.random.randint(0, self.size)]
        self.goal_pos = [1,3]
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
        if done: 
            reward = 10.0
        else:
            distance = np.sqrt((self.agent_pos[0]-self.goal_pos[0])**2 + (self.agent_pos[1]-self.goal_pos[1])**2)
            reward = -distance/self.size
        self.goal_post_move()
        return self.get_state(), reward, done

    def goal_post_move(self):
        random_action = np.random.randint(0,4)
        if random_action == 0:
            self.goal_pos[0] = max(self.goal_pos[0]-1,0)
        elif random_action == 1:
            self.goal_pos[0] = min(self.goal_pos[0]+1,self.size-1)
        elif random_action == 2:
            self.goal_pos[1] = max(self.goal_pos[1]-1,0)
        elif random_action == 3:
            self.goal_pos[1] = min(self.goal_pos[1]+1,self.size-1)
        
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

class Policy:
    def __init__(self, env, epsilon=0.1, gamma=0.9):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        # Value function — starts at zeros
        self.V = np.zeros((env.size, env.size, env.size, env.size))
        # Policy grid — stores best action per state
        self.policy_grid = np.zeros(
            (env.size, env.size, env.size, env.size), 
            dtype=int
        )
    
    def action_probabilities(self, agent_pos, goal_pos):
        """
        Returns probability distribution over actions
        using epsilon-greedy over current value function
        """
        ai, aj = agent_pos
        gi, gj = goal_pos
        
        # Get Q value for each action
        action_values = []
        for action in range(4):
            # Simulate agent movement
            next_ai, next_aj = self._simulate_action(ai, aj, action)
            
            # Expected value over random goal movements
            expected_value = 0
            for goal_action in range(4):
                next_gi, next_gj = self._simulate_action(gi, gj, goal_action)
                expected_value += 0.25 * self.V[next_ai, next_aj, next_gi, next_gj]
            
            # Distance reward
            if [next_ai, next_aj] == [gi, gj]:
                q = 10.0
            else:
                distance = np.sqrt((next_ai-gi)**2 + (next_aj-gj)**2)
                reward = -distance / self.env.size
                q = reward + self.gamma * expected_value
            
            action_values.append(q)
        
        # Epsilon greedy probabilities
        probs = np.ones(4) * self.epsilon / 4
        best_action = np.argmax(action_values)
        probs[best_action] += (1 - self.epsilon)
        
        return probs, action_values
    
    def select_action(self, agent_pos, goal_pos):
        """Sample action from probability distribution"""
        probs, _ = self.action_probabilities(agent_pos, goal_pos)
        return np.random.choice(4, p=probs)
    
    def evaluate(self, theta=1e-6):
        """
        Policy evaluation — update value function
        to reflect current policy
        """
        size = self.env.size
        
        while True:
            delta = 0
            
            for ai in range(size):
                for aj in range(size):
                    for gi in range(size):
                        for gj in range(size):
                            if [ai,aj] == [gi,gj]:
                                continue
                            
                            old_v = self.V[ai, aj, gi, gj]
                            
                            # Get action probabilities from current policy
                            probs, _ = self.action_probabilities(
                                [ai,aj], [gi,gj]
                            )
                            
                            new_v = 0
                            for action in range(4):
                                next_ai, next_aj = self._simulate_action(
                                    ai, aj, action
                                )
                                
                                # Average over random goal movements
                                for goal_action in range(4):
                                    next_gi, next_gj = self._simulate_action(
                                        gi, gj, goal_action
                                    )
                                    
                                    if [next_ai,next_aj] == [gi,gj]:
                                        reward = 10.0
                                        done = True
                                    else:
                                        distance = np.sqrt(
                                            (next_ai-gi)**2 + 
                                            (next_aj-gj)**2
                                        )
                                        reward = -distance / self.env.size
                                        done = False
                                    
                                    future = 0 if done else self.V[
                                        next_ai, next_aj, 
                                        next_gi, next_gj
                                    ]
                                    
                                    new_v += (probs[action] * 0.25 * 
                                             (reward + self.gamma * future))
                            
                            self.V[ai, aj, gi, gj] = new_v
                            delta = max(delta, abs(old_v - new_v))
            
            if delta < theta:
                break
        
        return self.V
    
    def improve(self):
        """
        Policy improvement — update policy to be
        greedy with respect to current value function
        """
        size = self.env.size
        policy_stable = True
        
        for ai in range(size):
            for aj in range(size):
                for gi in range(size):
                    for gj in range(size):
                        if [ai,aj] == [gi,gj]:
                            continue
                        
                        old_action = self.policy_grid[ai,aj,gi,gj]
                        _, action_values = self.action_probabilities(
                            [ai,aj], [gi,gj]
                        )
                        new_action = np.argmax(action_values)
                        
                        self.policy_grid[ai,aj,gi,gj] = new_action
                        
                        if old_action != new_action:
                            policy_stable = False
        
        return policy_stable
    
    def iterate(self, max_iterations=100):
        """
        Full policy iteration loop —
        evaluate then improve until convergence
        """
        for i in range(max_iterations):
            print(f"Policy iteration {i+1}")
            self.evaluate()
            stable = self.improve()
            
            if stable:
                print(f"Converged after {i+1} iterations")
                break
        
        return self.policy_grid
    
    def render(self, goal_pos):
        """Show policy arrows for a fixed goal position"""
        symbols = {0:'↑', 1:'↓', 2:'←', 3:'→'}
        gi, gj = goal_pos
        print(f"\nPolicy with goal at {goal_pos}:")
        for ai in range(self.env.size):
            row = ''
            for aj in range(self.env.size):
                if [ai,aj] == [gi,gj]:
                    row += 'G  '
                else:
                    row += symbols[self.policy_grid[ai,aj,gi,gj]] + '  '
            print(row)
    
    def _simulate_action(self, i, j, action):
        """Helper — returns next position given action"""
        size = self.env.size
        ni, nj = i, j
        if action == 0: ni = max(i-1, 0)
        elif action == 1: ni = min(i+1, size-1)
        elif action == 2: nj = max(j-1, 0)
        elif action == 3: nj = min(j+1, size-1)
        return ni, nj



env = GridWorld(size=10)
policy = Policy(env, epsilon=0.1, gamma=0.9)

print("Running policy iteration...")
policy.iterate(max_iterations=50)

# Show policy for different goal positions
policy.render([4,4])
policy.render([0,4])
policy.render([2,2])

# Run an episode
state = env.reset()
total_reward = 0
for step in range(200):
    action = policy.select_action(env.agent_pos, env.goal_pos)
    state, reward, done = env.step(action)
    total_reward += reward
    env.render()
    if done:
        print(f"Caught in {step+1} steps, reward: {total_reward:.2f}")
        break