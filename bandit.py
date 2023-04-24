import numpy as np

class Bandit:
    def _init_(self, n_arms):
        self.n_arms = n_arms
        self.action_values = np.random.normal(0, 1, self.n_arms)
        
    def get_reward(self, action):
        return np.random.normal(self.action_values[action], 1)
        
class EpsilonGreedy:
    def _init_(self, epsilon, n_arms):
        self.epsilon = epsilon
        self.n_arms = n_arms
        self.Q = np.zeros(self.n_arms)
        self.N = np.zeros(self.n_arms)
        
    def select_action(self):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_arms)
        else:
            action = np.argmax(self.Q)
        return action
    
    def update(self, action, reward):
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]
        
def run_bandit(n_arms):
    bandit = Bandit(n_arms)
    agent = EpsilonGreedy(0.1, n_arms)
    total_reward = 0
    for i in range(1000):
        action = agent.select_action()
        reward = bandit.get_reward(action)
        agent.update(action, reward)
        total_reward += reward
    return total_reward

n_arms_list = [8, 10]
for n_arms in n_arms_list:
    rewards = []
    for i in range(2000):
        reward = run_bandit(n_arms)
        rewards.append(reward)
    print("Average reward for {}-armed bandit: {:.2f}".format(n_arms, np.mean(rewards)))