# 4*4 SARSA
import gym
import numpy as np


# create the environment, it's not slippery
env = gym.make('FrozenLake-v0',is_slippery=False)
# action_size = 4(left,down,right,up)
action_size = env.action_space.n
# state_size = 16(0~15)
state_size = env.observation_space.n


# epsilon-greedy exploration strategy
def epsilon_greedy(qtable, epsilon, state):
    # selects a random action with probability epsilon
    n = np.random.uniform(0, 1)
    if n > epsilon:
        action = np.argmax(qtable[state])
        # choose the best(greedy) action
    else:
        action = env.action_space.sample()
        # choose a random action
    return action


# the maximum step number should be big enough
max_steps = 99


def sarsa(alpha, gamma, epsilon, n_episodes):
    # initialize Q table
    qtable = np.zeros((state_size, action_size))
    # x is a test parameter
    x = 0
    for episode in range(n_episodes):
        # initial state
        state = env.reset()
        # initial action
        action = epsilon_greedy(qtable, epsilon, state)
        # epsilon should be smaller as the episode is increasing
        if epsilon > 0:
            epsilon = epsilon - 0.0008
        for step in range(max_steps):
            # reward means step reward
            if episode == n_episodes-1:
                env.render()
            new_state, reward, done, info = env.step(action)
            # if new_state== hole(new_state == 5,7,11,12)
            if done and reward == 0 and step < max_steps:
                # negative result
                reward = -1.0
            # if the agent is going to go out of the environment
            if new_state == state:
                reward = -0.1
            #  to choose a new action according to epsilon greedy policy
            new_action = epsilon_greedy(qtable, epsilon, new_state)
            # update Q table with qtable[new state, new action]
            qtable[state, action] += alpha * (reward + (gamma * qtable[new_state, new_action]) - qtable[state, action])
            # update the state and action
            state, action = new_state, new_action
            # game over
            if done:
                # successfully reach the goal-state
                if state == 15:
                    x = x + 1
                break
    print(x)
    return qtable


def main():
    env.render()
    Q = sarsa(0.13, 0.9, 0.08, 1000)
    print(Q)

# run the main function
if __name__ == '__main__':
    main()