# 4*4 Q-learning
import numpy as np
import gym


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


def q_learning(alpha, gamma, epsilon, n_episodes):
    # initialize Q table
    qtable = np.zeros((state_size, action_size))
    # x is a test parameter
    x = 0

    # episode loop
    for episode in range(n_episodes):
        # initial state
        state = env.reset()
        # epsilon should be smaller as the episode is increasing
        if epsilon > 0:
            epsilon = epsilon - 0.0008
        # step loop
        for step in range(max_steps):
            # render the road in the last episode
            if episode == n_episodes-1:
                env.render()
            # choose action according to epsilon greedy policy
            action = epsilon_greedy(qtable, epsilon, state)
            # take the action
            new_state, reward, done, info = env.step(action)

            # change the reward
            # if new_state== hole(new_state == 5,7,11,12)
            if done and reward == 0 and step < max_steps:
                # negative result
                reward = -1.0
            # if the agent is going to go out of the environment
            if new_state == state:
                reward = -0.1
            # update Q table using max_{a'}(qtable)
            qtable[state, action] = qtable[state, action] + alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state, action])
            # update the state
            state = new_state
            # game over
            if done:
                # if reach the goal-state
                if state == 15:
                    x = x + 1
                break
    # print the success number
    print (x)
    return qtable

# Play the game
def main():
    env.render()
    # 1000 is enough for 4 by 4 problem
    Q = q_learning(0.2, 0.9, 0.17, 1000)
    print(Q)

# run the main function
if __name__ == '__main__':
    main()