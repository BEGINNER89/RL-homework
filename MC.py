# 4*4 grids FirstVisit MentoCarloControl without exploring starts
import gym
import numpy as np
import random

# create the environment, it's not slippery
env = gym.make('FrozenLake-v0',is_slippery=False)
# action_size = 4(left,down,right,up)
action_size = env.action_space.n
# state_size = 16(0~15)
state_size = env.observation_space.n


def create_policy():
    # Initialize  policy {state:p[action]}
    policy = {}
    for key in range(0, state_size):  # state
        p = {}
        for action in range(0, action_size):
            # action: posibility is 0.25
            p[action] = 1 / action_size
        policy[key] = p
    return policy


def create_Q(policy):
    # Initialize Q(s,a)
    Q = {}
    for key in policy.keys():
        # set Q(s,a) as 0
        Q[key] = {a: 0.0 for a in range(0, action_size)}
    return Q


def generate_episode(env, policy):
    # generate an episode
    env.reset()
    episode = []
    # game is not over
    done = False
    while not done:
        state = env.s # state
        # timestep contains S0,A0,R1,S1,A1,R2,...,S_{T-1},A_{T-1},R_{T}
        timestep = []
        # for example, S_i
        timestep.append(state)
        n = np.random.uniform(0, 1)
        choose_action = 0
        # choose action
        # p[0] is action, p[1] is probability
        for p in policy[state].items():
            # this is to realize choosing action with given probability
            choose_action += p[1]
            # n is a random number so the probability to choose the best action is its policy value
            if n < choose_action:
                action = p[0]
                break
        new_state, reward, done, info = env.step(action)
        # change the reward
        if reward == 0 and done:
            reward = -1
        if new_state == state:
            reward = -0.1

        # for example, A_i
        timestep.append(action)
        # for example, R_{i+1}
        timestep.append(reward)
        # for example, S_i,A_i,R_{i+1}
        episode.append(timestep)

    return episode


def monte_carlo(env, episodes=1000, gamma=0.9, epsilon=0.01):  
    # Create an empty dictionary to store state action values
    pi = create_policy()
    # Empty Q for storing rewards for each state-action pair
    Q = create_Q(pi)
    # Empty return
    my_return = {}
    # y is a test parameter
    y = 0
    for j in range(episodes):  # Loop for each episode
        # initialize reward =0
        G = 0
        # the episode with steps following the policy
        episode = generate_episode(env=env, policy=pi)
        # Store state, action and value respectively

        # loop for each step: i=T-1,T-2,...,1,0
        for i in reversed(range(0, len(episode))):
            # get s, a and r from the ith step
            s_t, a_t, r_t = episode[i]
            if r_t == 1:
                # the success times
                y = y + 1
            # create state action pair
            s_a = (s_t, a_t)
            # set gamma = 0.9 and update G
            G = gamma * G + r_t

            # unless pair(St,At)appears in S0,A0,S1,A1,...,S(T-1),A(T-1) (this is first-visit meaning)
            if s_a not in [(x[0], x[1]) for x in episode[0:i]]:
                # if s_a exists
                if my_return.get(s_a):
                    # update the returns[s_a] to G
                    my_return[s_a].append(G)
                # if not:
                else:
                    # create returns[s_a], whose value is G
                    my_return[s_a] = [G]
                # Average return (St,At)
                Q[s_t][a_t] = sum(my_return[s_a]) / len(my_return[s_a])
                # Finding the action with maximum value
                all_a = list(map(lambda x: x[1], Q[s_t].items()))
                # x[1]=a; Q_list means Q(St,a)

                # find the best a , maybe the best a is not the only one
                best_a = [i for i, x in enumerate(all_a) if x == max(all_a)]
                # choose  best action from indices
                action = random.choice(best_a)

                # Update action probability for s_t in policy(epsilon soft policy)
                for a in pi[s_t].items():
                    if a[0] == action:
                        # in this environment the |A(s)| should be 4
                        pi[s_t][a[0]] = 1 - epsilon + epsilon / 4.0
                    else:
                        pi[s_t][a[0]] = epsilon / 4.0
    return pi,y


def main():
    env.render()
    policy,y = monte_carlo(env=env, episodes=1000,gamma = 0.9,epsilon = 0.01)
    print(policy,y)


# run the main function
if __name__ == '__main__':
    main()