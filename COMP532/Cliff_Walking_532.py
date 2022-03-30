"""
@Date: 28/02/2022

@Author: Adeola Adebayo, Princewill Okosun, Osabouhein Ogbebor

@Description: Cliff walking problem using SARSA and Q-Learning from Sutton RL - COMP 532 CA-1

"""

import numpy as np
import matplotlib.pyplot as plt


def create_qtable(x=4, y=12):
    return np.zeros((4, x * y))


def move_agent(loc, action):
    x, y = loc

    if action == 0 and x > 0:
        x = x - 1

    if action == 1 and y > 0:
        y = y - 1

    if action == 2 and y < 11:
        y = y + 1

    if action == 3 and x < 3:
        x = x + 1

    return x, y


def update_grid(loc, grid):
    x, y = loc
    grid[x][y] = 1
    return grid


def get_state(loc, qtable):
    x, y = loc

    state = 12 * x + y

    state_action_values = qtable[:, int(state)]
    _max = np.amax(state_action_values)
    return state, _max


def get_reward(state):
    reward, ep_state = -1, False
    if state == 47:
        reward = 10
        ep_state = True
    if 37 <= state <= 46:
        reward = -100
        ep_state = True
    return reward, ep_state


def update_qtable(qtable, state, action, reward, n_state, discount=1, a=0.1):
    qtable[action, state] = qtable[action, state] + a * (reward + (discount * n_state - qtable[action, state]))
    return qtable


def e_greedy_policy(state, qtable, e=0.01):
    random_value = np.random.random_sample()
    if e > random_value:
        return np.random.choice(4)
    return np.argmax(qtable[:, state])


def q_learning(epochs=500, discount=1, a=0.1, epsilon=0.1):
    qtable = create_qtable()
    rewards = list()
    steps = list()
    for episode in range(epochs):
        agent = (3, 0)
        grid = np.zeros((4, 12))
        grid = update_grid(agent, grid)
        episode_completed = False
        sum_rewards = 0
        sum_steps = 0

        # choose action A derived from greedy policy and observe R, S' -- updates using max state instead of next state
        while not episode_completed:
            state, _ = get_state(agent, qtable)
            action = e_greedy_policy(state, qtable, epsilon)
            agent = move_agent(agent, action)
            grid = update_grid(agent, grid)
            next_state, max_state = get_state(agent, qtable)
            reward, episode_completed = get_reward(next_state)

            sum_rewards = sum_rewards + reward
            sum_steps = sum_steps + 1

            qtable = update_qtable(qtable, state, action, reward, max_state, discount, a)

        rewards.append(sum_rewards)
        steps.append(sum_steps)
        if episode == 499:
            print('final epoch qlearning grid')
    return qtable, rewards, steps


def sarsa(epochs=500, discount=1, a=0.1, epsilon=0.1):
    qtable = create_qtable()
    rewards = list()
    steps = list()

    for episode in range(epochs):
        agent = (3, 0)
        grid = np.zeros((4, 12))
        grid = update_grid(agent, grid)
        episode_completed = False
        sum_rewards = 0
        sum_steps = 0

        # choose action A using greedy policy and observe R, S', then choose action A' from S' using greedy policy
        state, _ = get_state(agent, qtable)
        action = e_greedy_policy(state, qtable, epsilon)

        while not episode_completed:
            agent = move_agent(agent, action)
            grid = update_grid(agent, grid)

            next_state, _ = get_state(agent, qtable)
            reward, episode_completed = get_reward(next_state)
            sum_rewards = sum_rewards + reward

            next_action = e_greedy_policy(next_state, qtable, epsilon)
            next_state_value = qtable[next_action][next_state]  # next e greedy action
            qtable = update_qtable(qtable, state, action, reward, next_state_value, discount, a)
            state = next_state
            action = next_action

        rewards.append(sum_rewards)
        steps.append(sum_steps)
        # print("s_epoch - " + str(episode + 1))
        if episode == 499:
            print('final epoch sarsa grid')
    return qtable, rewards, steps


def get_grid_values(qtable, action):
    print(qtable[action, :].reshape((4, 12)))


def plot_norm_rewards(s_rewards, q_rewards, epsilon=0.1):
    s_avg_rewards = get_average_rewards(s_rewards)
    q_avg_rewards = get_average_rewards(q_rewards)

    plt.plot(q_avg_rewards, label="Q_learning")
    plt.plot(s_avg_rewards, label="SARSA")
    plt.ylabel('Cumulative Rewards')
    plt.xlabel('Episodes (batch size 10)')
    plt.title("Q-Learning/SARSA Convergence of Cumulative Reward - ɛ = " + str(epsilon))
    plt.legend(loc='lower right', ncol=2, mode="expand", borderaxespad=0.)
    plt.show()


def get_average_rewards(rewards):
    avg_rewards = list()

    i = 0
    sum_reward = 0
    for reward in rewards:
        i = i + 1
        sum_reward = sum_reward + reward
        if i == 10:
            avg_reward = sum_reward / 10
            avg_rewards.append(avg_reward)
            sum_reward = 0
            i = 0

    return avg_rewards


def get_results(epsilon=0.1):
    sarsa_qtable, sarsa_rewards, sarsa_steps = sarsa(500, 1, 0.1, epsilon)
    qlearning_qtable, qlearning_rewards, qlearning_steps = q_learning(500, 1, 0.1, epsilon)

    plot_norm_rewards(sarsa_rewards, qlearning_rewards, epsilon)


if __name__ == '__main__':
    # get_results(0)
    # get_results(0.01)
    get_results(0.1)
    # get_results(0.5)
