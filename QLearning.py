# encoding=utf-8
"""
A simple example for Reinforcement Learning using table lookup Q-learning method.
An agent "o" is on the left of a 1 dimensional world, the treasure is on the rightmost location.
Run this program and to see how the agent will improve its strategy of finding the treasure.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd
import time

np.random.seed(2)  # reproducible


N_STATES = 10   # the length of the 1 dimensional world
ACTIONS = ['left', 'right']     # available actions
EPSILON = 0.9   # greedy police
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # discount factor  奖励递减值
MAX_EPISODES = 130   # maximum episodes
FRESH_TIME = 0.3    # fresh time for one move 移动间隔时间


def build_q_table(n_states, actions):
        #       left     right
        #  0  0.119902  0.403451
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     # q_table initial values
        columns=actions,    # actions's name
    )
    # print(table)    # show table
    return table


def choose_action(state, q_table):
    # This is how to choose an action
    # 选出这个 state 的所有 action 值
    state_actions = q_table.iloc[state, :]
     # 非贪婪 or 或者这个 state 还没有探索过
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):  # act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS)
    else:   # act greedy    
        # 贪婪模式
        action_name = state_actions.idxmax()    # replace argmax to idxmax as argmax means a different function in newer version of pandas
    return action_name


def get_env_feedback(S, A):
    # This is how agent will interact with the environment
    if A == 'right':    # move right
        if S == N_STATES - 2:   # terminate
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:   # move left
        R = 0
        if S == 0:
            S_ = S  # reach the wall
        else:
            S_ = S - 1
    return S_, R


def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    # main part of RL loop
    q_table = build_q_table(N_STATES, ACTIONS)# 初始 q table
    for episode in range(MAX_EPISODES): # 回合
        step_counter = 0
        S = 0 # 回合初始位置
        is_terminated = False # 回合初始位置
        update_env(S, episode, step_counter) # 环境更新
        while not is_terminated:

            A = choose_action(S, q_table) # 选行为
            # 实施行为并得到环境的反馈
            S_, R = get_env_feedback(S, A)  # take action & get next state and reward
            q_predict = q_table.loc[S, A]  # 估算的(状态-行为)值     
            if S_ != 'terminal':
                #  实际的(状态-行为)值 (回合没结束)
                q_target = R + GAMMA * q_table.iloc[S_, :].max()   # next state is not terminal
            else:
                #  实际的(状态-行为)值 (回合结束)
                q_target = R     # next state is terminal
                is_terminated = True    # terminate this episode
            #  q_table 更新
            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # update
             # 探索者移动到下一个 state
            S = S_  # move to next state
            # 环境更新
            update_env(S, episode, step_counter+1)
            step_counter += 1
    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
print(q_table)
