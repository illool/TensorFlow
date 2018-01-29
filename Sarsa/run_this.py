# encoding=utf-8
"""
Sarsa is a online updating method for Reinforcement learning.
Unlike Q learning which is a offline updating method, Sarsa is updating while in the current trajectory.
You will see the sarsa is more coward when punishment is close because it cares about all behaviours,
while q learning is more brave because it only cares about maximum behaviour.
"""

from maze_env import Maze
from RL_brain import SarsaTable,QLearningTable


def update():
    for episode in range(100):
        # initial observation
        # 初始化环境
        observation = env.reset()

        # RL choose action based on observation
        # Sarsa 根据 state 观测选择行为
        action = RL.choose_action(str(observation))

        while True:
            # fresh env
            # 刷新环境
            env.render()
            # RL take action and get next observation and reward
            # 在环境中采取行为, 获得下一个 state_ (obervation_), reward, 和是否终止
            observation_, reward, done = env.step(action)
            # RL choose action based on next observation
            # 根据下一个 state (obervation_) 选取下一个 action_
            action_ = RL.choose_action(str(observation_))
            # RL learn from this transition (s, a, r, s, a) ==> Sarsa
            # 从 (s, a, r, s, a) 中学习, 更新 Q_tabel 的参数 ==> Sarsa
            RL.learn(str(observation), action, reward, str(observation_), action_)
            # swap observation and action
            # 将下一个当成下一步的 state (observation) and action
            observation = observation_
            action = action_
            #print(RL.q_table)
            # break while loop when end of this episode
            # 终止时跳出循环
            if done:
                break
    # end of game
    # 大循环完毕
    print('game over')
    print(RL.q_table)
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = SarsaTable(actions=list(range(env.n_actions)))
    #RL = QLearningTable(actions=list(range(env.n_actions)))
    env.after(100, update)
    env.mainloop()
