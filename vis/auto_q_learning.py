#coding=utf8
#!/usr/bin/env python\
import numpy as np

from q_learning_model import QLearningModel
from q_learning_visualization import QLearningVisual

map = [[1, 0, 3, 0, 3, 0, 0, 0, 0, 0],
       [0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 3, 0, 0, 0, 3, 0, 2, 0],
       [0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 3, 0, 0, 3, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 3, 0, 0, 0, 3, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 3, 0, 0],
       [0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

COLLISION = 0
OUT_OF_TABLE = 1
REACH_GOAL = 2
OK_TO_GO = 3

REWARD_COLLISION = -3
REWARD_MOVE = -1
REWARD_GOAL = 1
REWARD_NOT_READY = 0


class QLearningTest:
    """This class is for testing the Q_learning method"""

    def __init__(self):
        """initialize QLearningModel and transfer a list to parameter @actions in __init__
                0: left, 1: up, 2: right, 3: down"""

        self.learning_model = QLearningModel([0, 1, 2, 3])
        self.iteration_num = 60
        self.iterations = []
        self.performances = []
        self.q_learning(self.learning_model, self.iteration_num, self.iterations, self.performances)

    def action_to_goal(self, _action, curr_state):
        """get target state of the current action"""

        target = [0, 0]
        if _action == 0:    #left
            target[0] = curr_state[0]
            target[1] = curr_state[1] - 1
        if _action == 1:    #up
            target[0] = curr_state[0] - 1
            target[1] = curr_state[1]
        if _action == 2:    #right
            target[0] = curr_state[0]
            target[1] = curr_state[1] + 1
        if _action == 3:    #down
            target[0] = curr_state[0] + 1
            target[1] = curr_state[1]
        return target

    def check_goal(self, target_x, target_y, dst_pos):
        """check the result of the goal"""

        action_result = OK_TO_GO

        if target_x < 0 or target_y < 0:
            action_result = OUT_OF_TABLE
            return action_result

        if target_x >= 10 or target_y >= 10:
            action_result = OUT_OF_TABLE
            return action_result

        if target_x == dst_pos[0] and target_y == dst_pos[1]:
            action_result = REACH_GOAL
            return action_result

        if map[target_x][target_y] == 3:
            action_result = COLLISION

        return action_result

    def state_to_key(self, state):
        """transfer state to key in the Q_table dictionary"""
        return state[0] * 10 + state[1]

    def get_result(self, check_result, curr_state, target):
        result = [0, [0, 0]]
        if check_result == COLLISION or check_result == OUT_OF_TABLE:    # 如果碰撞了或者出了桌子
            result[0] = REWARD_COLLISION
            result[1] = curr_state    # 位置不变
            print('Collision detected')
            return result

        if check_result == OK_TO_GO:
            result[0] = REWARD_MOVE
            curr_state = target
            result[1] = curr_state
            print('Moving one step')
            return result

        if check_result == REACH_GOAL:
            result[0] = REWARD_GOAL
            curr_state = target
            result[1] = curr_state
            print('Mission completed!')
            return result

    def get_init_pos(self):
        """get the initial position of beginning and ending"""

        i, j = 0, 0
        beg_pos, dst_pos = [0, 0]
        init_pos = [[0, 0], [0, 0]]
        for index_i in map:
            for index_j in index_i:
                if index_j == 1:
                    beg_pos = [i, j]
                if index_j == 2:
                    dst_pos = [i, j]
                # print("[" + str(i) + ", " + str(j) + "]" + "=" + str(index_j))
                j = j + 1
            i = i + 1
            j = 0

        print(beg_pos)
        print(dst_pos)

        init_pos[0] = beg_pos
        init_pos[1] = dst_pos
        return init_pos

    def q_learning(self, learning_model, iteration_num, iterations, performances):
        """This method is used to iteratively run Q_learning model and get final result"""

        beg_pos = self.get_init_pos()[0]
        dst_pos = self.get_init_pos()[1]


        for i in range(iteration_num):
            count_actions = 0
            curr_state = beg_pos    # 回到初始位置
            goal_state = dst_pos    # 目标位置（mat的位置）
            next_state = [beg_pos]
            print(" \nCurrent iteration:" + str(i+1))

            while curr_state != goal_state:
                curr_action = learning_model.get_action(self.state_to_key(curr_state))    # 得到下一动作(epsilon-greedy policy)
                curr_goal = self.action_to_goal(curr_action, curr_state)    # get target state
                goal_result = self.check_goal(curr_goal[0], curr_goal[1], dst_pos)    # check the result of the goal

                result = self.get_result(goal_result, curr_state, curr_goal)
                reward = result[0]    # get the reward of this action
                next_state.append(result[1])

                learning_model.learn(self.state_to_key(curr_state), curr_action,
                                     reward, self.state_to_key(next_state[-1]) )    # 更新Q_table

                curr_state = next_state[-1]     # 切换到下一状态
                count_actions = count_actions + 1   # 计算总共花费的actions（也可以用cumulative reward？跟游戏本身规则相关）

                # self.visualization(next_state)    # 实时可视化每一次探索的状况
            iterations.append(i)
            performances.append(count_actions)

            print("Reach goal!! Actions taken:" + str(count_actions))
            print("Trajectory: " + str(next_state))

        self.visualization(next_state)  # 实时可视化每一轮成功探索后的状况


    def visualization(self, trajectory_state):
        vis = QLearningVisual(self.learning_model.q_table, self.iterations, self.performances)
        vis.visual_iter_process()
        vis.visual_heatmap(trajectory_state)
        vis.visual_state_action(self.learning_model.q_table)


test = QLearningTest()
#test.visualization()
