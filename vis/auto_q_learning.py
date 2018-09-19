#coding=utf8
#!/usr/bin/env python\
import numpy as np

from q_learning_model import QLearningModel
from q_learning_visualization import QLearningVisual
from q_lambda_naive_model import QLambdaNaiveModel
from evaluation import num_reach_goal
# from data_visualization import random_walk

# map = [[1, 0, 3, 0, 3, 0, 0, 0, 0, 0],
#        [0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
#        [0, 0, 3, 0, 0, 0, 3, 0, 2, 0],
#        [0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 3, 0, 0, 3, 0, 0],
#        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#        [0, 3, 0, 0, 0, 3, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 0, 0, 3, 0, 0],
#        [0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

# map = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#        [0, 3, 0, 3, 0, 0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
#        [3, 0, 3, 0, 3, 0, 0, 0, 0, 0],
#        [0, 0, 0, 3, 0, 0, 0, 3, 0, 0],
#        [0, 0, 3, 2, 0, 0, 0, 0, 0, 0],
#        [0, 3, 0, 0, 0, 3, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 0, 0, 3, 0, 0],
#        [0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]


map = [[1, 0, 3, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 3, 0, 2, 0],
       [0, 0, 3, 0, 0, 0, 3, 0, 0, 0],
       [0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 3, 0, 0, 3, 3, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 3, 0, 0, 0, 3, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 1, 0, 3, 0, 0],
       [0, 1, 0, 0, 3, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

COLLISION = 0
OUT_OF_TABLE = 1
REACH_GOAL = 2
OK_TO_GO = 3

REWARD_COLLISION = -5
REWARD_MOVE = 0
REWARD_GOAL = 10
REWARD_NOT_READY = 0


class QLearningTest:
    """This class is for testing the Q_learning method"""

    def __init__(self):
        """initialize QLearningModel and transfer a list to parameter @actions in __init__
                0: left, 1: up, 2: right, 3: down"""
        self.learning_model = QLambdaNaiveModel([0, 1, 2, 3])
        # self.learning_model = QLearningModel([0, 1, 2, 3])
        self.iteration_num = 1000
        self.iterations = []
        self.performances = []
        self.avg_accu_reward = []
        self.avg_actions = []
        self.num_of_states = []
        self.num_reach_goal = num_reach_goal.NumReachGoal()
        self.q_learning(self.learning_model, self.iteration_num, self.iterations,
                        self.performances, self.avg_accu_reward)


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
            # print('Collision detected')
            return result

        if check_result == OK_TO_GO:
            result[0] = REWARD_MOVE
            curr_state = target
            result[1] = curr_state
            # print('Moving one step')
            return result

        if check_result == REACH_GOAL:
            result[0] = REWARD_GOAL
            curr_state = target
            result[1] = curr_state
            # print('Mission completed!')
            return result

    def get_init_pos(self):
        """get the initial position of beginning and ending"""

        i, j = 0, 0
        beg_pos = []
        dst_pos = []
        init_pos = [[], []]
        for index_i in map:
            for index_j in index_i:
                if index_j == 1:
                    beg_pos.append([i,j])
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

    def q_learning(self, learning_model, iteration_num, iterations, performances, avg_accu_reward):
        """This method is used to iteratively run Q_learning model and get final result"""

        beg_pos = self.get_init_pos()[0]
        dst_pos = self.get_init_pos()[1]

        for i in range(iteration_num):

            count_actions = 0
            # curr_state = beg_pos[0]    # 设置固定初始位置
            # curr_state = beg_pos[1]    # 设置固定初始位置
            # curr_state = beg_pos[2]    # 设置固定初始位置
            curr_state = beg_pos[i%3]    # 循环3个初始位置
            print("\nbegin_state: " + str(curr_state))
            goal_state = dst_pos    # 目标位置（mat的位置）
            next_state = [curr_state]
            print("Current iteration:" + str(i+1))

            while curr_state != goal_state:
                curr_action = learning_model.get_action(tuple(curr_state))    # 得到下一动作(epsilon-greedy policy)
                curr_goal = self.action_to_goal(curr_action, curr_state)    # get target state
                goal_result = self.check_goal(curr_goal[0], curr_goal[1], dst_pos)    # check the result of the goal

                result = self.get_result(goal_result, curr_state, curr_goal)
                reward = result[0]    # get the reward of this action
                next_state.append(result[1])

                learning_model.learn(tuple(curr_state), curr_action,
                                     reward, tuple(next_state[-1]) )    # 更新Q_table

                curr_state = next_state[-1]     # 切换到下一状态
                count_actions = count_actions + 1   # 计算总共花费的actions（也可以用cumulative reward？跟游戏本身规则相关）

                # print("q_table keys: " + str(sorted(learning_model.q_table.keys())))
                # print("eli_trace keys: " + str(sorted(learning_model.eligibility_traces.keys())))

                # self.visualization(next_state)    # 实时可视化每一次探索的状况

            learning_model.complete_one_episode()
            iterations.append(i)
            performances.append(count_actions)
            avg_accu_reward.append(self.cal_avg_accu_reward(learning_model.q_table))

            print("Reach goal!! Actions taken:" + str(count_actions))
            print("Trajectory: " + str(next_state))

            # print("q_table keys: " + str(sorted(learning_model.q_table.keys())))
            # print(learning_model.q_table)
            self.num_reach_goal.try_reach_goal(learning_model.q_table, self.avg_actions, self.num_of_states)

        self.num_reach_goal.show_num_of_states(iterations, self.num_of_states)
        self.num_reach_goal.show_avg_actions(iterations, self.avg_actions)
        self.num_reach_goal.show_total_actions(iterations, self.num_of_states, self.avg_actions)

        print("q_table keys: " + str(sorted(learning_model.q_table.keys())))
        print('number of keys: ' + str(len(learning_model.q_table.keys())))
        self.visualization(next_state, tuple(dst_pos))  # 实时可视化每一轮成功探索后的状况
        self.output_q_table(learning_model.q_table)

    def cal_avg_accu_reward(self, q_table_dict):
        max_q_value = []
        keys = q_table_dict.keys()
        for key in keys:
             max_q_value.append(np.max(q_table_dict[key]))

        return np.average(max_q_value)

    def visualization(self, trajectory_state, goal_state):
        vis = QLearningVisual(self.learning_model.q_table, goal_state, self.iterations, self.performances, self.avg_accu_reward)
        vis.performance_iter_process()
        vis.reward_iter_process()
        # vis.visual_heatmap(trajectory_state)
        vis.visual_state_action()

    def output_q_table(self, q_table_dict):
        filename = 'q_table_benchmark3.txt'
        keys = sorted(q_table_dict.keys())
        i, j = 0, 0
        with open(filename, 'w') as file_object:
            file_object.write('State' + '\t' + 'Left' + '\t' + 'Up' + '\t' + 'Right' + '\t' + 'Down' + '\n')
            for key in keys:
                while key != (i, j):
                    file_object.write(str((i, j)))
                    for value in range(4):
                        file_object.write('\t%.6f' % 0)
                    file_object.write('\n')
                    j = j + 1
                    if j == 10:
                        j = 0
                        i = i + 1

                file_object.write(str(key))
                for value in q_table_dict[key]:
                    file_object.write('\t%.6f' % value)
                file_object.write('\n')
                j = j + 1
                if j == 10:
                    j = 0
                    i = i + 1

        for key in keys:
            print(str(key) + '\t' + str(q_table_dict[key]))

test = QLearningTest()
#test.visualization()
