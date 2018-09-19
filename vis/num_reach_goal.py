# coding=utf8
# Author: Ziming Chen
# Beginning Date: 2018/09/09

from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
import os

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


class NumReachGoal:

    def __init__(self):

        self.avg_actions_mode1 = []
        self.avg_actions_mode2 = []
        self.avg_actions_mode3 = []
        self.avg_actions_mode4 = []
        self.num_of_states_mode1 = []
        self.num_of_states_mode2 = []
        self.num_of_states_mode3 = []
        self.num_of_states_mode4 = []
        self.q_table_dict = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        self.q_table_list = [[]]
        self.iterations = range(54)

    def file_process(self):
        path_user = '/Users/apple/Desktop/raw/'
        files_user = os.listdir(path_user)  # files of all users
        # files_user.sort(key=lambda x:int(x[0:2]))
        for file_user in files_user:
            if file_user == 'feng' :
                continue

            user_name = file_user
            self.reset_avg_actions()    # start a new user
            self.reset_num_of_states()  # start a new user

            path_files_user = path_user + file_user # files of a particular user
            # mode_name = os.listdir(path_files_user)
            # mode_name.sort(key=lambda x:int(x[4:]))
            for mode in range(1,5):
                # self.iterations.append(i)
                # path = '/Users/apple/Desktop/mingfei/mode' + str(mode) + '/q_table'
                path_files_user_mode = path_files_user + "/mode" + str(mode) + '/q_table/'  # the path of q_table files in one mode of a user
                files = os.listdir(path_files_user_mode)  # get all file names of the q_table
                q_table_pre = files[0].split('-')[0] + '-' + files[0].split('-')[1] + '-'
                print('\n' + 'mode: ' + str(mode) + ' ' + str(q_table_pre))
                lens = len(files)
                # files.sort(key=lambda x:int(x[-6:-4]))
                # print(files)
                # iter = 0
                for i in range(1, lens+1):
                    # self.iterations.append(iter)
                    self.file_object = open(path_files_user_mode + q_table_pre + str(i) + ".txt")  # 文件夹目录
                    # self.file_object = open(path + '/mingfei-q_table_value-' + str(i) + ".txt")  # 文件夹目录
                    self.q_table_dict = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
                    self.q_table_list = [[]]
                    self.read_data(self.file_object, self.q_table_list)
                    # self.visual_state_action()
                    if mode == 1:
                        self.try_reach_goal(self.q_table_dict, self.avg_actions_mode1, self.num_of_states_mode1)
                    elif mode == 2:
                        self.try_reach_goal(self.q_table_dict, self.avg_actions_mode2, self.num_of_states_mode2)
                    elif mode == 3:
                        self.try_reach_goal(self.q_table_dict, self.avg_actions_mode3, self.num_of_states_mode3)
                    elif mode == 4:
                        self.try_reach_goal(self.q_table_dict, self.avg_actions_mode4, self.num_of_states_mode4)

            # self.show_avg_actions()
            self.show_num_of_states_combine(user_name)

    def reset_avg_actions(self):
        self.avg_actions_mode1 = []
        self.avg_actions_mode2 = []
        self.avg_actions_mode3 = []
        self.avg_actions_mode4 = []

    def reset_num_of_states(self):
        self.num_of_states_mode1 = []
        self.num_of_states_mode2 = []
        self.num_of_states_mode3 = []
        self.num_of_states_mode4 = []

    def read_data(self, file_object, q_table_list):
        try:
            for line in file_object:
                # print(line)
                q_table_list.append(line.strip().split('\t'))
                # q_table_dict.append(line.strip())
            # print(q_table_list)
        finally:
            file_object.close()
        self.get_q_table_dict(q_table_list, self.q_table_dict)

    def get_q_table_dict(self, q_table_list, q_table_dict):
        for dict_i in range(10):
            for dict_j in range(10):
                for dict_value in range(4):
                    dict_key = (dict_i, dict_j)
                    list_i = dict_i * 10 + dict_j + 2
                    # print(q_table_list[list_i][dict_value+1])
                    q_table_dict[dict_key][dict_value] = float(q_table_list[list_i][dict_value+1])
        # for key in sorted(q_table_dict.keys()):
        #     print(str(key) + ": " + str(q_table_dict[key]))

    def get_action_max(self, state, q_table_dict):
        # take action according to the q function table
        # return self.q_table_dict[state].index(np.max(self.q_table_dict[state]))
        state_action = q_table_dict[state]
        # print(state_action)
        # print(np.max(state_action))
        action = state_action.index(np.max(state_action))
        return action

    def action_to_goal(self, _action, curr_state):
        """get target state of the current action"""

        target = (0, 0)
        if _action == 0:    #left
            target = (curr_state[0], curr_state[1] - 1)
            # target[1] = curr_state[1] - 1
        if _action == 1:    #up
            target = (curr_state[0] - 1, curr_state[1])
            # target[1] = curr_state[1]
        if _action == 2:    #right
            target = (curr_state[0], curr_state[1] + 1)
            # target[1] = curr_state[1] + 1
        if _action == 3:    #down
            target = (curr_state[0] + 1, curr_state[1])
            # target[1] = curr_state[1]
        return target

    def check_goal(self, target_x, target_y, dst_pos):
        """check the result of the goal"""

        action_result = "OK_TO_GO"

        if target_x < 0 or target_x>9 or target_y < 0 or target_y>9:
            action_result = "OUT_OF_TABLE"
            return action_result

        # if target_x == dst_pos[0] and target_y == dst_pos[1]:
        #     action_result = "REACH_GOAL"
        #     return action_result

        if map[target_x][target_y] == 3:
            # print(target_y, target_y)
            action_result = "COLLISION"

        return action_result

    def try_reach_goal(self, q_table_dict, avg_actions, num_of_states):
        """This method is used to iteratively run Q_learning model and get final result"""

        goal_state = (1, 8)
        keys = sorted(q_table_dict.keys())
        reach_goal_state = []
        performances = []
        # print(q_table_dict)

        for key in keys:
            fail_flag = 0
            visit = []
            count_actions = 0
            curr_state = key
            # print("keys: " + str(key))
            # print("\n" + "begin_state: " + str(curr_state))

            while curr_state != goal_state:
                if np.average(q_table_dict[key]) == 0:  # avg Q value equals 0
                    fail_flag = 1
                    # print("Q value equals 0")
                    break
                visit.append(curr_state)
                curr_action = self.get_action_max(curr_state, q_table_dict)    # 得到下一动作(epsilon-greedy policy)
                curr_goal = self.action_to_goal(curr_action, curr_state)    # get target state
                # print('curr_goal: ' + str(curr_goal))
                goal_result = self.check_goal(curr_goal[0], curr_goal[1], goal_state)    # check the result of the goal
                if goal_result == "OK_TO_GO" :  # including reach goal
                    curr_state = curr_goal
                else :  # collision or out of table
                    fail_flag = 1
                    # print("collision or out of table")
                    break
                if curr_state in visit: # a loop occurs
                    fail_flag = 1
                    # print("a loop occurs")
                    break

                count_actions = count_actions + 1   # 计算总共花费的actions（也可以用cumulative reward？跟游戏本身规则相关）
                # print('curr_state' + str(curr_state))

            if fail_flag != 1:
                reach_goal_state.append(key)
                performances.append(count_actions)
                # print("Reach goal!! Actions taken:" + str(count_actions))

        avg_actions.append(np.average(performances))
        num_of_states.append(len(reach_goal_state))

        print("Average actions to take: " + str(avg_actions[-1]))
        print("Number of states that can reach goal: " + str(num_of_states[-1]))
        print("These states are: " + str(reach_goal_state))
        for state in q_table_dict.keys():
            if state not in reach_goal_state:
                print("not reach goal: " + str(state))

    def visual_state_action(self):
        """This method visualizes the state_action pair lively, with the red arrow
            referring to the maximum Q_value state_action pair."""

        q_dict = self.q_table_dict
        plt.figure(dpi=220, figsize=(7, 7))
        ax = plt.axes()
        ax.set(xlim=[0, 10], ylim=[0, 10])

        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))  # 设置x主坐标间隔 1
        ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))  # 设置y主坐标间隔 1
        ax.grid(True, linestyle="-", color="0.6", linewidth="1")
        # ax.scatter(8.5, 7.5)

        keys = sorted(self.q_table_dict.keys())
        x, y = 0.5, 9.5
        for key in keys:
            # print("key: " + str(key))

            if key == (1, 8):
                ax.scatter(x, y)
                x = x + 1
                continue

            if np.average(q_dict[key]) == 0:
                x = x + 1
                if x == 10.5:
                    x = 0.5
                    y = y - 1
                continue

            if q_dict[key].index(np.max(q_dict[key])) == 0:
                plt.annotate('', xy=(x - 0.5, y), xytext=(x, y),
                             arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='red'))
            else:
                plt.annotate('', xy=(x - 0.5, y), xytext=(x, y),
                             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

            if q_dict[key].index(np.max(q_dict[key])) == 1:
                plt.annotate('', xy=(x, y + 0.5), xytext=(x, y),
                             arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='red'))
            else:
                plt.annotate('', xy=(x, y + 0.5), xytext=(x, y),
                             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

            if q_dict[key].index(np.max(q_dict[key])) == 2:
                plt.annotate('', xy=(x + 0.5, y), xytext=(x, y),
                             arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='red'))
            else:
                plt.annotate('', xy=(x + 0.5, y), xytext=(x, y),
                             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

            if q_dict[key].index(np.max(q_dict[key])) == 3:
                plt.annotate('', xy=(x, y - 0.5), xytext=(x, y),
                             arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='red'))
            else:
                plt.annotate('', xy=(x, y - 0.5), xytext=(x, y),
                             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

            x = x + 1
            if x == 10.5:
                x = 0.5
                y = y - 1

        # 设置刻度标记的大小
        plt.tick_params(axis='both', labelsize=10)
        plt.show()

    def show_avg_actions_combine(self):
        """This method visualizes the average actions of those states that can reach the goal in the training process of q_learning"""
        plt.title("Training process", fontsize=24)
        while len(self.avg_actions_mode1) < 54:
            self.avg_actions_mode1.append(0)
        while len(self.avg_actions_mode2) < 54:
            self.avg_actions_mode2.append(0)
        while len(self.avg_actions_mode3) < 54:
            self.avg_actions_mode3.append(0)
        while len(self.avg_actions_mode4) < 54:
            self.avg_actions_mode4.append(0)
        plt.plot(self.iterations, self.avg_actions_mode1, linewidth=5, label='mode1')  # 参数linewidth决定plot()绘制的线条的粗细
        plt.plot(self.iterations, self.avg_actions_mode2, linewidth=5, label='mode2')  # 参数linewidth决定plot()绘制的线条的粗细
        plt.plot(self.iterations, self.avg_actions_mode3, linewidth=5, label='mode3')  # 参数linewidth决定plot()绘制的线条的粗细
        plt.plot(self.iterations, self.avg_actions_mode4, linewidth=5, label='mode4')  # 参数linewidth决定plot()绘制的线条的粗细
        plt.legend()
        # 设置图标标题，并给坐标轴加上标签

        plt.xlabel("Iteration number", fontsize=14)
        plt.ylabel("Avg actions to reach the goal", fontsize=14)

        # 设置刻度标记的大小
        plt.tick_params(axis='both', labelsize=14)
        plt.show()

    def show_num_of_states_combine(self, user_name):
        """This method visualizes the the number of states that can reach the goal in the training process of q_learning"""
        print(user_name)
        plt.title(user_name, fontsize=24)
        while len(self.num_of_states_mode1) < 54:
            conti = self.num_of_states_mode1[-1]
            self.num_of_states_mode1.append(conti)
            # print(self.num_of_states_mode1)
        while len(self.num_of_states_mode2) < 54:
            conti = self.num_of_states_mode2[-1]
            self.num_of_states_mode2.append(conti)
        while len(self.num_of_states_mode3) < 54:
            conti = self.num_of_states_mode3[-1]
            self.num_of_states_mode3.append(conti)
        while len(self.num_of_states_mode4) < 54:
            conti = self.num_of_states_mode4[-1]
            self.num_of_states_mode4.append(conti)
        plt.plot(self.iterations, self.num_of_states_mode1, linewidth=5, label='mode1')  # 参数linewidth决定plot()绘制的线条的粗细
        plt.plot(self.iterations, self.num_of_states_mode2, linewidth=5, label='mode2')  # 参数linewidth决定plot()绘制的线条的粗细
        plt.plot(self.iterations, self.num_of_states_mode3, linewidth=5, label='mode3')  # 参数linewidth决定plot()绘制的线条的粗细
        plt.plot(self.iterations, self.num_of_states_mode4, linewidth=5, label='mode4')  # 参数linewidth决定plot()绘制的线条的粗细
        plt.legend()

        plt.xlabel("Iteration number", fontsize=14)
        plt.ylabel("number of states that can reach the goal", fontsize=14)

        # 设置刻度标记的大小
        plt.tick_params(axis='both', labelsize=14)
        plt.savefig(user_name)
        plt.show()

    def show_num_of_states(self, iterations, num_of_states):
        """This method visualizes the the number of states that can reach the goal in the training process of q_learning"""
        plt.title('Training Process', fontsize=24)
        plt.plot(iterations, num_of_states, linewidth=5)  # 参数linewidth决定plot()绘制的线条的粗细

        plt.xlabel("Iteration number", fontsize=14)
        plt.ylabel("number of states that can reach the goal", fontsize=14)

        # 设置刻度标记的大小
        plt.tick_params(axis='both', labelsize=14)
        plt.show()

    def show_avg_actions(self, iterations, avg_actions):
        """This method visualizes the the number of states that can reach the goal in the training process of q_learning"""
        plt.title('Training Process', fontsize=24)
        plt.plot(iterations, avg_actions, linewidth=5)  # 参数linewidth决定plot()绘制的线条的粗细

        plt.xlabel("Iteration number", fontsize=14)
        plt.ylabel("Avg actions to reach the goal", fontsize=14)

        # 设置刻度标记的大小
        plt.tick_params(axis='both', labelsize=14)
        plt.show()

    def show_total_actions(self, iterations, num_of_states, avg_actions):
        """This method visualizes the the number of states that can reach the goal in the training process of q_learning"""
        total_actions = []
        plt.title('Training Process', fontsize=24)
        for i in range(len(avg_actions)):
            total_actions.append(num_of_states[i] * avg_actions[i])

        plt.plot(iterations, total_actions, linewidth=5)  # 参数linewidth决定plot()绘制的线条的粗细

        plt.xlabel("Iteration number", fontsize=14)
        plt.ylabel("Total actions to reach the goal", fontsize=14)

        # 设置刻度标记的大小
        plt.tick_params(axis='both', labelsize=14)
        plt.show()

test = NumReachGoal()