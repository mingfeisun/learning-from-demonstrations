# coding=utf8
# Author: Ziming Chen
# Beginning Date: 2018/08/01
# -*- coding: utf-8 -*-

import numpy as np
import random
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib import cm
import matplotlib as mpl
from matplotlib import axes
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


class QLearningVisual:
    """This class is for visualizing the training process
    from several different aspects"""

    def __init__(self, q_table_dict, goal_state, iterations, performances, avg_accu_reward):
        self.q_table_dict = q_table_dict
        self.goal_state = goal_state
        self.iterations = iterations
        self.performances = performances
        self.avg_accu_reward = avg_accu_reward
        self.final_q_table = self.final_q_table_list(self.q_table_dict)

    def final_q_table_list(self, q_table_dict):
        """This method is used to process original q_table dictionary and
            transfer it to the final q_table list, which is a 2-dimensional matrix,
            with each value the maximal q_value of the corresponding state"""

        # print(q_table_dict)

        keys = sorted(q_table_dict.keys())
        # print(keys)

        list = [[0.0000] * 10 for i in range(0, 10)]
        i, j = 0, 0
        for key in keys:
            while key != (i, j):
                j = j + 1
                if j == 10:
                    j = 0
                    i = i + 1
            if np.average(q_table_dict[key]) == 0:
                list[i][j] = 0
            else:
                list[i][j] = round(np.max(q_table_dict[key]), 6)
            j = j + 1
            if j == 10:
                j = 0
                i = i + 1

        for row in list:
            print(row)

        # print(list)
        # print(q_table_dict[97])
        # print(q_table_dict[98])
        # print(q_table_dict[99])
        return list

    def visual_heatmap(self, trajectory_state):
        """Heatmap for visualizing the middle learning process of Q-learning and the final policy"""

        list = self.final_q_table
        xpos = np.arange(0, 10, 1)
        ypos = np.arange(0, 10, 1)

        # generate colors
        cm = plt.get_cmap('bwr')
        vv = range(len(list))
        cNorm = colors.Normalize(vmin=0, vmax=99)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        colorVals = [scalarMap.to_rgba(i) for i in range(100)]

        # generate plot data
        xpos = np.arange(0, 10, 1)
        ypos = np.arange(0, 10, 1)
        ypos, xpos = np.meshgrid(xpos, ypos)
        xpos = xpos.flatten()
        ypos = ypos.flatten()
        #print(xpos)
        #print(ypos)
        zpos = np.zeros(100)
        dx = 1.0 * np.ones_like(zpos)
        dy = dx.copy()
        dz = np.array(list).flatten()

        # generate plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        opacity = 1

        for i in range(100):
            if [xpos[i], ypos[i]] in trajectory_state:
                ax.bar3d(xpos[i], ypos[i], zpos[i], dx[i], dy[i], dz[i],
                        color='b', alpha=opacity, zsort='max')
            else:
                ax.bar3d(xpos[i], ypos[i], zpos[i], dx[i], dy[i], dz[i],
                         color='w', alpha=opacity, zsort='max')

            # ax.bar3d(xpos[i], ypos[i], zpos[i], dx[i], dy[i], dz[i],
            #          color=colorVals[sorted(dz).index(dz[i])], alpha=opacity, zsort='max')
            # print(xpos[i], ypos[i], zpos[i], dx[i], dy[i], dz[i])

        scalarMap.set_array(10)
        cb = fig.colorbar(scalarMap)

        ax.set_xlabel('state_x')
        ax.set_ylabel('state_y')
        ax.set_zlabel('Max_Q_Value')
        ax.set_title('Q_table')
        plt.autoscale(enable=True, axis='both', tight=True)
        plt.grid()
        plt.show(block=False)


        # mpl.rcParams['font.size'] = 10
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # xs = range(len(list))
        # ys = range(len(list[0]))
        # for z in range(len(list)):
        #     xs = range(len(list))
        #     ys = list[z]
        #     color = plt.cm.Set2(random.choice(range(plt.cm.Set2.N)))
        #     ax.bar(xs, ys, zs=z, zdir='y', color=color, alpha=0.5)
        #     ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(xs))
        #     ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(ys))
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('copies')
        # plt.show()


        # figure=plt.figure(facecolor='w')
        # ax=figure.add_subplot(2, 1, 1, position=[1, 1, 1, 1])
        # ax.set_yticks(range(len(ypos)))
        # ax.set_yticklabels(ypos)
        # ax.set_xticks(range(len(xpos)))
        # ax.set_xticklabels(xpos)
        # vmax=list[0][0]
        # vmin=list[0][0]
        # for i in list:
        #     for j in i:
        #         if j>vmax:
        #             vmax=j
        #         if j<vmin:
        #             vmin=j
        # map = ax.imshow(list,interpolation='nearest', cmap=cm.Blues, aspect='auto',vmin=vmin,vmax=vmax)
        # plt.colorbar(mappable=map,cax=None,ax=None,shrink=1)
        # plt.show()

    def performance_iter_process(self):
        """This method visualizes the performance in the training process of q_learning"""
        plt.plot(self.iterations, self.performances, linewidth=5)  # 参数linewidth决定plot()绘制的线条的粗细

        # 设置图标标题，并给坐标轴加上标签
        plt.title("Training process", fontsize=24)
        plt.xlabel("Iteration number", fontsize=14)
        plt.ylabel("Actions taken", fontsize=14)

        # 设置刻度标记的大小
        plt.tick_params(axis='both', labelsize=14)
        plt.show()

    def reward_iter_process(self):
        """This method visualizes the accumulated reward in the training process of q_learning"""
        plt.plot(self.iterations, self.avg_accu_reward, linewidth=5)  # 参数linewidth决定plot()绘制的线条的粗细

        # 设置图标标题，并给坐标轴加上标签
        plt.title("Training process", fontsize=24)
        plt.xlabel("Iteration number", fontsize=14)
        plt.ylabel("Average accumulative reward", fontsize=14)

        # 设置刻度标记的大小
        plt.tick_params(axis='both', labelsize=14)
        plt.show()

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

        keys = sorted(q_dict.keys())
        x, y, i = 0.5, 9.5, 1
        for key in keys:
            # print("key: " + str(key))
            while key[0]*10 + key[1] != i - 1:
                i = i + 1
                x = x + 1
                if x == 10.5:
                    x = 0.5
                    y = y - 1

            if key == self.goal_state:
                ax.scatter(x, y)
                i = i + 1
                x = x + 1
                continue

            if np.average(q_dict[key]) == 0:
                i = i + 1
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
            i = i + 1

        # 设置刻度标记的大小
        plt.tick_params(axis='both', labelsize=10)

        plt.show()
