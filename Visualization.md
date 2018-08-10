# Visualization of the Reinforcement Learning training process

* Author: Ziming Chen
* Created: 2018-08-10
* Updated: 2018-08-10

### 分为三个模块：

1.	q_learning_model.py 
  
    该模块包含具体的q_learning算法，learn() 用于更新Q_table的值，get_action() 通过epsilon-greedy policy来获得当前状态的下一action。

2.	q_learning_visualization.py 

    该模块用于q_learning学习过程中的可视化实现。包含两个可视化方法：draw_heatmap() 和 visual_iter_process(). 第一个方法实时绘制q_table对应的热图，q值越高，高度越高，颜色越深。第二个折线图用来表示迭代过程中performance(action)的收敛趋势。

3.	auto_q_learning.py 
    
    该模块用于调用另外两个模块，构建逻辑框架，实现测试功能。

### 调用方法：

* 测试模块:

  创建一个auto_q_learning模块下, QLearningTest类的test对象: test = QLearningTest()

* 可视化训练过程:

  visual = QLearningVisual(self.learning_model.q_table, self.iterations, self.performances)。
  
  传入learning_model的q_table（字典形式，key: 0~99，每个key对应4个action的Q值），迭代的次数（list），对应的performances，也就是每次迭代所花的actions（也是list形式）。需要可视化iteration process，调用visual.visual_iter_process()；可视化热图，调用visual.draw_heatmap()
