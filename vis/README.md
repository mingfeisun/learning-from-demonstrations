# Visualization of the Reinforcement Learning training process

* Author: Ziming Chen
* Created: 2018-08-10
* Updated: 2018-09-02


### 四个模块：

1.	q_learning_model.py 
  
    该模块包含具体的q_learning算法，learn() 用于更新Q_table的值，get_action() 通过epsilon-greedy policy来获得当前状态的下一action。
    
2. q_lambda_naive_model.py

    改进的naive Q(λ) model，增加了eligibility traces来更新当前episode中的所有Q值。

2.	q_learning_visualization.py 

    该模块用于q_learning学习过程中的可视化实现。包含四个可视化方法：
    
    * visual_heatmap( )：第一个方法实时绘制q_table对应的热图，q值越高，高度越高，对应的colorbar颜色越高。

    * performance_iter_process( )：第二个折线图表示迭代过程中performance(action)的变化情况。
    
    * reward_iter_process( ): 第三个折线图表示迭代过程中accumulative reward的变化情况。

    * visual_state_action( )。第四个图表示探索过程中的state_action pair, 每一个状态中Q值最大的action被显示为红色的箭头。

3.	auto_q_learning.py 
    
    该模块用于调用另外两个模块，构建逻辑框架，实现测试功能。

### 调用方法：

* 测试模块:

  创建一个auto_q_learning模块下, QLearningTest类的test对象: test = QLearningTest()

* 可视化训练过程:

  vis = QLearningVisual(self.learning_model.q_table, goal_state, self.iterations, self.performances, self.avg_accu_reward)
  
  参数：需要传入learning_model的q_table（字典形式，key: 0~99，每个key对应4个action的Q值）；目标位置；迭代的次数（list）；每次episode对应的performances，也就是所花费的actions（list）；每次episode对应的accumulative reward
  
  可视化performance的变化情况，调用vis.performance_iter_process()；
  可视化accumulative reward的变化情况，调用vis.reward_iter_process()；
  可视化热图，调用vis.visual_heatmap()；
  可视化state_action pair， 调用vis.visual_state_action()

