# import numpy as np
# import random
# import matplotlib.pyplot as plt
# from matplotlib import colors
# from matplotlib.animation import FuncAnimation
# import time

# # 定义迷宫环境
# class Maze:
#     def __init__(self, size=20, n_agents=3):
#         self.size = size
#         self.n_agents = n_agents
#         self.maze = np.zeros((size, size))
#         self.start_positions = []
#         self.goal_positions = []
#         self.agent_positions = []
#         self.reset_maze()

#     def reset_maze(self):
#         # 随机生成障碍物
#         self.maze = np.zeros((self.size, self.size))
#         num_walls = int(self.size * self.size * 0.2)  # 随机生成20%的障碍物
#         for _ in range(num_walls):
#             x, y = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
#             self.maze[x, y] = -1  # -1表示障碍物

#         # 随机设置智能体的起点和终点
#         self.start_positions = []
#         self.goal_positions = []
#         self.agent_positions = []
#         for i in range(self.n_agents):
#             while True:
#                 start = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
#                 goal = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
#                 if self.maze[start] == 0 and self.maze[goal] == 0 and start != goal:
#                     self.start_positions.append(start)
#                     self.goal_positions.append(goal)
#                     self.agent_positions.append(start)
#                     break

#     def is_valid_move(self, agent_idx, new_pos):
#         # 检查是否越界或撞墙或与其他智能体冲突
#         if (new_pos[0] < 0 or new_pos[0] >= self.size or
#             new_pos[1] < 0 or new_pos[1] >= self.size):
#             return False  # 越界
#         if self.maze[new_pos] == -1:  # 障碍物
#             return False
#         for i in range(self.n_agents):
#             if i != agent_idx and self.agent_positions[i] == new_pos:
#                 return False  # 另一个智能体
#         return True

#     def move_agent(self, agent_idx, action):
#         # action: 0=up, 1=down, 2=left, 3=right
#         cur_pos = self.agent_positions[agent_idx]
#         if action == 0:  # up
#             new_pos = (cur_pos[0] - 1, cur_pos[1])
#         elif action == 1:  # down
#             new_pos = (cur_pos[0] + 1, cur_pos[1])
#         elif action == 2:  # left
#             new_pos = (cur_pos[0], cur_pos[1] - 1)
#         elif action == 3:  # right
#             new_pos = (cur_pos[0], cur_pos[1] + 1)

#         if self.is_valid_move(agent_idx, new_pos):
#             self.agent_positions[agent_idx] = new_pos

#     def render(self, ax):
#         cmap = colors.ListedColormap(['white', 'black', 'red', 'green', 'blue'])
#         grid = np.copy(self.maze)
#         for i, pos in enumerate(self.agent_positions):
#             grid[pos] = 2 + i  # 智能体位置不同颜色
#         ax.imshow(grid, cmap=cmap)
#         ax.set_xticks([]), ax.set_yticks([])

# # Q-Learning 智能体
# class QLearningAgent:
#     def __init__(self, env, agent_idx, alpha=0.1, gamma=0.9, epsilon=0.1):
#         self.env = env
#         self.agent_idx = agent_idx
#         self.alpha = alpha  # 学习率
#         self.gamma = gamma  # 折扣因子
#         self.epsilon = epsilon  # 探索概率
#         self.q_table = np.zeros((env.size, env.size, 4))  # 每个格子有4个动作

#     def choose_action(self, state):
#         if random.uniform(0, 1) < self.epsilon:
#             return random.randint(0, 3)  # 探索
#         else:
#             return np.argmax(self.q_table[state[0], state[1], :])  # 利用

#     def learn(self, state, action, reward, next_state):
#         predict = self.q_table[state[0], state[1], action]
#         target = reward + self.gamma * np.max(self.q_table[next_state[0], next_state[1], :])
#         self.q_table[state[0], state[1], action] += self.alpha * (target - predict)

# # 初始化智能体和环境
# env = Maze(size=20, n_agents=3)
# agents = [QLearningAgent(env, i) for i in range(env.n_agents)]

# # 训练过程
# def train(episodes=1000):
#     for episode in range(episodes):
#         env.reset_maze()
#         for agent in agents:
#             done = False
#             while not done:
#                 state = env.agent_positions[agent.agent_idx]
#                 action = agent.choose_action(state)
#                 env.move_agent(agent.agent_idx, action)
#                 next_state = env.agent_positions[agent.agent_idx]
                
#                 if next_state == env.goal_positions[agent.agent_idx]:
#                     reward = 100  # 到达终点
#                     done = True
#                 else:
#                     reward = -1  # 每次移动的代价
                
#                 agent.learn(state, action, reward, next_state)

# # 实时演示
# fig, ax = plt.subplots()

# def update(frame):
#     ax.clear()
#     env.render(ax)
#     for agent in agents:
#         state = env.agent_positions[agent.agent_idx]
#         action = agent.choose_action(state)
#         env.move_agent(agent.agent_idx, action)
#     time.sleep(0.2)  # 每次移动后延迟 0.2 秒

# def run_demo():
#     env.reset_maze()
#     ani = FuncAnimation(fig, update, frames=range(100), repeat=False)
#     plt.show()

# # 训练智能体
# train(episodes=1)

# # 运行演示
# run_demo()
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import torch
import torch.nn as nn
import torch.optim as optim

# 环境类
class MazeEnv:
    def __init__(self, size=10, n_agents=2):
        self.size = size
        self.n_agents = n_agents
        self.maze = np.zeros((size, size))
        self.agent_positions = []
        self.start_positions = []
        self.goal_positions = []
        self.reset()

    def reset(self):
        self.maze = np.zeros((self.size, self.size))
        self.agent_positions = []
        self.start_positions = []
        self.goal_positions = []

        # 随机生成障碍物
        num_walls = int(self.size * self.size * 0.2)  # 20%的障碍物
        for _ in range(num_walls):
            x, y = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
            self.maze[x, y] = -1  # -1 表示障碍物

        # 随机分配智能体起点和终点
        for i in range(self.n_agents):
            while True:
                start = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
                goal = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
                if self.maze[start] == 0 and self.maze[goal] == 0 and start != goal:
                    self.start_positions.append(start)
                    self.goal_positions.append(goal)
                    self.agent_positions.append(start)
                    break

    def is_valid_move(self, agent_idx, new_pos):
        # 检查是否超出边界或与障碍物/其他智能体冲突
        if (new_pos[0] < 0 or new_pos[0] >= self.size or
            new_pos[1] < 0 or new_pos[1] >= self.size):
            return False
        if self.maze[new_pos] == -1:  # 障碍物
            return False
        for i, pos in enumerate(self.agent_positions):
            if i != agent_idx and pos == new_pos:
                return False  # 其他智能体位置
        return True

    def move_agent(self, agent_idx, action):
        # 动作：0=上，1=下，2=左，3=右
        cur_pos = self.agent_positions[agent_idx]
        if action == 0:  # 上
            new_pos = (cur_pos[0] - 1, cur_pos[1])
        elif action == 1:  # 下
            new_pos = (cur_pos[0] + 1, cur_pos[1])
        elif action == 2:  # 左
            new_pos = (cur_pos[0], cur_pos[1] - 1)
        elif action == 3:  # 右
            new_pos = (cur_pos[0], cur_pos[1] + 1)

        if self.is_valid_move(agent_idx, new_pos):
            self.agent_positions[agent_idx] = new_pos

    def render(self, ax):
        cmap = plt.get_cmap('Greys')
        grid = np.copy(self.maze)
        for i, pos in enumerate(self.agent_positions):
            grid[pos] = 2 + i  # 智能体不同颜色
        ax.imshow(grid, cmap=cmap)
        ax.set_xticks([]), ax.set_yticks([])

# QMIX 神经网络架构
class QMIXNet(nn.Module):
    def __init__(self, n_agents, state_dim, action_dim):
        super(QMIXNet, self).__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Q网络部分：每个智能体的局部Q网络
        self.agent_q_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, action_dim)
            ) for _ in range(n_agents)
        ])

        # 混合网络
        self.hyper_w1 = nn.Linear(state_dim, n_agents * 64)
        self.hyper_w2 = nn.Linear(state_dim, 64 * 1)
        self.hyper_b1 = nn.Linear(state_dim, 64)
        self.hyper_b2 = nn.Linear(state_dim, 1)

    def forward(self, states, actions):
        agent_q_values = []
        for i in range(self.n_agents):
            q_value = self.agent_q_nets[i](states[i])
            agent_q_values.append(q_value[actions[i]])

        q_tot = sum(agent_q_values)

        # 使用混合网络计算总Q值
        w1 = self.hyper_w1(states.view(-1)).view(self.n_agents, 64)
        b1 = self.hyper_b1(states.view(-1)).view(64)
        w2 = self.hyper_w2(states.view(-1)).view(64)
        b2 = self.hyper_b2(states.view(-1))

        q_tot = w2 @ torch.tanh(w1 @ q_tot + b1) + b2
        return q_tot

# Q-learning算法
class QMixAgent:
    def __init__(self, env, n_agents=2, state_dim=2, action_dim=4, lr=0.01, gamma=0.99):
        self.env = env
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.gamma = gamma
        self.qmix_net = QMIXNet(n_agents, state_dim, action_dim)
        self.optimizer = optim.Adam(self.qmix_net.parameters(), lr=lr)
        self.epsilon = 1.0  # 探索率

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_dim - 1)  # 随机探索
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            q_values = self.qmix_net.agent_q_nets[0](state_tensor)  # 使用智能体0的Q网络
            return torch.argmax(q_values).item()  # 贪婪选择

    def train(self, episodes=1000):
        for episode in range(episodes):
            self.env.reset()
            done = False
            while not done:
                for agent_idx in range(self.n_agents):
                    state = self.env.agent_positions[agent_idx]
                    action = self.choose_action(state)
                    self.env.move_agent(agent_idx, action)
                    # 假设这里简化为：每个智能体到达目标时才结束
                    if self.env.agent_positions == self.env.goal_positions:
                        done = True
                        break

            if episode % 100 == 0:
                print(f'Episode {episode}/{episodes} finished')

    def render(self, ax):
        self.env.render(ax)

# 初始化环境与智能体
env = MazeEnv(size=10, n_agents=2)
agent = QMixAgent(env)

# 设置图形化显示
fig, ax = plt.subplots()

def update(frame):
    ax.clear()
    agent.render(ax)
    time.sleep(0.5)  # 延迟以查看移动效果

# 运行演示
def run_demo():
    ani = FuncAnimation(fig, update, frames=range(100), repeat=False)
    plt.show()

# 训练智能体并展示
agent.train(episodes=1)
run_demo()

