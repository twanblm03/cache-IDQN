import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


class DQNModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        构建一个简单的多层感知器作为 DQN 模型。

        参数:
            state_dim: 状态向量长度
            action_dim: 动作空间大小
            hidden_dim: 隐藏层神经元数量
        """
        super(DQNModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 epsilon1 = 0.2, epsilon2 = 0.3, epsilon_decay=0.999, min_epsilon1 = 0.0, min_epsilon2=0.05,
                 target_update_freq=100, memory_capacity=10000, batch_size=32):
        """
        初始化 DQN 智能体。

        参数:
            state_dim: 状态向量维度（例如：2*M）
            action_dim: 动作空间大小（缓存容量 L）
            lr: 学习率
            gamma: 折扣因子
            epsilon1: 随机探索概率
            epsilon2: 使用 LRU 策略的概率
            epsilon_decay: epsilon 衰减率
            min_epsilon1, min_epsilon2: epsilon 的下限
            target_update_freq: 目标网络更新频率
            memory_capacity: 经验回放容量
            batch_size: 批次大小
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        # DQN评估网络和目标网络
        self.eval_net = DQNModel(state_dim, action_dim)
        self.target_net = DQNModel(state_dim, action_dim)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()
        # 优化器
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=lr)
        # 折扣因子
        self.gamma = gamma
        # ε-贪心参数
        self.epsilon1 = epsilon1  # 随机探索概率
        self.epsilon2 = epsilon2  # LRU动作概率
        self.epsilon_decay = epsilon_decay
        self.min_epsilon1 = min_epsilon1
        self.min_epsilon2 = min_epsilon2
        # 经验回放缓冲
        self.memory = deque(maxlen = memory_capacity)
        self.batch_size = batch_size
        # 目标网络更新频率
        self.target_update_freq = target_update_freq
        self.learn_step_counter = 0  # 学习步计数（用于判断何时更新目标网络）

    def select_action(self, state, env):
        """
        根据当前状态和环境选择动作，采用改进的 ε-贪心策略。

        参数:
            state: 当前状态向量（list 或 numpy 数组）
            env: 环境对象，用于调用 LRU 策略
        返回:
            动作索引（0 到 action_dim-1）
        """
        # 将state转换为tensor
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        # 随机数决定策略
        rand = random.random()
        if rand < self.epsilon1:
            # 随机动作
            action = random.randrange(self.action_dim)
        elif rand < self.epsilon1 + self.epsilon2:
            # LRU动作
            action = env.get_lru_action(None)
            if action is None:
                action = random.randrange(self.action_dim)
        else:
            # DQN动作：选择Q值最大的动作
            with torch.no_grad():
                q_values = self.eval_net(state_tensor)
                action = int(torch.argmax(q_values).item())
        return action

    def store_experience(self, state, action, reward, next_state, done):
        """存储经验到回放缓冲中。"""
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        """从回放缓冲中采样并更新网络参数。"""
        if len(self.memory) < self.batch_size:
            return  # 经验不够时不更新
        # 随机采样一个小批量
        batch = random.sample(self.memory, self.batch_size)
        # 将批数据转换为tensor
        state_batch = torch.tensor([b[0] for b in batch], dtype=torch.float)
        action_batch = torch.tensor([b[1] for b in batch], dtype=torch.long).unsqueeze(1)
        reward_batch = torch.tensor([b[2] for b in batch], dtype=torch.float).unsqueeze(1)
        next_state_batch = torch.tensor([b[3] for b in batch], dtype=torch.float)
        done_batch = torch.tensor([b[4] for b in batch], dtype=torch.float).unsqueeze(1)

        # 当前 Q值
        q_eval = self.eval_net(state_batch).gather(1, action_batch)
        # 下一个状态最大 Q值（目标网络）
        with torch.no_grad():
            q_next = self.target_net(next_state_batch)
            q_next_max = torch.max(q_next, dim=1)[0].unsqueeze(1)
            # 计算目标 Q值：如果done（结束），则只有即时奖励，否则包含下一状态价值
            q_target = reward_batch + self.gamma * q_next_max * (1 - done_batch)
        # 均方误差损失
        loss = nn.MSELoss()(q_eval, q_target)
        # 反向传播更新评估网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新目标网络参数
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        # 减小epsilon，提高策略贪心程度
        self.epsilon1 = max(self.min_epsilon1, self.epsilon1 * self.epsilon_decay)
        self.epsilon2 = max(self.min_epsilon2, self.epsilon2 * self.epsilon_decay)
