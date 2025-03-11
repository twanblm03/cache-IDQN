import copy
import numpy as np
from data_generator import generate_zipf_sequence, generate_request_stream
from environment import EdgeCacheEnv
from dqn_agent import DQNAgent
from collections import deque

def calc_kl_divergence(p, q, eps=1e-9):
    # 计算 sum( p[i] * log(p[i]/q[i]) ), 忽略 p[i] = 0 的项
    kl = 0.0
    for i in range(len(p)):
        if p[i] > 0:
            kl += p[i] * np.log(p[i] / max(q[i], eps))
    return kl

# 假设一些全局参数
C = 10
M = 200   # 内容总数
K = 1000  # 周期窗口长度
W = 300   # KL散度检测窗口长度

# 离线训练：准备历史任务数据
offline_alphas = [0.8, 1.0, 1.4]  # 3种历史场景的Zipf参数
offline_length = 5000       # 每种场景用于训练的请求数
offline_models = []
offline_distributions = []

print("=== 离线训练历史模型 ===")
for alpha in offline_alphas:
    # 生成训练数据
    train_seq = generate_zipf_sequence(M, alpha, offline_length)
    # 初始化环境和Agent
    env = EdgeCacheEnv(capacity = C, total_contents = M, window_size = K)  # 示例缓存容量10
    agent = DQNAgent(state_dim = 2*M, action_dim = env.capacity,
                     epsilon1 = 0.2, epsilon2 = 0.3, epsilon_decay = 0.99)  # 初始更高的epsilon探索
    # 训练若干个epoch以收敛
    epochs = 1
    for ep in range(epochs):
        #env = EdgeCacheEnv(capacity=10, total_contents=M, window_size=100)  # 每轮重置环境
        for content in train_seq:
            hit, state, need_replace = env.request(content)
            if need_replace:
                # 未命中，需要决策替换
                action = agent.select_action(state, env)
                # 在执行替换之后，计算替换前后的滑动窗口命中率变化
                next_state = env.replace(action, content)
                reward = env.get_window_hit_rate()
                # done 标记可用于episode结束，这里每轮结尾作为done
                done = False
                agent.store_experience(state, action, reward, next_state, done)
                agent.update()
        # 每轮结束后，可适当降低epsilon或重置环境已完成一轮
    # 保存训练好的模型（深拷贝网络参数）
    trained_model = copy.deepcopy(agent.eval_net.state_dict())
    offline_models.append(trained_model)
    # 保存该场景的内容分布
    unique, counts = np.unique(train_seq, return_counts=True)
    dist = np.zeros(M)
    dist[unique-1] = counts / len(train_seq)
    offline_distributions.append(dist)
    print(f"训练完毕模型（alpha={alpha}），命中率={env.hits/len(train_seq):.2f}")

# 在线模拟序列，包含历史和新任务
online_alphas = [0.8, 1.4, 1.0]  # 第一段类似历史alpha=0.8，第二段1.0是新分布，第三段1.2类似历史
online_lengths = [3000, 3000, 3000]
online_requests, online_dists = generate_request_stream(M, online_lengths, online_alphas)

print("\n=== 在线模拟: IDQN 策略 ===")
env_idqn = EdgeCacheEnv(capacity = C, total_contents = M, window_size = K)
# 用于统计最近 W 步请求的出现频率
recent_requests = deque()
freq_counter = np.zeros(M)  # 长度为 M，freq_counter[i] 表示内容 i+1 在窗口内出现的次数
KL_THRESHOLD = 0.5  # KL 散度阈值，可自行调整
# 初始模型：假设第一段可能匹配alpha=0.8的历史模型
current_agent = DQNAgent(state_dim=2*M, action_dim=env_idqn.capacity,
                          epsilon1=0.1, epsilon2=0.1, epsilon_decay=0.99)
# # 如果识别出与历史模型相似，则加载其参数
# # 这里简单通过已知alpha判断模拟，在实际实现中应通过KL散度计算
# if abs(online_alphas[0] - 0.8) < 1e-3:
#     current_agent.eval_net.load_state_dict(offline_models[0])
#     current_agent.target_net.load_state_dict(offline_models[0])
model_library = {"alpha_0.8": offline_models[0], "alpha_1.0": offline_models[1], "alpha_1.4": offline_models[2]}  # 模型库
history_distributions = {"alpha_0.8": offline_distributions[0], "alpha_1.0": offline_distributions[1], "alpha_1.4": offline_distributions[2]}

# 运行在线请求序列
current_segment = 0  # 当前所属段索引
for t, content in enumerate(online_requests, start=1):
    # 将请求提交给环境
    hit, state, need_replace = env_idqn.request(content)
    # 定期进行相似度检测，例如每隔W请求，或监测内容分布变化
    # 1) 队列与计数器中加入新请求
    freq_counter[content - 1] += 1
    recent_requests.append(content)

    # 2) 如果队列长度 > W，则需要移除最旧的一条请求
    if len(recent_requests) > W:
        old_content = recent_requests.popleft()
        freq_counter[old_content - 1] -= 1
    # 为简化，我们在段切换点进行检测（t 达到每段长度时）
    # 在完成上面的队列更新后，加一个判断
    if t % W == 0:
        # 计算当前窗口内的概率分布 p
        window_sum = freq_counter.sum()  # 也就是 W（队列满的时候），但最好写通用一些
        if window_sum > 0:
            p = freq_counter / window_sum
        else:
            p = np.zeros(M)

        # 和已有的历史分布做KL散度比较
        similar_model = None
        min_kl = float('inf')
        for key, hist_dist in history_distributions.items():
            kl_val = calc_kl_divergence(p, hist_dist)
            if kl_val < min_kl:
                min_kl = kl_val
                similar_model = key

        # 如果 min_kl 小于阈值，则说明找到了相似的历史模型
        if min_kl < KL_THRESHOLD and similar_model is not None:
            print(f"检测到当前分布与历史模型 {similar_model} 相似 (KL={min_kl:.3f})，切换模型")
            # 如果当前模型还没存过，可以放进库
            model_library[f"some_label_{t}"] = copy.deepcopy(current_agent.eval_net.state_dict())
            history_distributions[f"some_label_{t}"] = p.copy()

            # 切换到历史模型
            new_agent = DQNAgent(state_dim=2 * M, action_dim=env_idqn.capacity,
                                 epsilon1=0.05, epsilon2=0.05, epsilon_decay=0.9997)
            new_agent.eval_net.load_state_dict(model_library[similar_model])
            new_agent.target_net.load_state_dict(model_library[similar_model])
            current_agent = new_agent
        else:
            print(f"未匹配到已有历史模型 (KL={min_kl:.3f})，保持/训练当前模型")
            # 也可以在此处决定是否开辟一个新模型，或者继续用当前模型在线更新
            # ...

    # 若当前请求未命中且需要替换，则由Agent选择动作并执行
    if need_replace:
        action = current_agent.select_action(state, env_idqn)
        next_state = env_idqn.replace(action, content)
        # 计算奖励: 使用滑动窗口命中率提高量作为奖励（可选）
        reward = env_idqn.get_window_hit_rate()
        done = False
        current_agent.store_experience(state, action, reward, next_state, done)
        current_agent.update()
# 计算IDQN结果
idqn_hit_rate = env_idqn.hits / len(online_requests)
idqn_avg_delay = (env_idqn.hits*5 + env_idqn.misses*15) / len(online_requests)  # 假设σ=5, φ=15(ms)
print(f"IDQN 命中率: {idqn_hit_rate:.3f}, 平均传输延迟: {idqn_avg_delay:.2f} ms")

# 基线算法模拟
print("\n=== 基线算法模拟 ===")
def simulate_lru(requests, capacity):
    cache = []
    hits = 0
    last_used = {}  # 记录每个内容上次使用时间
    time = 0
    for content in requests:
        time += 1
        if content in cache:
            hits += 1
            last_used[content] = time
            # 将命中的内容移到列表末尾表示最近使用（维护简单起见直接移除再append）
            cache.remove(content)
            cache.append(content)
        else:
            # 未命中
            if len(cache) < capacity:
                cache.append(content)
            else:
                # 移除最久未使用内容
                # 我们可以利用cache[0]为最久未用（因为每次命中都把内容移到末尾）
                evict = cache.pop(0)
                cache.append(content)
            last_used[content] = time
    hit_rate = hits / len(requests)
    return hit_rate, hits

def simulate_lfu(requests, capacity):
    cache = set()
    freq = {}   # 记录内容请求次数
    hits = 0
    for content in requests:
        freq[content] = freq.get(content, 0) + 1
        if content in cache:
            hits += 1
        else:
            if len(cache) < capacity:
                cache.add(content)
            else:
                # 找到当前缓存中使用频率最低的内容移除
                least_freq = min(freq[c] for c in cache)
                # 如有多个频率最低，移除其中任意一个（这里选择第一个满足的）
                to_remove = None
                for c in cache:
                    if freq[c] == least_freq:
                        to_remove = c
                        break
                if to_remove:
                    cache.remove(to_remove)
                cache.add(content)
    hit_rate = hits / len(requests)
    return hit_rate, hits

lru_hit_rate, lru_hits = simulate_lru(online_requests, capacity=10)
lfu_hit_rate, lfu_hits = simulate_lfu(online_requests, capacity=10)
# 平均延迟 (假设相同σ和φ)
lru_delay = (lru_hits*5 + (len(online_requests)-lru_hits)*15) / len(online_requests)
lfu_delay = (lfu_hits*5 + (len(online_requests)-lfu_hits)*15) / len(online_requests)
print(f"LRU 命中率: {lru_hit_rate:.3f}, 平均延迟: {lru_delay:.2f} ms")
print(f"LFU 命中率: {lfu_hit_rate:.3f}, 平均延迟: {lfu_delay:.2f} ms")
