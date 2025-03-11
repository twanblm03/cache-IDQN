from collections import deque

class EdgeCacheEnv:
    def __init__(self, capacity, total_contents, window_size=1000):
        """
        初始化边缘缓存环境。

        参数:
            capacity: 缓存容量 L
            total_contents: 内容总数 M
            window_size: 用于计算滑动窗口命中率的窗口大小 K
        """
        self.capacity = capacity
        self.total_contents = total_contents
        # 使用固定大小列表表示缓存槽位，None表示空闲
        self.cache = [None] * capacity
        self.current_size = 0  # 当前缓存中内容数量
        # 记录内容最后一次被请求的时间，用于LRU决策（索引为内容ID-1）
        self.last_request_time = [-1] * total_contents
        # 当前时间步计数
        self.time = 0
        # 用于计算奖励的窗口
        self.window_size = window_size
        self.recent_hits = deque(maxlen=window_size)  # 存储最近窗口内每个请求是否命中（1或0）
        # 统计命中次数
        self.hits = 0
        self.misses = 0

    def request(self, content_id):
        """处理一个内容请求，返回是否命中、当前状态向量和是否需要替换动作标志。"""
        self.time += 1  # 时间步+1
        hit = False
        if content_id in self.cache:
            # 请求命中
            hit = True
            self.hits += 1
            # 更新该内容的最后请求时间（content_id从1开始，索引需减1）
            self.last_request_time[content_id - 1] = self.time
            # 在LRU策略中，我们会在外部更新缓存的使用顺序，但这里last_request_time足够提供信息
        else:
            # 未命中
            self.misses += 1
        # 记录当前请求的命中情况到滑动窗口
        self.recent_hits.append(1 if hit else 0)
        # 构造状态向量: [当前请求内容one-hot, 当前缓存内容one-hot]
        state = self._get_state_vector(content_id)
        # 如果未命中，需要替换动作
        need_replace = not hit
        return hit, state, need_replace

    def _get_state_vector(self, current_request):
        """构造状态向量：请求内容的 one-hot 与缓存内容的 one-hot 拼接"""
        # one-hot 向量长度 M 表示请求内容
        req_vector = [0] * self.total_contents
        req_vector[current_request - 1] = 1
        # one-hot 向量长度 M 表示缓存中内容
        cache_vector = [0] * self.total_contents
        for cid in self.cache:
            if cid is not None:
                cache_vector[cid - 1] = 1
        # 合并两个向量作为状态
        return req_vector + cache_vector

    def replace(self, action, content_id):
        """
        执行缓存替换动作，将指定槽位内容替换为 content_id。

        参数:
            action: 要替换的槽位索引 (0~L-1)
            content_id: 新内容 ID
        返回:
            替换后的新状态向量
        """
        # 执行替换
        if action < 0 or action >= self.capacity:
            raise ValueError("无效的动作索引")
        # 更新缓存槽位
        evicted = None
        if self.cache[action] is not None:
            evicted = self.cache[action]
        self.cache[action] = content_id
        # 如该槽位之前为空，则当前缓存数量增加
        if self.current_size < self.capacity and evicted is None:
            self.current_size += 1
        # 新内容的最后请求时间更新
        self.last_request_time[content_id - 1] = self.time
        # 返回执行动作后的下一个状态
        new_state = self._get_state_vector(content_id)
        return new_state

    def get_lru_action(self, _):
        """基于 LRU 策略返回要替换的槽位索引：选择最久未使用的内容所在槽位。"""
        if self.current_size < self.capacity:
            # 缓存未满，直接使用下一个空槽位
            return self.current_size  # 下一个可用槽位索引
            # 找到缓存中最久未被请求的内容
        oldest_time = float('inf')
        lru_index = None
        for idx, cid in enumerate(self.cache):
            if cid is None:
                # 空槽位（理论上不会在缓存满的情况下出现None）
                continue
            last_time = self.last_request_time[cid - 1]
            if last_time < oldest_time:
                oldest_time = last_time
                lru_index = idx
        # lru_index现在是最久未被访问的内容槽位
        return lru_index

    def get_window_hit_rate(self):
        """计算当前滑动窗口内的命中率。"""
        if len(self.recent_hits) == 0:
            return 0.0
        return sum(self.recent_hits) / len(self.recent_hits)
