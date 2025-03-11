import numpy as np

def generate_zipf_sequence(num_contents, alpha, length):
    """
    生成给定内容总数和 Zipf 参数的请求序列。

    参数:
        num_contents: 内容种类数量 M
        alpha: Zipf 分布参数
        length: 要生成的请求数

    返回:
        长度为 length 的内容 ID 序列 (numpy 数组)
    """
    # Zipf 概率分布
    ranks = np.arange(1, num_contents + 1)  # 内容ID从1到M
    # 计算每个内容的Zipf概率
    probabilities = 1 / np.power(ranks, alpha)
    probabilities /= probabilities.sum()  # 归一化
    # 按照Zipf概率分布随机抽取内容ID序列
    requests = np.random.choice(ranks, size=length, p=probabilities)
    return requests

def generate_request_stream(num_contents, segment_lengths, alphas):
    """
    生成带有多个阶段的用户请求序列，每个阶段使用不同的 Zipf 参数。

    参数:
        num_contents: 内容种类数量 M
        segment_lengths: 列表，各阶段请求数量
        alphas: 列表，各阶段对应的 Zipf 参数

    返回:
        full_sequence: 完整的请求序列 (numpy 数组)
        distributions: 每段请求序列的实际请求频率分布列表 (list of numpy 数组)
    """
    assert len(segment_lengths) == len(alphas)
    full_sequence = []
    distributions = []  # 保存每段的内容概率分布
    for length, alpha in zip(segment_lengths, alphas):
        seq = generate_zipf_sequence(num_contents, alpha, length)
        full_sequence.extend(seq)
        # 计算该段的实际请求频率分布（相对频率）
        unique, counts = np.unique(seq, return_counts = True)
        freq_dist = np.zeros(num_contents)
        freq_dist[unique - 1] = counts / length  # 内容ID从1开始，转为0索引
        distributions.append(freq_dist)
    return np.array(full_sequence), distributions

# 示例: 生成包含两段的请求序列，第一段Zipf参数0.8 (10000请求)，第二段Zipf参数1.2 (10000请求)
if __name__ == '__main__':
    # 简单测试
    requests, dists = generate_request_stream(num_contents=100,
                                              segment_lengths=[10000, 10000],
                                              alphas=[0.8, 1.2])
    print("请求序列长度:", len(requests))
    print("第一段前10个请求:", requests[:10])
    print("第一段内容1的请求概率:", dists[0][0])
