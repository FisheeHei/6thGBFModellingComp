
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据准备
indicators = [
    '市值(10-31当天)', '市场主导份额', '年内市值增长', '支持区块链数量',
    '平均交易量/市值比率', '交易对数量', '链上活跃度',
    '合规交易所占比', '机构投资者占比', '企业合作数量',
    '日平均脱钩率', '日平均波动率'
]

units = ['亿美元', '%', '%', '条', '%', '个', '万笔', '%', '%', '家', '%', '%']

data = np.array([
    [1833, 60, 33, 12, 46, 10000, 240, 59.90, 30, 127, 0.03, 8],  # USDT
    [759, 26, 73, 28, 17, 8000, 28.4, 75, 65, 345, 0.015, 1.2]  # USDC
])

# 判断指标方向（1为正向指标，-1为负向指标）
direction = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1])


# 1. 生成对比图像
def create_comparison_charts():
    # 第一张大图：市场规模类 (1-4)
    fig1 = plt.figure(figsize=(15, 10))
    fig1.suptitle('市场规模类指标对比', fontsize=16, fontweight='bold')
    gs1 = GridSpec(2, 2, figure=fig1)

    # 指标1：市值
    ax1 = fig1.add_subplot(gs1[0, 0])
    bars1 = ax1.bar(['USDT', 'USDC'], [data[0, 0], data[1, 0]],
                    color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
    ax1.set_title('市值对比 (亿美元)')
    ax1.bar_label(bars1, fmt='%.0f')

    # 指标2：市场主导份额（饼状图）
    ax2 = fig1.add_subplot(gs1[0, 1])
    labels = ['USDT', 'USDC', '其他']
    sizes = [60, 26, 14]  # 其他=100-60-26
    colors = ['#FF6B6B', '#4ECDC4', '#95A5A6']
    ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title('市场主导份额分布')

    # 指标3：年内市值增长
    ax3 = fig1.add_subplot(gs1[1, 0])
    bars3 = ax3.bar(['USDT', 'USDC'], [data[0, 2], data[1, 2]],
                    color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
    ax3.set_title('年内市值增长 (%)')
    ax3.bar_label(bars3, fmt='%.1f%%')

    # 指标4：支持区块链数量
    ax4 = fig1.add_subplot(gs1[1, 1])
    bars4 = ax4.bar(['USDT', 'USDC'], [data[0, 3], data[1, 3]],
                    color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
    ax4.set_title('支持区块链数量 (条)')
    ax4.bar_label(bars4, fmt='%.0f')

    plt.tight_layout()
    plt.savefig('市场规模类对比.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 第二张大图：流动性与交易活跃类 (5-7)
    fig2 = plt.figure(figsize=(15, 8))
    fig2.suptitle('流动性与交易活跃类指标对比', fontsize=16, fontweight='bold')
    gs2 = GridSpec(1, 3, figure=fig2)

    indicators_5_7 = indicators[4:7]
    data_5_7 = data[:, 4:7]

    for i in range(3):
        ax = fig2.add_subplot(gs2[0, i])
        bars = ax.bar(['USDT', 'USDC'], [data_5_7[0, i], data_5_7[1, i]],
                      color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
        ax.set_title(f'{indicators_5_7[i]} ({units[4 + i]})')
        if i == 6:  # 链上活跃度数值较大，格式化显示
            ax.bar_label(bars, fmt='%.1f')
        else:
            ax.bar_label(bars, fmt='%.0f')

    plt.tight_layout()
    plt.savefig('流动性与交易活跃类对比.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 第三张大图：合规性与机构采用类 (8-10)
    fig3 = plt.figure(figsize=(15, 8))
    fig3.suptitle('合规性与机构采用类指标对比', fontsize=16, fontweight='bold')
    gs3 = GridSpec(1, 3, figure=fig3)

    indicators_8_10 = indicators[7:10]
    data_8_10 = data[:, 7:10]

    for i in range(3):
        ax = fig3.add_subplot(gs3[0, i])
        bars = ax.bar(['USDT', 'USDC'], [data_8_10[0, i], data_8_10[1, i]],
                      color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
        ax.set_title(f'{indicators_8_10[i]} ({units[7 + i]})')
        if i == 0:  # 百分比显示
            ax.bar_label(bars, fmt='%.1f%%')
        else:
            ax.bar_label(bars, fmt='%.0f')

    plt.tight_layout()
    plt.savefig('合规性与机构采用类对比.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 第四张大图：风险与稳定性 (11-12)
    fig4 = plt.figure(figsize=(12, 6))
    fig4.suptitle('风险与稳定性指标对比', fontsize=16, fontweight='bold')
    gs4 = GridSpec(1, 2, figure=fig4)

    # 指标11：年平均脱钩率
    ax1 = fig4.add_subplot(gs4[0, 0])
    bars1 = ax1.bar(['USDT', 'USDC'], [data[0, 10], data[1, 10]],
                    color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
    ax1.set_title('年平均脱钩率 (%) - 越低越好')
    ax1.bar_label(bars1, fmt='%.3f%%')

    # 指标12：日平均波动率
    ax2 = fig4.add_subplot(gs4[0, 1])
    bars2 = ax2.bar(['USDT', 'USDC'], [data[0, 11], data[1, 11]],
                    color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
    ax2.set_title('日平均波动率 (%) - 越低越好')
    ax2.bar_label(bars2, fmt='%.3f%%')

    plt.tight_layout()
    plt.savefig('风险与稳定性对比.png', dpi=300, bbox_inches='tight')
    plt.close()


# 2. MinMax标准化
def minmax_normalization(data_matrix):
    min_vals = np.min(data_matrix, axis=0)
    max_vals = np.max(data_matrix, axis=0)

    # 避免除零
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1

    normalized_data = (data_matrix - min_vals) / range_vals
    return normalized_data


# 3. 熵值法计算权重
def entropy_weight(normalized_data):
    # 计算比重
    p_ij = normalized_data / np.sum(normalized_data, axis=0)

    # 避免log(0)的情况
    p_ij[p_ij == 0] = 1e-5

    # 计算熵值
    k = 1 / np.log(len(normalized_data))
    e_j = -k * np.sum(p_ij * np.log(p_ij), axis=0)

    # 计算差异系数
    d_j = 1 - e_j

    # 计算权重
    weights = d_j / np.sum(d_j)
    return weights


# 4. TOPSIS分析法
def topsis_analysis(normalized_data, weights, direction):
    # 方向调整：负向指标取倒数
    direction_adjusted = normalized_data.copy()
    for i in range(len(direction)):
        if direction[i] == -1:  # 负向指标
            direction_adjusted[:, i] = 1 - normalized_data[:, i]

    # 构建加权规范矩阵
    weighted_matrix = direction_adjusted * weights

    # 确定正理想解和负理想解
    ideal_best = np.max(weighted_matrix, axis=0)
    ideal_worst = np.min(weighted_matrix, axis=0)

    # 计算到正负理想解的距离
    dist_best = np.sqrt(np.sum((weighted_matrix - ideal_best) ** 2, axis=1))
    dist_worst = np.sqrt(np.sum((weighted_matrix - ideal_worst) ** 2, axis=1))

    # 计算相对贴近度
    closeness = dist_worst / (dist_best + dist_worst)

    return closeness, weighted_matrix


# 5. 生成雷达图
def create_radar_charts(normalized_data, direction):
    # 方向调整
    direction_adjusted = normalized_data.copy()
    for i in range(len(direction)):
        if direction[i] == -1:  # 负向指标
            direction_adjusted[:, i] = 1 - normalized_data[:, i]

    # 应用二次映射：将0和1映射成0.2和0.8
    mapped_data = 0.2 + direction_adjusted * 0.6

    # 分类别数据
    categories = ['市场规模', '流动性', '合规性', '风险']
    category_indices = [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11]]

    # 雷达图1：按大类别的竞争力
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='polar')

    # 计算每个类别的平均得分
    usdt_category_scores = []
    usdc_category_scores = []

    for indices in category_indices:
        usdt_category_scores.append(np.mean(mapped_data[0, indices]))
        usdc_category_scores.append(np.mean(mapped_data[1, indices]))

    # 准备雷达图数据
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形

    # USDT数据
    usdt_values = usdt_category_scores.copy()
    usdt_values += usdt_values[:1]

    # USDC数据
    usdc_values = usdc_category_scores.copy()
    usdc_values += usdc_values[:1]

    # 绘制雷达图
    ax1.plot(angles, usdt_values, 'o-', linewidth=2, label='USDT', color='#FF6B6B')
    ax1.fill(angles, usdt_values, alpha=0.25, color='#FF6B6B')

    ax1.plot(angles, usdc_values, 'o-', linewidth=2, label='USDC', color='#4ECDC4')
    ax1.fill(angles, usdc_values, alpha=0.25, color='#4ECDC4')

    # 设置角度标签
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories, fontsize=10)

    # 设置径向标签
    ax1.set_ylim(0, 1)
    ax1.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax1.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], fontsize=8)
    ax1.grid(True)

    ax1.set_title('稳定币竞争力雷达图（按类别）', size=14, fontweight='bold')
    ax1.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('竞争力雷达图_类别.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 雷达图2：按具体维度的竞争力
    fig2 = plt.figure(figsize=(12, 10))
    ax2 = fig2.add_subplot(111, projection='polar')

    # 准备雷达图数据
    angles = np.linspace(0, 2 * np.pi, len(indicators), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形

    # USDT数据
    usdt_values = mapped_data[0, :].tolist()
    usdt_values += usdt_values[:1]

    # USDC数据
    usdc_values = mapped_data[1, :].tolist()
    usdc_values += usdc_values[:1]

    # 绘制雷达图
    ax2.plot(angles, usdt_values, 'o-', linewidth=2, label='USDT', color='#FF6B6B')
    ax2.fill(angles, usdt_values, alpha=0.25, color='#FF6B6B')

    ax2.plot(angles, usdc_values, 'o-', linewidth=2, label='USDC', color='#4ECDC4')
    ax2.fill(angles, usdc_values, alpha=0.25, color='#4ECDC4')

    # 设置角度标签
    ax2.set_xticks(angles[:-1])

    # 缩短指标名称以便在雷达图上显示
    short_indicators = []
    for indicator in indicators:
        if len(indicator) > 6:
            short_indicators.append(indicator[:6] + '...')
        else:
            short_indicators.append(indicator)

    ax2.set_xticklabels(short_indicators, fontsize=8)

    # 设置径向标签
    ax2.set_ylim(0, 1)
    ax2.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax2.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], fontsize=8)
    ax2.grid(True)

    ax2.set_title('稳定币竞争力雷达图（按具体维度）', size=14, fontweight='bold')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('竞争力雷达图_维度.png', dpi=300, bbox_inches='tight')
    plt.close()


# 主程序
def main():
    # 生成对比图像
    create_comparison_charts()
    print("对比图像已生成并保存")

    # 数据标准化
    normalized_data = minmax_normalization(data)
    print("\n标准化后的数据:")
    for i, indicator in enumerate(indicators):
        print(f"{indicator}: USDT={normalized_data[0, i]:.4f}, USDC={normalized_data[1, i]:.4f}")

    # 熵值法计算权重
    weights = entropy_weight(normalized_data)
    print("\n各指标权重:")
    for i, (indicator, weight) in enumerate(zip(indicators, weights)):
        print(f"{indicator}: {weight:.4f}")

    # TOPSIS分析
    closeness, weighted_matrix = topsis_analysis(normalized_data, weights, direction)
    print("\nTOPSIS分析结果:")
    print(f"USDT相对贴近度: {closeness[0]:.4f}")
    print(f"USDC相对贴近度: {closeness[1]:.4f}")

    if closeness[0] > closeness[1]:
        print("USDT综合竞争力更强")
    else:
        print("USDC综合竞争力更强")

    # 生成雷达图
    create_radar_charts(normalized_data, direction)
    print("竞争力雷达图已生成并保存")

    # 优势领域分析
    usdt_strengths = []
    usdc_strengths = []

    for i in range(len(indicators)):
        if direction[i] == 1:  # 正向指标
            if data[0, i] > data[1, i]:
                usdt_strengths.append(i)
            else:
                usdc_strengths.append(i)
        else:  # 负向指标
            if data[0, i] < data[1, i]:
                usdt_strengths.append(i)
            else:
                usdc_strengths.append(i)

    print('\n优势领域分析:')
    print(f'USDT 优势领域 ({len(usdt_strengths)}个):')
    for idx in usdt_strengths:
        if units[idx] == '%':
            print(f'  - {indicators[idx]}: {data[0, idx]:.2f}% vs {data[1, idx]:.2f}%')
        else:
            print(f'  - {indicators[idx]}: {data[0, idx]:.2f} {units[idx]} vs {data[1, idx]:.2f} {units[idx]}')

    print(f'USDC 优势领域 ({len(usdc_strengths)}个):')
    for idx in usdc_strengths:
        if units[idx] == '%':
            print(f'  - {indicators[idx]}: {data[1, idx]:.2f}% vs {data[0, idx]:.2f}%')
        else:
            print(f'  - {indicators[idx]}: {data[1, idx]:.2f} {units[idx]} vs {data[0, idx]:.2f} {units[idx]}')


if __name__ == "__main__":
    main()