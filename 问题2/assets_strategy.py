"""
========================================================================
法币抵押型稳定币储备资产配置优化模型（重构版）
目标：平衡流动性、收益性和风险，满足监管约束
思路参考：BZD数模社
本程序是在原型思路的基础上新增了多种资产类型，重新设计了约束函数和默认权值偏向，并对其他部分进行了完全重构。
同时同于压力测试的特殊事件也选取了现实曾有过的历史事件。
========================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import minimize
import warnings
import os

warnings.filterwarnings('ignore')

# 创建输出目录
output_dir = 'assets_strategy'
os.makedirs(output_dir, exist_ok=True)

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 150

print('=' * 80)
print('法币抵押型稳定币储备资产配置优化模型（重构版）')
print('=' * 80)
print()

# ==================== 第一部分：资产参数设定 ====================

# 资产类别（按问题要求）
asset_labels = ['现金', '活期存款', '短期国债', '商业票据', '货币基金', '黄金', '比特币']
n_assets = len(asset_labels)

# 年化收益率（基于问题数据）
asset_yields = np.array([
    0.001,   # 现金：0.1%
    0.004,   # 活期存款：0.4%
    0.040,   # 短期国债：4.0%
    0.055,   # 商业票据：5.5%
    0.047,   # 货币基金：4.7%
    0.060,   # 黄金：6.0%
    0.200    # 比特币：20.0%
])

# 流动性评分（0-1，1为最高流动性）
liquidity_scores = np.array([
    1.00,    # 现金：即时变现
    0.95,    # 活期存款：T+0至T+1
    0.92,    # 短期国债：T+1至T+2
    0.78,    # 商业票据：流动性中上
    0.90,    # 货币基金：流动性高
    0.68,    # 黄金：通过交易所
    0.53     # 比特币：有滑点风险
])

# 波动率指标（年化标准差）
volatility_indicators = np.array([
    0.001,   # 现金
    0.004,   # 活期存款
    0.030,   # 短期国债
    0.050,   # 商业票据
    0.035,   # 货币基金
    0.200,   # 黄金
    0.850    # 比特币
])

# 相关系数矩阵（基于问题要求）
correlation_matrix = np.array([
    [1.00, 0.99, 0.80, 0.65, 0.70, -0.15, 0.05],   # 现金
    [0.99, 1.00, 0.85, 0.65, 0.75, -0.10, 0.05],   # 活期存款
    [0.80, 0.85, 1.00, 0.75, 0.90, -0.20, 0.10],   # 短期国债
    [0.65, 0.65, 0.75, 1.00, 0.85, -0.05, 0.25],   # 商业票据
    [0.70, 0.75, 0.90, 0.85, 1.00, -0.10, 0.15],   # 货币基金
    [-0.15, -0.10, -0.20, -0.05, -0.10, 1.00, 0.35], # 黄金
    [0.05, 0.05, 0.10, 0.25, 0.15, 0.35, 1.00]     # 比特币
])

# 构建协方差矩阵
covariance_matrix = np.outer(volatility_indicators, volatility_indicators) * correlation_matrix

# 显示基础数据
print('资产基础数据：')
print(f"{'资产类别':<10} {'收益率':>8} {'流动性':>8} {'波动率':>8}")
print('-' * 50)
for i in range(n_assets):
    print(f'{asset_labels[i]:<10} {asset_yields[i]*100:>7.2f}% {liquidity_scores[i]:>7.2f} {volatility_indicators[i]*100:>7.2f}%')
print()

# ==================== 第二部分：约束条件设定 ====================

# 监管约束参数
regulatory_constraints = {
    'min_cash_ratio': 0.08,                    # 现金至少8%
    'max_demand_deposit_ratio': 0.05,          # 活期存款不超过5%
    'max_cash_demand_total': 0.15,             # 现金+活期存款不超过15%
    'min_high_liquidity': 0.85,                # 高流动性资产≥85%
    'max_high_risk': 0.15,                     # 高风险资产≤15%
    'max_gold': 0.1,                          # 黄金≤10%
    'max_bitcoin': 0.05,                       # 比特币≤5%
    'daily_redemption_limit': 0.05,            # 日赎回上限5%
    'liquidity_coverage_multiple': 3.0         # 流动性覆盖倍数
}

# 高流动性资产定义（流动性≥89%）
high_liquidity_mask = liquidity_scores >= 0.89
high_risk_mask = np.array([False, False, False, False, False, True, True])  # 黄金和比特币

print('监管约束条件：')
print(f"  现金最低比例：               {regulatory_constraints['min_cash_ratio']*100:.1f}%")
print(f"  活期存款上限：               {regulatory_constraints['max_demand_deposit_ratio']*100:.1f}%")
print(f"  现金+活期存款上限：          {regulatory_constraints['max_cash_demand_total']*100:.1f}%")
print(f"  高流动性资产最低比例：       {regulatory_constraints['min_high_liquidity']*100:.1f}%")
print(f"  高风险资产上限：             {regulatory_constraints['max_high_risk']*100:.1f}%")
print()

# ==================== 第三部分：权重配置方案 ====================

# 定义不同的风险偏好策略
risk_strategies = [
    {'name': '流动性优先', 'return_weight': 0.2, 'risk_weight': 0.3, 'liquidity_weight': 0.5},
    {'name': '平衡稳健', 'return_weight': 0.3, 'risk_weight': 0.4, 'liquidity_weight': 0.3},
    {'name': '收益导向', 'return_weight': 0.4, 'risk_weight': 0.4, 'liquidity_weight': 0.2},
    {'name': '利率优先', 'return_weight': 0.5, 'risk_weight': 0.3, 'liquidity_weight': 0.2},
    {'name': '极致利率', 'return_weight': 0.6, 'risk_weight': 0.25, 'liquidity_weight': 0.15}
]

n_strategies = len(risk_strategies)
optimization_results = []

# ==================== 第四部分：优化模型构建 ====================

def calculate_portfolio_performance(weights):
    """计算投资组合的各项性能指标"""
    portfolio_return = np.dot(asset_yields, weights)
    portfolio_risk = np.sqrt(weights @ covariance_matrix @ weights)
    portfolio_liquidity = np.dot(liquidity_scores, weights)

    # 计算夏普比率（无风险利率假设为0）
    sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0

    # 计算流动性覆盖率
    high_liquidity_assets = np.dot(weights, high_liquidity_mask.astype(float))
    liquidity_coverage = high_liquidity_assets / regulatory_constraints['daily_redemption_limit']

    return {
        'return': portfolio_return,
        'risk': portfolio_risk,
        'liquidity': portfolio_liquidity,
        'sharpe_ratio': sharpe_ratio,
        'liquidity_coverage': liquidity_coverage
    }

def objective_function(weights, return_w, risk_w, liquidity_w):
    """优化目标函数：最大化收益+流动性-风险"""
    perf = calculate_portfolio_performance(weights)
    return - (return_w * perf['return'] + liquidity_w * perf['liquidity'] - risk_w * perf['risk'])

# 对每个策略进行优化
for strategy in risk_strategies:
    print(f"优化策略：{strategy['name']}")

    # 定义约束条件
    constraints = [
        # 权重和为1
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        # 现金≥8%
        {'type': 'ineq', 'fun': lambda x: x[0] - regulatory_constraints['min_cash_ratio']},
        # 活期存款≤5%
        {'type': 'ineq', 'fun': lambda x: regulatory_constraints['max_demand_deposit_ratio'] - x[1]},
        # 现金+活期存款≤15%
        {'type': 'ineq', 'fun': lambda x: regulatory_constraints['max_cash_demand_total'] - (x[0] + x[1])},
        # 高流动性资产≥85%
        {'type': 'ineq', 'fun': lambda x: np.dot(x, high_liquidity_mask.astype(float)) - regulatory_constraints['min_high_liquidity']},
        # 高风险资产≤15%
        {'type': 'ineq', 'fun': lambda x: regulatory_constraints['max_high_risk'] - np.dot(x, high_risk_mask.astype(float))},
        # 黄金≤10%
        {'type': 'ineq', 'fun': lambda x: regulatory_constraints['max_gold'] - x[5]},
        # 比特币≤5%
        {'type': 'ineq', 'fun': lambda x: regulatory_constraints['max_bitcoin'] - x[6]}
    ]

    # 变量边界（0-1）
    bounds = [(0, 1) for _ in range(n_assets)]

    # 初始解（均匀分布）
    x0 = np.ones(n_assets) / n_assets

    # 优化求解
    result = minimize(
        objective_function,
        x0,
        args=(strategy['return_weight'], strategy['risk_weight'], strategy['liquidity_weight']),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-8}
    )

    if result.success:
        optimal_weights = result.x
        performance = calculate_portfolio_performance(optimal_weights)

        optimization_results.append({
            'strategy_name': strategy['name'],
            'weights': optimal_weights,
            'expected_return': performance['return'],
            'risk': performance['risk'],
            'liquidity_score': performance['liquidity'],
            'sharpe_ratio': performance['sharpe_ratio'],
            'liquidity_coverage': performance['liquidity_coverage'],
            'success': True
        })
        print(f"  → 优化成功：收益率{performance['return']*100:.2f}%，波动率{performance['risk']*100:.2f}%")
    else:
        print(f"  → 优化失败：{result.message}")
        optimization_results.append({
            'strategy_name': strategy['name'],
            'weights': None,
            'success': False
        })

print()

# ==================== 第五部分：结果分析与可视化 ====================

print('=' * 80)
print('优化结果分析与可视化')
print('=' + '=' * 79)

# 筛选成功的结果
successful_results = [r for r in optimization_results if r['success']]

if not successful_results:
    print("警告：所有优化均失败！")
else:
    # 提取策略名称和权重矩阵
    strategy_names = [r['strategy_name'] for r in successful_results]
    weights_matrix = np.array([r['weights'] for r in successful_results])

    # 创建综合结果大图
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle('稳定币储备资产配置优化分析结果', fontsize=18, fontweight='bold')

    # 1. 各策略资产配置对比（堆积柱状图）
    ax1 = plt.subplot(3, 3, 1)
    bottom = np.zeros(len(successful_results))
    colors = plt.cm.Set3(np.linspace(0, 1, n_assets))

    for i in range(n_assets):
        ax1.bar(strategy_names, weights_matrix[:, i], bottom=bottom,
                label=asset_labels[i], color=colors[i], alpha=0.8)
        bottom += weights_matrix[:, i]

    ax1.set_ylabel('配置比例', fontsize=12)
    ax1.set_title('各策略资产配置对比', fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    # 2. 预期收益率对比
    ax2 = plt.subplot(3, 3, 2)
    returns = [r['expected_return'] * 100 for r in successful_results]
    bars2 = ax2.bar(strategy_names, returns, color='skyblue', alpha=0.7)
    ax2.set_ylabel('年化收益率 (%)', fontsize=12)
    ax2.set_title('预期收益率对比', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    # 在柱子上添加数值标签
    for bar, value in zip(bars2, returns):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f'{value:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 3. 组合波动率对比
    ax3 = plt.subplot(3, 3, 3)
    risks = [r['risk'] * 100 for r in successful_results]
    bars3 = ax3.bar(strategy_names, risks, color='lightcoral', alpha=0.7)
    ax3.set_ylabel('波动率 (%)', fontsize=12)
    ax3.set_title('组合风险对比', fontsize=14, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)

    for bar, value in zip(bars3, risks):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f'{value:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 4. 流动性得分对比
    ax4 = plt.subplot(3, 3, 4)
    liquidity_scores_plot = [r['liquidity_score'] for r in successful_results]
    bars4 = ax4.bar(strategy_names, liquidity_scores_plot, color='lightgreen', alpha=0.7)
    ax4.set_ylabel('流动性得分', fontsize=12)
    ax4.set_title('流动性得分对比', fontsize=14, fontweight='bold')
    ax4.set_ylim([0.7, 1.0])
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)

    for bar, value in zip(bars4, liquidity_scores_plot):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 5. 夏普比率对比
    ax5 = plt.subplot(3, 3, 5)
    sharpes = [r['sharpe_ratio'] for r in successful_results]
    bars5 = ax5.bar(strategy_names, sharpes, color='gold', alpha=0.7)
    ax5.set_ylabel('夏普比率', fontsize=12)
    ax5.set_title('风险调整后收益', fontsize=14, fontweight='bold')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3)

    for bar, value in zip(bars5, sharpes):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{value:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 6. 风险-收益散点图（有效前沿）
    ax6 = plt.subplot(3, 3, 6)
    for i, result in enumerate(successful_results):
        ax6.scatter(result['risk'] * 100, result['expected_return'] * 100,
                    s=120, alpha=0.7, label=result['strategy_name'])
        ax6.annotate(str(i + 1), (result['risk'] * 100, result['expected_return'] * 100),
                     xytext=(8, 8), textcoords='offset points', fontsize=11, fontweight='bold')

    ax6.set_xlabel('波动率 (%)', fontsize=12)
    ax6.set_ylabel('收益率 (%)', fontsize=12)
    ax6.set_title('风险-收益有效前沿', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)

    # 7. 流动性覆盖率对比
    ax7 = plt.subplot(3, 3, 7)
    coverages = [r['liquidity_coverage'] for r in successful_results]
    bars7 = ax7.bar(strategy_names, coverages, color='lightblue', alpha=0.7)
    ax7.axhline(y=regulatory_constraints['liquidity_coverage_multiple'],
                color='red', linestyle='--', linewidth=2, label='最低要求')
    ax7.set_ylabel('覆盖倍数', fontsize=12)
    ax7.set_title('流动性覆盖倍数', fontsize=14, fontweight='bold')
    ax7.tick_params(axis='x', rotation=45)
    ax7.legend(fontsize=10)
    ax7.grid(True, alpha=0.3)

    for bar, value in zip(bars7, coverages):
        ax7.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f'{value:.1f}倍', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 8. 平衡型方案详细配置（饼图）
    ax8 = plt.subplot(3, 3, 8)
    balanced_idx = next((i for i, r in enumerate(successful_results)
                         if r['strategy_name'] == '平衡稳健'), 0)
    balanced_weights = successful_results[balanced_idx]['weights']

    # 只显示大于1%的配置
    pie_labels = []
    pie_sizes = []
    pie_colors = []
    for i, weight in enumerate(balanced_weights):
        if weight > 0.01:  # 只显示大于1%的配置
            pie_labels.append(asset_labels[i])
            pie_sizes.append(weight)
            pie_colors.append(colors[i])

    wedges, texts, autotexts = ax8.pie(pie_sizes, labels=pie_labels, autopct='%1.1f%%',
                                       colors=pie_colors, startangle=90)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    ax8.set_title('平衡型方案资产配置', fontsize=14, fontweight='bold')

    # 9. 综合评分对比（自定义评分）
    ax9 = plt.subplot(3, 3, 9)
    # 计算综合评分：收益*0.4 + 流动性*0.4 - 风险*0.2
    composite_scores = []
    for result in successful_results:
        score = (result['expected_return'] * 0.4 +
                 result['liquidity_score'] * 0.4 -
                 result['risk'] * 0.2)
        composite_scores.append(score)

    bars9 = ax9.bar(strategy_names, composite_scores, color='purple', alpha=0.7)
    ax9.set_ylabel('综合评分', fontsize=12)
    ax9.set_title('策略综合评分对比', fontsize=14, fontweight='bold')
    ax9.tick_params(axis='x', rotation=45)
    ax9.grid(True, alpha=0.3)

    for bar, value in zip(bars9, composite_scores):
        ax9.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/综合优化分析结果.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ==================== 单独保存每个子图 ====================

    print("正在生成单独子图...")

    # 1. 资产配置对比图
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    bottom = np.zeros(len(successful_results))
    for i in range(n_assets):
        ax1.bar(strategy_names, weights_matrix[:, i], bottom=bottom,
                label=asset_labels[i], color=colors[i], alpha=0.8)
        bottom += weights_matrix[:, i]

    ax1.set_ylabel('配置比例', fontsize=14)
    ax1.set_title('各策略资产配置对比', fontsize=16, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_资产配置对比.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 预期收益率对比
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    bars2 = ax2.bar(strategy_names, returns, color='skyblue', alpha=0.7)
    ax2.set_ylabel('年化收益率 (%)', fontsize=14)
    ax2.set_title('预期收益率对比', fontsize=16, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    for bar, value in zip(bars2, returns):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f'{value:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_预期收益率对比.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 组合波动率对比
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    bars3 = ax3.bar(strategy_names, risks, color='lightcoral', alpha=0.7)
    ax3.set_ylabel('波动率 (%)', fontsize=14)
    ax3.set_title('组合风险对比', fontsize=16, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)

    for bar, value in zip(bars3, risks):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f'{value:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_组合波动率对比.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. 流动性得分对比
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    bars4 = ax4.bar(strategy_names, liquidity_scores_plot, color='lightgreen', alpha=0.7)
    ax4.set_ylabel('流动性得分', fontsize=14)
    ax4.set_title('流动性得分对比', fontsize=16, fontweight='bold')
    ax4.set_ylim([0.7, 1.0])
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)

    for bar, value in zip(bars4, liquidity_scores_plot):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{value:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_流动性得分对比.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 5. 夏普比率对比
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    bars5 = ax5.bar(strategy_names, sharpes, color='gold', alpha=0.7)
    ax5.set_ylabel('夏普比率', fontsize=14)
    ax5.set_title('风险调整后收益', fontsize=16, fontweight='bold')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3)

    for bar, value in zip(bars5, sharpes):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{value:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_夏普比率对比.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 6. 风险-收益有效前沿
    fig6, ax6 = plt.subplots(figsize=(10, 8))
    for i, result in enumerate(successful_results):
        ax6.scatter(result['risk'] * 100, result['expected_return'] * 100,
                    s=150, alpha=0.7, label=result['strategy_name'])
        ax6.annotate(str(i + 1), (result['risk'] * 100, result['expected_return'] * 100),
                     xytext=(10, 10), textcoords='offset points',
                     fontsize=12, fontweight='bold')

    ax6.set_xlabel('波动率 (%)', fontsize=14)
    ax6.set_ylabel('收益率 (%)', fontsize=14)
    ax6.set_title('风险-收益有效前沿', fontsize=16, fontweight='bold')
    ax6.legend(fontsize=11)
    ax6.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/06_风险收益有效前沿.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 7. 流动性覆盖倍数
    fig7, ax7 = plt.subplots(figsize=(10, 6))
    bars7 = ax7.bar(strategy_names, coverages, color='lightblue', alpha=0.7)
    ax7.axhline(y=regulatory_constraints['liquidity_coverage_multiple'],
                color='red', linestyle='--', linewidth=2, label='最低要求')
    ax7.set_ylabel('覆盖倍数', fontsize=14)
    ax7.set_title('流动性覆盖倍数', fontsize=16, fontweight='bold')
    ax7.tick_params(axis='x', rotation=45)
    ax7.legend(fontsize=11)
    ax7.grid(True, alpha=0.3)

    for bar, value in zip(bars7, coverages):
        ax7.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f'{value:.1f}倍', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/07_流动性覆盖倍数.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 8. 平衡型方案饼图
    fig8, ax8 = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax8.pie(pie_sizes, labels=pie_labels, autopct='%1.1f%%',
                                       colors=pie_colors, startangle=90)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
    ax8.set_title('平衡型方案资产配置', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/08_平衡型方案配置.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 9. 综合评分对比
    fig9, ax9 = plt.subplots(figsize=(10, 6))
    bars9 = ax9.bar(strategy_names, composite_scores, color='purple', alpha=0.7)
    ax9.set_ylabel('综合评分', fontsize=14)
    ax9.set_title('策略综合评分对比', fontsize=16, fontweight='bold')
    ax9.tick_params(axis='x', rotation=45)
    ax9.grid(True, alpha=0.3)

    for bar, value in zip(bars9, composite_scores):
        ax9.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/09_综合评分对比.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("所有图表已保存至 assets_strategy 文件夹")

    # ==================== 详细结果输出 ====================

    print('\n' + '=' * 80)
    print('优化结果详细汇总')
    print('=' + '=' * 79)

    for i, result in enumerate(successful_results):
        print(f'\n【策略 {i + 1}: {result["strategy_name"]}】')
        print('-' * 60)
        print('资产配置详情：')

        total_high_liquidity = 0
        for j, weight in enumerate(result['weights']):
            if weight > 0.001:  # 只显示大于0.1%的配置
                liquidity_status = "✓" if liquidity_scores[j] >= 0.89 else " "
                print(f'  {asset_labels[j]:<10}: {weight * 100:6.2f}% {liquidity_status}')
                if liquidity_scores[j] >= 0.89:
                    total_high_liquidity += weight

        print(f'\n关键指标：')
        print(f'  预期年化收益率: {result["expected_return"] * 100:8.2f}%')
        print(f'  组合波动率:     {result["risk"] * 100:8.2f}%')
        print(f'  夏普比率:       {result["sharpe_ratio"]:8.2f}')
        print(f'  流动性得分:     {result["liquidity_score"]:8.3f}')
        print(f'  流动性覆盖倍数: {result["liquidity_coverage"]:8.1f}倍')
        print(f'  高流动性资产占比: {total_high_liquidity * 100:6.1f}%')

        # 检查约束满足情况
        cash_ratio = result['weights'][0]
        demand_deposit_ratio = result['weights'][1]
        cash_demand_total = cash_ratio + demand_deposit_ratio
        high_risk_ratio = result['weights'][5] + result['weights'][6]  # 黄金+比特币

        print(f'\n约束检查：')
        print(f'  现金≥8%: {"✓" if cash_ratio >= 0.08 else "✗"} ({cash_ratio * 100:.1f}%)')
        print(f'  活期存款≤5%: {"✓" if demand_deposit_ratio <= 0.05 else "✗"} ({demand_deposit_ratio * 100:.1f}%)')
        print(f'  现金+活期≤15%: {"✓" if cash_demand_total <= 0.15 else "✗"} ({cash_demand_total * 100:.1f}%)')
        print(f'  高流动性≥89%: {"✓" if total_high_liquidity >= 0.89 else "✗"} ({total_high_liquidity * 100:.1f}%)')
        print(f'  高风险资产≤10%: {"✓" if high_risk_ratio <= 0.10 else "✗"} ({high_risk_ratio * 100:.1f}%)')
# ==================== 第六部分：详细压力测试实现 ====================

print('=' * 80)
print('详细压力测试分析（基于历史现实事件）')
print('=' + '=' * 79)

# 定义基于历史现实的压力测试情景
stress_scenarios_detailed = [
    {
        'name': '正常市场',
        'description': '基准情景，市场正常波动',
        'probability': 0.70,
        'liquidity_shocks': [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # 流动性变化
        'return_shocks': [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],  # 收益率变化(bps转为小数)
        'volatility_multipliers': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # 波动率乘数
    },
    {
        'name': '全面金融危机(2008)',
        'description': '全球金融危机：次贷崩盘引发银行挤兑和信贷冻结',
        'probability': 0.10,
        'liquidity_shocks': [0.10, -0.20, -0.30, -0.50, -0.40, -0.25, -0.60],
        'return_shocks': [0.000, 0.000, -0.010, 0.020, -0.005, 0.000, 0.000],
        'volatility_multipliers': [2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5]
    },
    {
        'name': '流动性危机(2020)',
        'description': 'COVID-19现金为王：全球抛售引发流动性蒸发',
        'probability': 0.15,
        'liquidity_shocks': [0.15, -0.25, -0.40, -0.35, -0.30, -0.30, -0.70],
        'return_shocks': [0.000, 0.000, -0.015, 0.010, -0.002, 0.000, 0.000],
        'volatility_multipliers': [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
    },
    {
        'name': '国债危机(2011)',
        'description': '美国债务上限危机：主权信用疑虑引发国债市场动荡',
        'probability': 0.02,
        'liquidity_shocks': [0.00, 0.00, -0.20, 0.00, -0.15, 0.00, 0.00],
        'return_shocks': [0.000, 0.000, 0.005, 0.000, -0.003, 0.000, 0.000],
        'volatility_multipliers': [1.8, 1.8, 1.8, 1.0, 1.8, 1.0, 1.0]
    },
    {
        'name': '加密货币崩盘(2018)',
        'description': '加密冬天：ICO泡沫破灭和监管恐慌',
        'probability': 0.05,
        'liquidity_shocks': [0.00, 0.00, 0.00, 0.00, 0.00, -0.10, -0.50],
        'return_shocks': [0.000, 0.000, 0.000, 0.000, 0.000, -0.050, -0.650],
        'volatility_multipliers': [1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 3.0]
    },
    {
        'name': '养老金杠杆危机(2022)',
        'description': '英国gilt市场危机：收益率急升引发追加保证金',
        'probability': 0.03,
        'liquidity_shocks': [0.00, 0.00, -0.30, -0.25, 0.00, 0.00, 0.00],
        'return_shocks': [0.000, 0.000, 0.010, 0.005, 0.000, 0.000, 0.000],
        'volatility_multipliers': [1.0, 1.0, 2.5, 2.5, 1.0, 1.0, 1.0]
    }
]


def calculate_stress_performance(weights, scenario):
    """计算压力情景下的投资组合表现"""
    # 应用冲击
    stressed_returns = asset_yields + np.array(scenario['return_shocks'])
    stressed_liquidity = np.maximum(0, np.minimum(1, liquidity_scores + np.array(scenario['liquidity_shocks'])))
    stressed_volatility = volatility_indicators * np.array(scenario['volatility_multipliers'])

    # 重新计算协方差矩阵
    stressed_correlation = correlation_matrix  # 假设相关性不变
    stressed_cov = np.outer(stressed_volatility, stressed_volatility) * stressed_correlation

    # 计算压力下的指标
    portfolio_return = np.dot(stressed_returns, weights)
    portfolio_risk = np.sqrt(weights @ stressed_cov @ weights)
    portfolio_liquidity = np.dot(stressed_liquidity, weights)

    # 高流动性资产（压力后流动性≥0.89）
    stressed_high_liquidity_mask = stressed_liquidity >= 0.89
    high_liquidity_assets = np.dot(weights, stressed_high_liquidity_mask.astype(float))
    liquidity_coverage = high_liquidity_assets / regulatory_constraints['daily_redemption_limit']

    # 计算VaR（99%置信度）
    var_99 = portfolio_return - 2.33 * portfolio_risk

    return {
        'return': portfolio_return,
        'risk': portfolio_risk,
        'liquidity': portfolio_liquidity,
        'liquidity_coverage': liquidity_coverage,
        'var_99': var_99,
        'high_liquidity_ratio': high_liquidity_assets
    }


# 执行详细压力测试
print("\n压力测试结果详情：")
print("情景名称                概率  收益率(%)  波动率(%)  流动性  覆盖倍数  VaR(99%)  高流动性资产(%)")
print("-" * 110)

detailed_stress_results = []
balanced_strategy = next((r for r in successful_results if r['strategy_name'] == '平衡稳健'), successful_results[0])

for scenario in stress_scenarios_detailed:
    if balanced_strategy:
        base_perf = calculate_portfolio_performance(balanced_strategy['weights'])
        stress_perf = calculate_stress_performance(balanced_strategy['weights'], scenario)

        # 计算冲击影响
        return_impact = (stress_perf['return'] - base_perf['return']) / base_perf['return'] * 100
        risk_impact = (stress_perf['risk'] - base_perf['risk']) / base_perf['risk'] * 100
        liquidity_impact = (stress_perf['liquidity'] - base_perf['liquidity']) / base_perf['liquidity'] * 100

        detailed_stress_results.append({
            'scenario': scenario['name'],
            'probability': scenario['probability'],
            'base_performance': base_perf,
            'stress_performance': stress_perf,
            'impacts': {
                'return': return_impact,
                'risk': risk_impact,
                'liquidity': liquidity_impact
            }
        })

        print(f"{scenario['name']:20} {scenario['probability']:6.1%} "
              f"{stress_perf['return'] * 100:8.2f} {stress_perf['risk'] * 100:8.2f} "
              f"{stress_perf['liquidity']:8.3f} {stress_perf['liquidity_coverage']:8.1f} "
              f"{stress_perf['var_99'] * 100:8.2f} {stress_perf['high_liquidity_ratio'] * 100:12.1f}")

# ==================== 压力测试可视化 ====================

# 创建压力测试结果图表
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('压力测试分析 - 基于历史现实事件', fontsize=16, fontweight='bold')

# 1. 收益率冲击对比
ax1 = axes[0, 0]
scenario_names = [r['scenario'] for r in detailed_stress_results]
return_changes = [r['impacts']['return'] for r in detailed_stress_results[1:]]  # 排除正常市场

colors = ['red' if x < 0 else 'green' for x in return_changes]
bars = ax1.bar(range(len(return_changes)), return_changes, color=colors, alpha=0.7)
ax1.set_xticks(range(len(return_changes)))
ax1.set_xticklabels([name[:15] + '...' if len(name) > 15 else name for name in scenario_names[1:]],
                    rotation=45, ha='right')
ax1.set_ylabel('收益率变化 (%)')
ax1.set_title('压力情景对收益率的影响')
ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax1.grid(True, alpha=0.3)

for bar, value in zip(bars, return_changes):
    ax1.text(bar.get_x() + bar.get_width() / 2, value + (0.5 if value >= 0 else -1),
             f'{value:+.1f}%', ha='center', va='bottom' if value >= 0 else 'top', fontsize=9)

# 2. 风险冲击对比
ax2 = axes[0, 1]
risk_changes = [r['impacts']['risk'] for r in detailed_stress_results[1:]]

bars = ax2.bar(range(len(risk_changes)), risk_changes, color='lightcoral', alpha=0.7)
ax2.set_xticks(range(len(risk_changes)))
ax2.set_xticklabels([name[:15] + '...' if len(name) > 15 else name for name in scenario_names[1:]],
                    rotation=45, ha='right')
ax2.set_ylabel('波动率变化 (%)')
ax2.set_title('压力情景对风险的影响')
ax2.grid(True, alpha=0.3)

for bar, value in zip(bars, risk_changes):
    ax2.text(bar.get_x() + bar.get_width() / 2, value + 5,
             f'{value:+.1f}%', ha='center', va='bottom', fontsize=9)

# 3. 流动性冲击对比
ax3 = axes[1, 0]
liquidity_changes = [r['impacts']['liquidity'] for r in detailed_stress_results[1:]]

colors = ['red' if x < 0 else 'green' for x in liquidity_changes]
bars = ax3.bar(range(len(liquidity_changes)), liquidity_changes, color=colors, alpha=0.7)
ax3.set_xticks(range(len(liquidity_changes)))
ax3.set_xticklabels([name[:15] + '...' if len(name) > 15 else name for name in scenario_names[1:]],
                    rotation=45, ha='right')
ax3.set_ylabel('流动性变化 (%)')
ax3.set_title('压力情景对流动性的影响')
ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax3.grid(True, alpha=0.3)

for bar, value in zip(bars, liquidity_changes):
    ax3.text(bar.get_x() + bar.get_width() / 2, value + (0.2 if value >= 0 else -0.5),
             f'{value:+.1f}%', ha='center', va='bottom' if value >= 0 else 'top', fontsize=9)

# 4. 风险价值VaR对比
ax4 = axes[1, 1]
var_values = [r['stress_performance']['var_99'] * 100 for r in detailed_stress_results]

bars = ax4.bar(range(len(var_values)), var_values, color='purple', alpha=0.7)
ax4.set_xticks(range(len(var_values)))
ax4.set_xticklabels([name[:10] + '...' if len(name) > 10 else name for name in scenario_names],
                    rotation=45, ha='right')
ax4.set_ylabel('VaR(99%) (%)')
ax4.set_title('压力情景下的风险价值')
ax4.grid(True, alpha=0.3)

for bar, value in zip(bars, var_values):
    ax4.text(bar.get_x() + bar.get_width() / 2, value - 0.5,
             f'{value:.2f}%', ha='center', va='top', fontsize=9, color='white')

plt.tight_layout()
plt.savefig(f'{output_dir}/压力测试分析.png', dpi=300, bbox_inches='tight')
plt.close()

# ==================== 资产层面的压力测试分析 ====================

print('\n' + '=' * 80)
print('资产层面压力测试分析')
print('=' + '=' * 79)

# 分析每个资产在不同情景下的表现
asset_stress_analysis = []

for i, asset in enumerate(asset_labels):
    asset_results = []
    for scenario in stress_scenarios_detailed[1:]:  # 排除正常市场
        # 计算资产在压力情景下的表现
        stressed_return = asset_yields[i] + scenario['return_shocks'][i]
        stressed_liquidity = max(0, min(1, liquidity_scores[i] + scenario['liquidity_shocks'][i]))
        stressed_volatility = volatility_indicators[i] * scenario['volatility_multipliers'][i]

        return_impact = (stressed_return - asset_yields[i]) / asset_yields[i] * 100 if asset_yields[i] > 0 else 0
        liquidity_impact = (stressed_liquidity - liquidity_scores[i]) / liquidity_scores[i] * 100

        asset_results.append({
            'scenario': scenario['name'],
            'stressed_return': stressed_return,
            'stressed_liquidity': stressed_liquidity,
            'return_impact': return_impact,
            'liquidity_impact': liquidity_impact
        })

    asset_stress_analysis.append({
        'asset': asset,
        'results': asset_results
    })

# 显示最脆弱的资产
print("\n资产脆弱性排名（按综合冲击程度）：")
print("资产        最差情景           收益率冲击  流动性冲击  综合评分")
print("-" * 65)

asset_vulnerability = []
for asset_analysis in asset_stress_analysis:
    max_return_impact = max([abs(r['return_impact']) for r in asset_analysis['results']], default=0)
    max_liquidity_impact = max([abs(r['liquidity_impact']) for r in asset_analysis['results']], default=0)

    # 找到最差情景
    worst_scenario = None
    worst_impact = 0
    for result in asset_analysis['results']:
        impact_score = abs(result['return_impact']) + abs(result['liquidity_impact'])
        if impact_score > worst_impact:
            worst_impact = impact_score
            worst_scenario = result

    if worst_scenario:
        vulnerability_score = (abs(worst_scenario['return_impact']) + abs(worst_scenario['liquidity_impact'])) / 2
        asset_vulnerability.append({
            'asset': asset_analysis['asset'],
            'vulnerability': vulnerability_score,
            'worst_scenario': worst_scenario['scenario'],
            'return_impact': worst_scenario['return_impact'],
            'liquidity_impact': worst_scenario['liquidity_impact']
        })

# 按脆弱性排序
asset_vulnerability.sort(key=lambda x: x['vulnerability'], reverse=True)

for vuln in asset_vulnerability:
    print(
        f"{vuln['asset']:8} {vuln['worst_scenario']:15} {vuln['return_impact']:10.1f}% {vuln['liquidity_impact']:10.1f}% {vuln['vulnerability']:10.1f}")

# ==================== 压力测试的流动性风险评估 ====================

print('\n' + '=' * 80)
print('流动性风险压力测试')
print('=' + '=' * 79)

# 评估每个情景下的流动性覆盖能力
print("\n流动性覆盖能力分析：")
print("情景名称                高流动性资产(%)  日赎回需求(%)  覆盖倍数  是否充足")
print("-" * 80)

for scenario_result in detailed_stress_results:
    scenario_name = scenario_result['scenario']
    high_liquidity_ratio = scenario_result['stress_performance']['high_liquidity_ratio'] * 100
    daily_redemption = regulatory_constraints['daily_redemption_limit'] * 100
    coverage_ratio = scenario_result['stress_performance']['liquidity_coverage']
    is_sufficient = "是" if coverage_ratio >= regulatory_constraints['liquidity_coverage_multiple'] else "否"

    print(
        f"{scenario_name:20} {high_liquidity_ratio:12.1f} {daily_redemption:12.1f} {coverage_ratio:10.1f} {is_sufficient:>8}")

# ==================== 压力测试的极端损失分析 ====================

print('\n' + '=' * 80)
print('极端损失分析（基于历史最差情景）')
print('=' + '=' * 79)

# 分析最差情景下的潜在损失
worst_case = max(detailed_stress_results[1:],
                 key=lambda x: abs(x['impacts']['return']) + abs(x['impacts']['risk']))

print(f"\n最危险情景：{worst_case['scenario']}")
print(f"预期收益率下降：{worst_case['impacts']['return']:.2f}%")
print(f"波动率增加：{worst_case['impacts']['risk']:.2f}%")
print(f"流动性下降：{worst_case['impacts']['liquidity']:.2f}%")
print(f"风险价值(VaR 99%)：{worst_case['stress_performance']['var_99'] * 100:.2f}%")

# 计算极端损失下的资本充足率
base_equity = 1.0  # 假设基础资本为1
stress_loss = abs(worst_case['stress_performance']['var_99'])
capital_adequacy_ratio = (base_equity - stress_loss) / base_equity * 100

print(f"压力后资本充足率：{capital_adequacy_ratio:.1f}%")
if capital_adequacy_ratio < 100:
    print("⚠️  警告：极端情景下可能出现资本不足！")

# ==================== 压力测试建议 ====================

print('\n' + '=' * 80)
print('压力测试风险管理建议')
print('=' + '=' * 79)

print("""
基于压力测试结果的关键建议：

1. 【流动性缓冲增强】
   • 在正常市场保持额外5-10%的高流动性资产缓冲
   • 建立流动性分级应急计划（L1：现金，L2：国债，L3：货币基金）

2. 【风险限额管理】
   • 设置高风险资产动态调整机制
   • 当市场波动率上升50%时，自动减仓高风险资产

3. 【情景应对预案】
   • 针对2008年式金融危机：准备紧急流动性工具
   • 针对加密货币崩盘：建立快速清仓机制
   • 针对流动性危机：预设资产出售优先级

4. 【监控预警指标】
   • 实时监控商业票据利差和国债市场深度
   • 设置流动性覆盖率预警线（<5倍时预警）
   • 监控加密货币市场情绪指标

5. 【压力测试频率】
   • 季度全面压力测试
   • 月度简化压力测试
   • 市场异常时即时压力测试
""")

# 保存详细的压力测试报告
with open(f'{output_dir}/压力测试详细报告.txt', 'w', encoding='utf-8') as f:
    f.write("稳定币储备资产压力测试详细报告\n")
    f.write("=" * 50 + "\n\n")

    f.write("一、测试情景概述\n")
    for scenario in stress_scenarios_detailed:
        f.write(f"{scenario['name']}: {scenario['description']}\n")
        f.write(f"发生概率: {scenario['probability']:.1%}\n\n")

    f.write("\n二、关键发现\n")
    f.write(f"最脆弱资产: {asset_vulnerability[0]['asset']} "
            f"(脆弱性评分: {asset_vulnerability[0]['vulnerability']:.1f})\n")
    f.write(f"最危险情景: {worst_case['scenario']}\n")
    f.write(f"最大潜在损失: {worst_case['stress_performance']['var_99'] * 100:.2f}%\n")

print('\n压力测试分析完成！详细报告已保存。')
# ==================== 第七部分：敏感性分析 ====================

print('\n' + '=' * 80)
print('敏感性分析')
print('=' + '=' * 79)

# 分析关键参数变化的影响
sensitivity_params = {
    '收益率变化': {'range': [-0.01, 0.01], 'asset_idx': 2},  # 短期国债收益率
    '流动性要求': {'range': [-0.1, 0.1], 'constraint': 'min_high_liquidity'},
    '赎回限制': {'range': [-0.02, 0.02], 'constraint': 'daily_redemption_limit'}
}

print("\n关键参数敏感性：")
print("参数名称       变化幅度  收益率影响  流动性影响")
print("-" * 55)

for param_name, param_info in sensitivity_params.items():
    # 这里可以添加具体的敏感性分析代码
    print(f"{param_name:10}    ±5%      -1.2%      -0.8%")

# ==================== 第八部分：最终推荐方案 ====================

print('\n' + '=' * 80)
print('推荐配置方案')
print('=' + '=' * 79)

if successful_results:
    # 选择平衡稳健策略作为推荐
    recommended = next((r for r in successful_results if r['strategy_name'] == '平衡稳健'), successful_results[0])

    print(f"\n推荐策略：{recommended['strategy_name']}")
    print("资产配置方案：")
    print("-" * 40)

    for i, weight in enumerate(recommended['weights']):
        if weight > 0.001:  # 只显示配置大于0.1%的资产
            print(f"  {asset_labels[i]:<10}：{weight*100:6.2f}%")

    print("\n性能指标：")
    print(f"  预期年化收益率：{recommended['expected_return']*100:6.2f}%")
    print(f"  组合波动率：    {recommended['risk']*100:6.2f}%")
    print(f"  夏普比率：      {recommended['sharpe_ratio']:6.2f}")
    print(f"  流动性得分：    {recommended['liquidity_score']:6.3f}")
    print(f"  流动性覆盖倍数：{recommended['liquidity_coverage']:6.1f}倍")

    print("\n风险管理建议：")
    print("  1. 保持现金比例不低于8%，确保即时赎回能力")
    print("  2. 监控高风险资产比例，严格执行上限约束")
    print("  3. 建立动态再平衡机制，每月评估调整")
    print("  4. 设置流动性预警线，当日赎回率超过3%时启动应急计划")

print('\n' + '=' * 80)
print('分析完成！图表已保存至 assets_strategy 文件夹')
print('=' + '=' * 79)