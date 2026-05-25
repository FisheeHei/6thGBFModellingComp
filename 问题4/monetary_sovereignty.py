import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建数据框
data = {
    'year': [2019, 2019, 2019, 2019, 2019, 2019, 2020, 2020, 2020, 2020, 2020, 2020,
             2021, 2021, 2021, 2021, 2021, 2021, 2022, 2022, 2022, 2022, 2022, 2022,
             2023, 2023, 2023, 2023, 2023, 2023, 2024, 2024, 2024, 2024, 2024, 2024,
             2025, 2025, 2025, 2025, 2025, 2025],
    'country': ['US', 'China', 'Argentina', 'Guinea', 'Fiji', 'Japan'] * 7,
    'LCCR': [8.2, 11.0, 24.0, 16.0, 14.0, 19.0, 8.0, 10.5, 21.0, 15.5, 13.8, 18.5,
             7.9, 10.2, 19.0, 14.5, 13.5, 18.2, 7.7, 9.8, 17.0, 13.5, 13.0, 17.8,
             7.6, 9.5, 16.0, 12.5, 12.5, 17.5, 7.5, 9.2, 15.0, 11.5, 12.0, 17.2,
             7.4, 9.0, 14.0, 10.5, 11.5, 17.0],
    'FCDR': [0.4, 1.0, 50.0, 18.0, 4.5, 1.5, 0.3, 1.2, 52.0, 20.0, 4.8, 1.7,
             0.3, 1.5, 55.0, 22.0, 5.0, 1.8, 0.3, 2.0, 58.0, 25.0, 5.5, 2.0,
             0.3, 2.2, 62.0, 28.0, 6.0, 2.2, 0.3, 2.5, 65.0, 30.0, 6.5, 2.5,
             0.3, 2.8, 68.0, 32.0, 7.0, 2.8],
    'KAOPEN': [2.35, -1.19, -1.89, -1.20, 1.50, 1.90, 2.35, -1.19, -1.89, -1.20, 1.50, 1.90,
               2.35, -1.19, -1.89, -1.20, 1.50, 1.90, 2.35, -1.19, -1.89, -1.20, 1.50, 1.90,
               2.35, -1.00, -1.50, -1.00, 1.60, 1.95, 2.35, -1.00, -1.50, -1.00, 1.60, 1.95,
               2.35, -1.00, -1.50, -1.00, 1.60, 1.95],
    'SP': [0.1, 0.0, 0.5, 0.2, 0.1, 0.01, 0.2, 0.0, 1.0, 0.3, 0.15, 0.02,
           0.3, 0.05, 2.0, 0.4, 0.2, 0.03, 0.4, 0.08, 3.5, 0.6, 0.25, 0.04,
           0.45, 0.1, 4.5, 0.7, 0.3, 0.05, 0.5, 0.12, 5.0, 0.8, 0.35, 0.06,
           0.55, 0.15, 5.5, 0.9, 0.4, 0.07],
    'DDP': [100, 60, 65, 45, 48, 62, 100, 59, 68, 47, 49, 61,
            100, 58, 72, 50, 50, 60, 100, 57, 75, 52, 51, 59,
            100, 56, 80, 54, 52, 58, 100, 55, 82, 55, 53, 57,
            100, 54, 85, 56, 54, 56],
    'GDP_per_cap': [65000, 10200, 11000, 1100, 5000, 40000, 63000, 10400, 8500, 1200, 4800, 40000,
                    70000, 12500, 10500, 1300, 4900, 40000, 76000, 12700, 13700, 1200, 5200, 34000,
                    81000, 12600, 13500, 1200, 5300, 33800, 85000, 12500, 13000, 1200, 5400, 33100,
                    87000, 12700, 12800, 1200, 5500, 32800],
    'inflation': [1.8, 2.9, 53.5, 9.1, 3.5, 0.5, 1.2, 2.4, 42.0, 10.8, 0.1, -0.0,
                  4.7, 0.9, 48.4, 12.0, 1.7, -0.2, 8.0, 2.0, 94.8, 11.8, 1.1, 2.5,
                  4.1, 0.2, 211.4, 7.8, 5.1, 3.3, 3.0, 0.2, 276.2, 6.5, 2.5, 2.5,
                  2.5, 1.5, 50.0, 8.0, 3.0, 2.0],
    'remittance_GDP': [0.0, 0.2, 0.3, 1.5, 6.5, 0.1, 0.0, 0.2, 0.3, 1.6, 6.8, 0.1,
                       0.0, 0.2, 0.3, 1.7, 7.0, 0.1, 0.0, 0.2, 0.3, 1.8, 7.2, 0.1,
                       0.0, 0.2, 0.3, 1.9, 7.3, 0.1, 0.0, 0.2, 0.3, 2.0, 7.5, 0.1,
                       0.0, 0.2, 0.3, 2.1, 7.6, 0.1]
}

df = pd.DataFrame(data)

# 添加分类变量
df['country_type'] = df['country'].apply(lambda x: '发达' if x in ['US', 'Japan'] else
'新兴' if x in ['China', 'Argentina'] else '发展中')
df['high_inflation'] = (df['inflation'] > 20).astype(int)

print("数据概览:")
print(df.describe())
print(f"\n国家分布: {df['country'].unique()}")

# 1. 相关性分析
print("\n=== 相关性分析 ===")
corr_matrix = df[['LCCR', 'FCDR', 'KAOPEN', 'SP', 'DDP', 'GDP_per_cap', 'inflation']].corr()
print(corr_matrix.round(3))


# 2. 面板回归模型
def run_panel_regression(df, dependent_var, independent_vars):
    """运行面板回归分析"""
    formula = f"{dependent_var} ~ {independent_vars} + C(country) + C(year)"

    try:
        # 混合效应模型
        model = mixedlm(formula, df, groups=df["country"])
        result = model.fit()
        return result
    except:
        # 如果混合模型失败，使用OLS
        formula_ols = f"{dependent_var} ~ {independent_vars} + C(country) + C(year)"
        model = sm.OLS.from_formula(formula_ols, data=df)
        result = model.fit()
        return result


# 模型1: 稳定币对货币主权的影响
print("\n=== 模型1: 稳定币普及对货币主权的影响 ===")
independent_vars = "SP + GDP_per_cap + inflation + remittance_GDP"

# LCCR作为因变量
result_lccr = run_panel_regression(df, "LCCR", independent_vars)
print("LCCR模型结果:")
print(result_lccr.summary())

# FCDR作为因变量
result_fcdr = run_panel_regression(df, "FCDR", independent_vars)
print("\nFCDR模型结果:")
print(result_fcdr.summary())


# 3. 货币主权风险指数计算
def calculate_sovereignty_risk(df):
    """计算货币主权风险指数"""
    # 标准化指标（0-1，值越高风险越大）
    df['LCCR_risk'] = 1 - (df['LCCR'] - df['LCCR'].min()) / (df['LCCR'].max() - df['LCCR'].min())
    df['FCDR_risk'] = (df['FCDR'] - df['FCDR'].min()) / (df['FCDR'].max() - df['FCDR'].min())
    df['KAOPEN_risk'] = 1 - (df['KAOPEN'] - df['KAOPEN'].min()) / (df['KAOPEN'].max() - df['KAOPEN'].min())

    # 综合风险指数（权重可调整）
    df['sovereignty_risk'] = 0.4 * df['LCCR_risk'] + 0.4 * df['FCDR_risk'] + 0.2 * df['KAOPEN_risk']

    return df


df_risk = calculate_sovereignty_risk(df)


# 4. 风险评估
def assess_sovereignty_risk(df_risk, risk_threshold=0.6):
    """评估货币主权风险"""
    latest_data = df_risk[df_risk['year'] == 2025].copy()

    high_risk_countries = latest_data[latest_data['sovereignty_risk'] > risk_threshold]
    medium_risk_countries = latest_data[(latest_data['sovereignty_risk'] > 0.4) &
                                        (latest_data['sovereignty_risk'] <= risk_threshold)]

    print(f"\n=== 货币主权风险评估 (2025年) ===")
    print(f"高风险国家 (风险指数 > {risk_threshold}):")
    for _, row in high_risk_countries.iterrows():
        print(f"  {row['country']}: 风险指数 {row['sovereignty_risk']:.3f}, SP: {row['SP']}%")

    print(f"\n中等风险国家 (0.4 < 风险指数 <= {risk_threshold}):")
    for _, row in medium_risk_countries.iterrows():
        print(f"  {row['country']}: 风险指数 {row['sovereignty_risk']:.3f}, SP: {row['SP']}%")

    return high_risk_countries, medium_risk_countries


high_risk, medium_risk = assess_sovereignty_risk(df_risk)


# 5. 可视化分析
def create_visualizations(df_risk):
    """创建可视化图表"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 图1: 稳定币普及与货币主权风险的关系
    colors = {'Argentina': 'red', 'Guinea': 'orange', 'Fiji': 'green',
              'China': 'blue', 'US': 'purple', 'Japan': 'brown'}

    for country in df_risk['country'].unique():
        country_data = df_risk[df_risk['country'] == country]
        axes[0, 0].scatter(country_data['SP'], country_data['sovereignty_risk'],
                           c=colors[country], label=country, s=100, alpha=0.7)

    axes[0, 0].set_xlabel('稳定币普及度 (SP, %)')
    axes[0, 0].set_ylabel('货币主权风险指数')
    axes[0, 0].set_title('稳定币普及与货币主权风险关系')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 图2: 各国风险趋势
    for country in df_risk['country'].unique():
        country_data = df_risk[df_risk['country'] == country]
        axes[0, 1].plot(country_data['year'], country_data['sovereignty_risk'],
                        marker='o', label=country, linewidth=2)

    axes[0, 1].set_xlabel('年份')
    axes[0, 1].set_ylabel('货币主权风险指数')
    axes[0, 1].set_title('各国货币主权风险趋势 (2019-2025)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 图3: 通胀与稳定币使用
    axes[1, 0].scatter(df_risk['inflation'], df_risk['SP'], c=df_risk['sovereignty_risk'],
                       cmap='Reds', s=100, alpha=0.7)
    axes[1, 0].set_xlabel('通胀率 (%)')
    axes[1, 0].set_ylabel('稳定币普及度 (SP, %)')
    axes[1, 0].set_title('通胀率与稳定币使用关系')
    axes[1, 0].grid(True, alpha=0.3)

    # 图4: 2025年风险分布
    latest_risk = df_risk[df_risk['year'] == 2025]
    bars = axes[1, 1].bar(latest_risk['country'], latest_risk['sovereignty_risk'],
                          color=['red' if x > 0.6 else 'orange' if x > 0.4 else 'green'
                                 for x in latest_risk['sovereignty_risk']])
    axes[1, 1].set_ylabel('货币主权风险指数')
    axes[1, 1].set_title('2025年各国货币主权风险比较')
    axes[1, 1].tick_params(axis='x', rotation=45)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


create_visualizations(df_risk)


# 6. 预测分析
def predict_sovereignty_trend(df_risk, target_countries, sp_growth_scenarios):
    """预测货币主权趋势"""
    print("\n=== 主权风险预测分析 ===")

    base_2025 = df_risk[df_risk['year'] == 2025].set_index('country')

    for country in target_countries:
        if country not in base_2025.index:
            continue

        base_risk = base_2025.loc[country, 'sovereignty_risk']
        base_sp = base_2025.loc[country, 'SP']

        print(f"\n{country}预测分析 (2025基准: 风险={base_risk:.3f}, SP={base_sp}%):")

        for scenario, sp_growth in sp_growth_scenarios.items():
            # 简化的线性预测模型
            predicted_sp = base_sp * (1 + sp_growth)
            # 基于历史关系的风险变化预测
            risk_increase = 0.15 * sp_growth  # 假设SP每增长100%，风险增加15%
            predicted_risk = min(1.0, base_risk + risk_increase)

            risk_level = "高风险" if predicted_risk > 0.7 else "中等风险" if predicted_risk > 0.5 else "低风险"

            print(f"  {scenario}: SP增长{sp_growth:.0%} → 风险指数{predicted_risk:.3f} ({risk_level})")


# 预测情景
scenarios = {
    "保守情景": 0.5,  # SP增长50%
    "基准情景": 1.0,  # SP增长100%
    "激进情景": 2.0  # SP增长200%
}

target_countries = ['Argentina', 'Guinea', 'Fiji']
predict_sovereignty_trend(df_risk, target_countries, scenarios)

# 7. 政策建议
print("\n=== 政策建议 ===")
print("1. 高风险国家需建立数字货币监管框架，平衡金融创新与主权保护")
print("2. 中等风险国家应加强本币信用建设，控制通胀预期")
print("3. 所有国家需监控稳定币跨境流动，防范资本外逃风险")
print("4. 推动本土数字货币发展，提供官方数字支付替代方案")

# 保存结果
df_risk.to_csv('currency_sovereignty_analysis.csv', index=False)
print(f"\n分析结果已保存至 currency_sovereignty_analysis.csv")