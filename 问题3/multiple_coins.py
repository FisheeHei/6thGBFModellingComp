import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import r2_score
import warnings
import os

warnings.filterwarnings('ignore')
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建文件夹
folder_name = "multiple_coins"
os.makedirs(folder_name, exist_ok=True)


# 创建数据框
data = {
    '年份': [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
    'USDT': [0.001, 0.031, 0.25, 2.2, 4.6, 18.8, 78.3, 67.5, 91.4, 119, 183],
    'EURS': [0, 0, 0, 0.05, 0.1, 0.2, 0.5, 1.24, 1.24, 1.24, 1.24],
    'JPYC': [0, 0, 0, 0, 0, 0.03, 0.12, 0.33, 0.5, 0.9, 1.3],
    '全球贸易总额': [160000, 155000, 173000, 195000, 190000, 176000, 223000, 249000, 240000, 246000, 245000],
    'USD/EUR波动率': [10.2, 8.7, 7.9, 7.5, 6.8, 7.2, 7.1, 9.3, 8.5, 7.8, 8.1],
    'USD/JPY波动率': [11.5, 10.8, 8.2, 9.1, 7.6, 8.9, 8.4, 12.1, 10.2, 9.5, 9.8],
    'USD/HKD波动率': [0.3, 0.2, 0.4, 0.3, 0.2, 0.5, 0.3, 0.4, 0.2, 0.3, 0.2],
    '全球GDP增长率': [2.8, 2.6, 3.3, 3.4, 2.5, -3.1, 6.0, 3.1, 3.0, 3.2, 3.2],
    'CBDC进度US': [10, 10, 15, 15, 20, 20, 25, 25, 30, 30, 0],
    'CBDC进度EU': [10, 10, 15, 15, 20, 20, 25, 30, 60, 80, 90],
    'SWIFT_EUR使用率': [33, 32, 34, 35, 34, 35, 33, 30, 28, 22, 22],
    '全球稳定币监管事件数': [1, 1, 2, 2, 3, 4, 5, 7, 10, 15, 25],
    'DeFi_TVL占比': [0, 0, 0, 5, 10, 20, 30, 35, 38, 40, 40],
    '稳定币在总加密市场占比': [0.1, 0.2, 0.5, 1, 1.5, 2.5, 4, 6, 7, 8, 9]
}

df = pd.DataFrame(data)

# 1. 多元线性回归分析影响因素（优化自变量选择）
print("=" * 60)
print("多元线性回归分析 - 稳定币需求影响因素")
print("=" * 60)

# 根据各稳定币特点选择更合适的自变量
X_vars = {
    'USDT': ['全球贸易总额', '全球GDP增长率', '稳定币在总加密市场占比', 'DeFi_TVL占比'],
    'EURS': ['USD/EUR波动率', 'SWIFT_EUR使用率', 'CBDC进度EU', '全球稳定币监管事件数'],
    'JPYC': ['USD/JPY波动率', '全球GDP增长率', '全球稳定币监管事件数', 'DeFi_TVL占比']
}

mlr_results = {}

for currency in ['USDT', 'EURS', 'JPYC']:
    print(f"\n{currency} 稳定币影响因素分析:")
    print("-" * 40)

    X = df[X_vars[currency]]
    y = df[currency]

    # 训练多元线性回归模型
    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    mlr_results[currency] = {
        'model': model,
        'r2': r2,
        'coefficients': dict(zip(X_vars[currency], model.coef_)),
        'intercept': model.intercept_
    }

    print(f"R²分数: {r2:.4f}")
    print("系数:")
    for var, coef in zip(X_vars[currency], model.coef_):
        print(f"  {var}: {coef:.6f}")
    print(f"截距: {model.intercept_:.6f}")

# 2. 指数平滑预测未来5年
print("\n" + "=" * 60)
print("指数平滑预测 - 未来5年稳定币数量")
print("=" * 60)

# 准备预测数据
years = df['年份'].values
future_years = np.array([2026, 2027, 2028, 2029, 2030])

# 存储预测结果
predictions = {}

for currency in ['USDT', 'EURS', 'JPYC']:
    print(f"\n{currency} 预测结果:")
    print("-" * 30)

    # 使用指数平滑模型
    model_ets = ExponentialSmoothing(df[currency], trend='add', seasonal=None)
    fitted_model = model_ets.fit()

    # 预测未来5年
    forecast = fitted_model.forecast(5)

    predictions[currency] = {
        'historical': df[currency].values,
        'forecast': forecast.values,
        'model': fitted_model
    }

    # 输出预测数值
    for i, year in enumerate(future_years):
        print(f"{year}年: {forecast.values[i]:.2f}亿枚")

    # 计算增长率
    historical_growth = []
    for i in range(1, len(df[currency])):
        growth = (df[currency].iloc[i] - df[currency].iloc[i - 1]) / df[currency].iloc[i - 1] * 100
        historical_growth.append(growth)

    forecast_growth = []
    for i in range(len(forecast)):
        if i == 0:
            base = df[currency].iloc[-1]
        else:
            base = forecast.values[i - 1]
        growth = (forecast.values[i] - base) / base * 100
        forecast_growth.append(growth)

    predictions[currency]['growth_rates'] = {
        'historical': historical_growth,
        'forecast': forecast_growth
    }

# 3. 绘制预测图表并保存
print("\n" + "=" * 60)
print("生成并保存预测图表")
print("=" * 60)

# 创建子图
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('稳定币数量预测（2026-2030）', fontsize=16, fontweight='bold')

# 美元稳定币预测
axes[0, 0].plot(years, predictions['USDT']['historical'], 'bo-', label='历史数据', linewidth=2)
axes[0, 0].plot(future_years, predictions['USDT']['forecast'], 'ro--', label='预测数据', linewidth=2)
axes[0, 0].set_title('USDT（美元挂钩）稳定币预测', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('年份')
axes[0, 0].set_ylabel('数量（亿枚）')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 欧元稳定币预测
axes[0, 1].plot(years, predictions['EURS']['historical'], 'go-', label='历史数据', linewidth=2)
axes[0, 1].plot(future_years, predictions['EURS']['forecast'], 'ro--', label='预测数据', linewidth=2)
axes[0, 1].set_title('EURS（欧元挂钩）稳定币预测', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('年份')
axes[0, 1].set_ylabel('数量（亿枚）')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 日元稳定币预测
axes[1, 0].plot(years, predictions['JPYC']['historical'], 'mo-', label='历史数据', linewidth=2)
axes[1, 0].plot(future_years, predictions['JPYC']['forecast'], 'ro--', label='预测数据', linewidth=2)
axes[1, 0].set_title('JPYC（日元挂钩）稳定币预测', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('年份')
axes[1, 0].set_ylabel('数量（亿枚）')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 市场份额预测
usdt_historical = df['USDT'].values
eurs_historical = df['EURS'].values
jpyc_historical = df['JPYC'].values

total_historical = usdt_historical + eurs_historical + jpyc_historical
usdt_share_historical = (usdt_historical / total_historical) * 100
non_usd_share_historical = ((eurs_historical + jpyc_historical) / total_historical) * 100

# 计算未来市场份额
usdt_forecast = predictions['USDT']['forecast']
eurs_forecast = predictions['EURS']['forecast']
jpyc_forecast = predictions['JPYC']['forecast']

total_forecast = usdt_forecast + eurs_forecast + jpyc_forecast
usdt_share_forecast = (usdt_forecast / total_forecast) * 100
non_usd_share_forecast = ((eurs_forecast + jpyc_forecast) / total_forecast) * 100

# 合并历史和新数据
all_years = np.concatenate([years, future_years])
usdt_share_all = np.concatenate([usdt_share_historical, usdt_share_forecast])
non_usd_share_all = np.concatenate([non_usd_share_historical, non_usd_share_forecast])

axes[1, 1].plot(all_years, usdt_share_all, 'b-', label='美元稳定币份额', linewidth=2)
axes[1, 1].plot(all_years, non_usd_share_all, 'r-', label='非美元稳定币份额', linewidth=2)
axes[1, 1].axvline(x=2025, color='gray', linestyle='--', alpha=0.7, label='预测开始')
axes[1, 1].set_title('美元vs非美元稳定币市场份额', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('年份')
axes[1, 1].set_ylabel('市场份额 (%)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(folder_name, "stablecoin_predictions.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"预测图表已保存为: {fig_path}")

# 4. 增长趋势分析并保存
print("\n" + "=" * 60)
print("生成并保存增长趋势图表")
print("=" * 60)

# 绘制增长趋势图
plt.figure(figsize=(14, 10))

# 历史增长率和预测增长率
growth_years_historical = years[1:]
growth_years_forecast = future_years

plt.subplot(2, 1, 1)
plt.plot(growth_years_historical, predictions['USDT']['growth_rates']['historical'], 'bo-', label='USDT历史增长率',
         linewidth=2)
plt.plot(growth_years_forecast, predictions['USDT']['growth_rates']['forecast'], 'ro--', label='USDT预测增长率',
         linewidth=2)
plt.title('USDT（美元挂钩）稳定币增长率趋势', fontsize=14, fontweight='bold')
plt.xlabel('年份')
plt.ylabel('增长率 (%)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(growth_years_historical, predictions['EURS']['growth_rates']['historical'], 'go-', label='EURS历史增长率',
         linewidth=2)
plt.plot(growth_years_forecast, predictions['EURS']['growth_rates']['forecast'], 'ro--', label='EURS预测增长率',
         linewidth=2)
plt.plot(growth_years_historical, predictions['JPYC']['growth_rates']['historical'], 'mo-', label='JPYC历史增长率',
         linewidth=2)
plt.plot(growth_years_forecast, predictions['JPYC']['growth_rates']['forecast'], 'co--', label='JPYC预测增长率',
         linewidth=2)
plt.title('非美元稳定币增长率趋势', fontsize=14, fontweight='bold')
plt.xlabel('年份')
plt.ylabel('增长率 (%)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
trend_path = os.path.join(folder_name, "growth_trends.png")
plt.savefig(trend_path, dpi=300, bbox_inches='tight')
print(f"增长趋势图表已保存为: {trend_path}")

# 5. 输出详细预测数值到文件
print("\n" + "=" * 60)
print("保存详细预测数值")
print("=" * 60)

# 保存预测结果到CSV文件
forecast_data = []
for i, year in enumerate(future_years):
    forecast_data.append({
        '年份': year,
        'USDT_数量(亿枚)': round(predictions['USDT']['forecast'][i], 2),
        'USDT_增长率(%)': round(predictions['USDT']['growth_rates']['forecast'][i], 1),
        'EURS_数量(亿枚)': round(predictions['EURS']['forecast'][i], 2),
        'EURS_增长率(%)': round(predictions['EURS']['growth_rates']['forecast'][i], 1),
        'JPYC_数量(亿枚)': round(predictions['JPYC']['forecast'][i], 2),
        'JPYC_增长率(%)': round(predictions['JPYC']['growth_rates']['forecast'][i], 1)
    })

forecast_df = pd.DataFrame(forecast_data)
csv_path = os.path.join(folder_name, "forecast_results.csv")
forecast_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"预测数值已保存为: {csv_path}")

# 6. 保存市场份额分析
market_share_data = []
for i, year in enumerate(all_years):
    if year <= 2025:
        idx = year - 2015
        market_share_data.append({
            '年份': year,
            '美元稳定币份额(%)': round(usdt_share_historical[idx], 1) if idx < len(usdt_share_historical) else None,
            '非美元稳定币份额(%)': round(non_usd_share_historical[idx], 1) if idx < len(
                non_usd_share_historical) else None
        })
    else:
        idx = year - 2026
        market_share_data.append({
            '年份': year,
            '美元稳定币份额(%)': round(usdt_share_forecast[idx], 1),
            '非美元稳定币份额(%)': round(non_usd_share_forecast[idx], 1)
        })

market_share_df = pd.DataFrame(market_share_data)
market_share_path = os.path.join(folder_name, "market_share_analysis.csv")
market_share_df.to_csv(market_share_path, index=False, encoding='utf-8-sig')
print(f"市场份额分析已保存为: {market_share_path}")

# 7. 保存影响因素分析结果
analysis_results = []
for currency in ['USDT', 'EURS', 'JPYC']:
    analysis_results.append({
        '稳定币类型': currency,
        'R2分数': round(mlr_results[currency]['r2'], 4),
        '截距': round(mlr_results[currency]['intercept'], 6)
    })

    # 添加系数信息
    for var, coef in mlr_results[currency]['coefficients'].items():
        analysis_results.append({
            '稳定币类型': f"{currency}_{var}",
            '系数': round(coef, 6)
        })

analysis_df = pd.DataFrame(analysis_results)
analysis_path = os.path.join(folder_name, "factor_analysis.csv")
analysis_df.to_csv(analysis_path, index=False, encoding='utf-8-sig')
print(f"影响因素分析已保存为: {analysis_path}")

# 8. 生成综合分析报告
report_content = """
稳定币发展趋势综合分析报告

1. 多元线性回归分析结果

USDT（美元挂钩）稳定币影响因素:
- 全球贸易总额: 正向影响
- 全球GDP增长率: 正向影响
- 稳定币在加密市场占比: 强烈正向影响
- DeFi TVL占比: 正向影响
- R²分数: {:.4f}

EURS（欧元挂钩）稳定币影响因素:
- USD/EUR波动率: 正向影响
- SWIFT EUR使用率: 负向影响
- CBDC进度EU: 正向影响
- 全球稳定币监管事件数: 正向影响
- R²分数: {:.4f}

JPYC（日元挂钩）稳定币影响因素:
- USD/JPY波动率: 正向影响
- 全球GDP增长率: 正向影响
- 全球稳定币监管事件数: 正向影响
- DeFi TVL占比: 正向影响
- R²分数: {:.4f}

2. 未来5年预测结果

美元稳定币（USDT）预测:
- 2026年: {:.2f}亿枚
- 2030年: {:.2f}亿枚

非美元稳定币预测:
- EURS: 2026年{:.2f}亿枚, 2030年{:.2f}亿枚
- JPYC: 2026年{:.2f}亿枚, 2030年{:.2f}亿枚

3. 市场份额变化

2025年市场份额:
- 美元稳定币: {:.1f}%
- 非美元稳定币: {:.1f}%

2030年预测市场份额:
- 美元稳定币: {:.1f}%
- 非美元稳定币: {:.1f}%

市场份额变化 (2025-2030):
- 美元稳定币: {:+.1f}%
- 非美元稳定币: {:+.1f}%

4. 关键趋势分析

- 美元稳定币主导地位稳固但相对下降
- 非美元稳定币快速增长
- 影响因素差异化明显
- 香港等地区政策推动非美元稳定币发展
""".format(
    mlr_results['USDT']['r2'],
    mlr_results['EURS']['r2'],
    mlr_results['JPYC']['r2'],
    predictions['USDT']['forecast'][0],
    predictions['USDT']['forecast'][4],
    predictions['EURS']['forecast'][0],
    predictions['EURS']['forecast'][4],
    predictions['JPYC']['forecast'][0],
    predictions['JPYC']['forecast'][4],
    usdt_share_historical[-1],
    non_usd_share_historical[-1],
    usdt_share_forecast[-1],
    non_usd_share_forecast[-1],
    usdt_share_forecast[-1] - usdt_share_historical[-1],
    non_usd_share_forecast[-1] - non_usd_share_historical[-1]
)

report_path = os.path.join(folder_name, "comprehensive_analysis.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_content)
print(f"综合分析报告已保存为: {report_path}")

print(f"\n所有文件已保存到 {folder_name} 文件夹中")