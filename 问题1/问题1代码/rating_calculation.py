import pandas as pd
import numpy as np
from datetime import datetime
import os


def calculate_average_decoupling_rate(file_path, start_date, end_date):
    """
    计算稳定币在指定日期范围内的平均脱钩率

    参数:
    file_path: 数据文件路径
    start_date: 开始日期 (格式: '2025-01-01')
    end_date: 结束日期 (格式: '2025-10-31')

    返回:
    average_decoupling_rate: 平均脱钩率
    data_count: 使用的数据点数量
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"错误: 文件 {file_path} 不存在")
            return None, 0

        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 检查必要的列是否存在
        if 'snapped_at' not in df.columns or 'price' not in df.columns:
            print(f"错误: 文件 {file_path} 缺少必要的列 'snapped_at' 或 'price'")
            return None, 0

        # 确保snapped_at列是datetime类型
        df['snapped_at'] = pd.to_datetime(df['snapped_at'])

        # 过滤指定日期范围的数据
        mask = (df['snapped_at'] >= start_date) & (df['snapped_at'] <= end_date)
        filtered_df = df.loc[mask].copy()  # 使用.copy()避免SettingWithCopyWarning

        if len(filtered_df) == 0:
            print(f"警告: 在 {start_date} 到 {end_date} 范围内没有找到数据")
            return None, 0

        # 计算脱钩率: |price - 1|
        filtered_df['decoupling_rate'] = abs(filtered_df['price'] - 1)

        # 计算平均脱钩率
        average_decoupling_rate = filtered_df['decoupling_rate'].mean()
        data_count = len(filtered_df)

        return average_decoupling_rate, data_count

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return None, 0


def calculate_volume_marketcap_ratio(file_path, start_date, end_date):
    """
    计算稳定币在指定日期范围内的交易量与市值比率

    参数:
    file_path: 数据文件路径
    start_date: 开始日期 (格式: '2025-01-01')
    end_date: 结束日期 (格式: '2025-10-31')

    返回:
    avg_ratio: 平均交易量/市值比率
    data_count: 使用的数据点数量
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"错误: 文件 {file_path} 不存在")
            return None, 0

        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 检查必要的列是否存在
        if 'snapped_at' not in df.columns or 'total_volume' not in df.columns or 'market_cap' not in df.columns:
            print(f"错误: 文件 {file_path} 缺少必要的列 'snapped_at', 'total_volume' 或 'market_cap'")
            return None, 0

        # 确保snapped_at列是datetime类型
        df['snapped_at'] = pd.to_datetime(df['snapped_at'])

        # 过滤指定日期范围的数据
        mask = (df['snapped_at'] >= start_date) & (df['snapped_at'] <= end_date)
        filtered_df = df.loc[mask].copy()

        if len(filtered_df) == 0:
            print(f"警告: 在 {start_date} 到 {end_date} 范围内没有找到数据")
            return None, 0

        # 处理可能的零市值情况
        filtered_df = filtered_df[filtered_df['market_cap'] > 0]

        if len(filtered_df) == 0:
            print(f"警告: 在 {start_date} 到 {end_date} 范围内所有市值为零")
            return None, 0

        # 计算交易量/市值比率
        filtered_df['volume_marketcap_ratio'] = filtered_df['total_volume'] / filtered_df['market_cap']

        # 计算平均比率
        avg_ratio = filtered_df['volume_marketcap_ratio'].mean()
        data_count = len(filtered_df)

        return avg_ratio, data_count

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return None, 0


def calculate_daily_volatility(file_path, start_date, end_date):
    """
    计算稳定币在指定日期范围内的日平均波动率

    参数:
    file_path: 数据文件路径
    start_date: 开始日期 (格式: '2025-01-01')
    end_date: 结束日期 (格式: '2025-10-31')

    返回:
    daily_volatility: 日平均波动率
    data_count: 使用的数据点数量
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"错误: 文件 {file_path} 不存在")
            return None, 0

        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 检查必要的列是否存在
        if 'snapped_at' not in df.columns or 'price' not in df.columns:
            print(f"错误: 文件 {file_path} 缺少必要的列 'snapped_at' 或 'price'")
            return None, 0

        # 确保snapped_at列是datetime类型
        df['snapped_at'] = pd.to_datetime(df['snapped_at'])

        # 过滤指定日期范围的数据
        mask = (df['snapped_at'] >= start_date) & (df['snapped_at'] <= end_date)
        filtered_df = df.loc[mask].copy()

        if len(filtered_df) == 0:
            print(f"警告: 在 {start_date} 到 {end_date} 范围内没有找到数据")
            return None, 0

        # 按日期排序
        filtered_df = filtered_df.sort_values('snapped_at')

        # 计算日收益率（对数收益率）
        filtered_df['daily_return'] = np.log(filtered_df['price'] / filtered_df['price'].shift(1))

        # 移除第一个NaN值
        filtered_df = filtered_df.dropna(subset=['daily_return'])

        if len(filtered_df) == 0:
            print(f"警告: 在 {start_date} 到 {end_date} 范围内无法计算收益率")
            return None, 0

        # 计算日波动率（收益率的标准差）
        daily_volatility = filtered_df['daily_return'].std()
        data_count = len(filtered_df)

        return daily_volatility, data_count

    except Exception as e:
        print(f"计算日波动率时出错: {e}")
        return None, 0


def main():
    # 文件路径 - 根据实际文件名设置
    usdt_file = 'usdt-usd-max.csv'
    usdc_file = 'usdc-usd-max.csv'

    # 日期范围 - 只计算2025年1月1日到10月31日
    start_date = '2025-01-01'
    end_date = '2025-10-31'

    print("稳定币分析程序")
    print("=" * 70)
    print(f"分析时间范围: {start_date} 到 {end_date}")
    print("=" * 70)

    # 计算USDT的脱钩率
    print("\n1. USDT分析:")
    usdt_decoupling, usdt_count = calculate_average_decoupling_rate(usdt_file, start_date, end_date)

    if usdt_decoupling is not None:
        print(f"   - 使用的数据点数量: {usdt_count}")
        print(f"   - 平均脱钩率: {usdt_decoupling:.8f}")
        print(f"   - 平均脱钩率(百分比): {usdt_decoupling * 100:.6f}%")
    else:
        print("   - 无法计算USDT脱钩率")

    # 计算USDC的脱钩率
    print("\n2. USDC分析:")
    usdc_decoupling, usdc_count = calculate_average_decoupling_rate(usdc_file, start_date, end_date)

    if usdc_decoupling is not None:
        print(f"   - 使用的数据点数量: {usdc_count}")
        print(f"   - 平均脱钩率: {usdc_decoupling:.8f}")
        print(f"   - 平均脱钩率(百分比): {usdc_decoupling * 100:.6f}%")
    else:
        print("   - 无法计算USDC脱钩率")

    print("\n" + "=" * 70)
    print("稳定性比较:")
    print("=" * 70)

    # 比较两个稳定币的脱钩率
    if usdt_decoupling is not None and usdc_decoupling is not None:
        if usdt_decoupling < usdc_decoupling:
            stability_diff = usdc_decoupling - usdt_decoupling
            print(f"✓ USDT的稳定性优于USDC")
            print(f"  稳定性差异: {stability_diff:.8f} ({stability_diff * 100:.6f}%)")
        elif usdt_decoupling > usdc_decoupling:
            stability_diff = usdt_decoupling - usdc_decoupling
            print(f"✓ USDC的稳定性优于USDT")
            print(f"  稳定性差异: {stability_diff:.8f} ({stability_diff * 100:.6f}%)")
        else:
            print("USDT和USDC的稳定性相同")

        # 计算相对稳定性
        total_decoupling = usdt_decoupling + usdc_decoupling
        usdt_stability_ratio = (1 - usdt_decoupling / total_decoupling) * 100
        usdc_stability_ratio = (1 - usdc_decoupling / total_decoupling) * 100

        print(f"\n相对稳定性指数:")
        print(f"  USDT: {usdt_stability_ratio:.2f}%")
        print(f"  USDC: {usdc_stability_ratio:.2f}%")

    else:
        print("无法比较两个稳定币的稳定性 - 数据不完整")

    print("\n" + "=" * 70)
    print("交易量与市值比率分析:")
    print("=" * 70)

    # 计算USDT的交易量/市值比率
    print("\n1. USDT交易量/市值比率:")
    usdt_ratio, usdt_ratio_count = calculate_volume_marketcap_ratio(usdt_file, start_date, end_date)

    if usdt_ratio is not None:
        print(f"   - 使用的数据点数量: {usdt_ratio_count}")
        print(f"   - 平均交易量/市值比率: {usdt_ratio:.6f}")
        print(f"   - 平均交易量/市值比率(百分比): {usdt_ratio * 100:.4f}%")
    else:
        print("   - 无法计算USDT交易量/市值比率")

    # 计算USDC的交易量/市值比率
    print("\n2. USDC交易量/市值比率:")
    usdc_ratio, usdc_ratio_count = calculate_volume_marketcap_ratio(usdc_file, start_date, end_date)

    if usdc_ratio is not None:
        print(f"   - 使用的数据点数量: {usdc_ratio_count}")
        print(f"   - 平均交易量/市值比率: {usdc_ratio:.6f}")
        print(f"   - 平均交易量/市值比率(百分比): {usdc_ratio * 100:.4f}%")
    else:
        print("   - 无法计算USDC交易量/市值比率")

    print("\n" + "=" * 70)
    print("日平均波动率分析:")
    print("=" * 70)

    # 计算USDT的日平均波动率
    print("\n1. USDT日平均波动率:")
    usdt_volatility, usdt_vol_count = calculate_daily_volatility(usdt_file, start_date, end_date)

    if usdt_volatility is not None:
        print(f"   - 使用的数据点数量: {usdt_vol_count}")
        print(f"   - 日平均波动率: {usdt_volatility:.8f}")
        print(f"   - 日平均波动率(百分比): {usdt_volatility * 100:.6f}%")

        # 年化波动率（假设一年有365个交易日）
        annualized_volatility = usdt_volatility * np.sqrt(365)
        print(f"   - 年化波动率(估算): {annualized_volatility:.6f} ({annualized_volatility * 100:.4f}%)")
    else:
        print("   - 无法计算USDT日平均波动率")

    # 计算USDC的日平均波动率
    print("\n2. USDC日平均波动率:")
    usdc_volatility, usdc_vol_count = calculate_daily_volatility(usdc_file, start_date, end_date)

    if usdc_volatility is not None:
        print(f"   - 使用的数据点数量: {usdc_vol_count}")
        print(f"   - 日平均波动率: {usdc_volatility:.8f}")
        print(f"   - 日平均波动率(百分比): {usdc_volatility * 100:.6f}%")

        # 年化波动率（假设一年有365个交易日）
        annualized_volatility = usdc_volatility * np.sqrt(365)
        print(f"   - 年化波动率(估算): {annualized_volatility:.6f} ({annualized_volatility * 100:.4f}%)")
    else:
        print("   - 无法计算USDC日平均波动率")

    print("\n" + "=" * 70)
    print("比率分析说明:")
    print("=" * 70)

    # 分析交易量/市值比率的意义
    print("\n交易量/市值比率可能说明:")
    print("1. 流动性水平: 比率越高，表示相对于市值的交易活动越活跃")
    print("2. 市场深度: 高比率可能表明市场深度较好，买卖价差较小")
    print("3. 投资者行为: 比率变化可以反映投资者情绪和交易行为")
    print("4. 市场效率: 较高的比率通常与较高的市场效率相关")

    if usdt_ratio is not None and usdc_ratio is not None:
        print(f"\n具体分析:")
        if usdt_ratio > usdc_ratio:
            ratio_diff = usdt_ratio - usdc_ratio
            print(f"  - USDT的交易量/市值比率高于USDC ({ratio_diff:.6f} 或 {ratio_diff * 100:.4f}%)")
            print(f"  - 这表明USDT相对于其市值有更高的交易活跃度")
            print(f"  - 可能意味着USDT在市场中的流动性更好或交易更频繁")
        elif usdt_ratio < usdc_ratio:
            ratio_diff = usdc_ratio - usdt_ratio
            print(f"  - USDC的交易量/市值比率高于USDT ({ratio_diff:.6f} 或 {ratio_diff * 100:.4f}%)")
            print(f"  - 这表明USDC相对于其市值有更高的交易活跃度")
            print(f"  - 可能意味着USDC在市场中的流动性更好或交易更频繁")
        else:
            print(f"  - USDT和USDC的交易量/市值比率相同")
            print(f"  - 这表明两种稳定币在市场中的交易活跃度相对一致")

    print("\n" + "=" * 70)
    print("波动率分析说明:")
    print("=" * 70)

    print("\n日平均波动率说明:")
    print("1. 计算方法: 基于每日对数收益率的标准差计算")
    print("2. 意义: 波动率衡量了价格的变动幅度，是风险评估的重要指标")
    print("3. 稳定币特性: 理想情况下稳定币的波动率应该非常低")
    print("4. 风险比较: 波动率越高，价格风险越大")

    if usdt_volatility is not None and usdc_volatility is not None:
        print(f"\n具体分析:")
        if usdt_volatility > usdc_volatility:
            vol_diff = usdt_volatility - usdc_volatility
            print(f"  - USDT的日平均波动率高于USDC ({vol_diff:.8f} 或 {vol_diff * 100:.6f}%)")
            print(f"  - 这表明USDT的价格变动幅度更大，风险相对较高")
            print(f"  - 可能与USDT的市场结构或流动性有关")
        elif usdt_volatility < usdc_volatility:
            vol_diff = usdc_volatility - usdt_volatility
            print(f"  - USDC的日平均波动率高于USDT ({vol_diff:.8f} 或 {vol_diff * 100:.6f}%)")
            print(f"  - 这表明USDC的价格变动幅度更大，风险相对较高")
            print(f"  - 可能与USDC的市场结构或流动性有关")
        else:
            print(f"  - USDT和USDC的日平均波动率相同")
            print(f"  - 这表明两种稳定币的价格风险水平相似")

    print("\n" + "=" * 70)
    print("综合分析:")
    print("=" * 70)

    # 综合分析脱钩率、交易量/市值比率和波动率
    if (usdt_decoupling is not None and usdc_decoupling is not None and
            usdt_ratio is not None and usdc_ratio is not None and
            usdt_volatility is not None and usdc_volatility is not None):

        print("\n结合脱钩率、交易量/市值比率和波动率的综合分析:")

        # 创建综合评分系统
        # 脱钩率越低越好，交易量/市值比率越高越好，波动率越低越好
        usdt_score = (1 - usdt_decoupling) * 0.4 + usdt_ratio * 0.3 + (1 - usdt_volatility * 100) * 0.3
        usdc_score = (1 - usdc_decoupling) * 0.4 + usdc_ratio * 0.3 + (1 - usdc_volatility * 100) * 0.3

        print(f"  USDT综合评分: {usdt_score:.4f}")
        print(f"  USDC综合评分: {usdc_score:.4f}")

        if usdt_score > usdc_score:
            score_diff = usdt_score - usdc_score
            print(f"  ✓ USDT的综合表现优于USDC (差异: {score_diff:.4f})")
        elif usdt_score < usdc_score:
            score_diff = usdc_score - usdt_score
            print(f"  ✓ USDC的综合表现优于USDT (差异: {score_diff:.4f})")
        else:
            print(f"  USDT和USDC的综合表现相同")

    print("\n" + "=" * 70)
    print("分析完成")


if __name__ == "__main__":
    main()