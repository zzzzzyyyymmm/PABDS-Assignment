import pandas as pd
import numpy as np
import os
from scipy import stats

# 定义数据文件路径
YELLOW_MERGED_PATH = 'yellow_weather_merged.parquet'
GREEN_MERGED_PATH = 'green_weather_merged.parquet'


def load_data(file_path):
    """加载 Parquet 文件"""
    if not os.path.exists(file_path):
        print(f"警告: 找不到文件 {file_path}，跳过该服务分析。")
        return None
    try:
        df = pd.read_parquet(file_path)
        print(f"\n{file_path} 数据加载成功，总行数: {len(df)}")
        return df
    except Exception as e:
        print(f"加载 {file_path} 失败: {e}")
        return None


def classify_weather_for_efficiency(conditions_str):
    """
    根据 'conditions' 字段对天气进行二元分类：恶劣天气或正常天气
    """
    if not isinstance(conditions_str, str):
        return 'Normal Weather'

    conditions_str = conditions_str.strip().lower()

    if conditions_str in ['rain, overcast', 'snow, rain, overcast', 'rain, partially cloudy']:
        return 'Bad Weather'

    return 'Normal Weather'


def run_kruskal_test(data_series, group_labels, metric_name):
    """
    对不同组别的数据进行克鲁斯卡尔-沃利斯H检验
    """
    groups = [group for group in data_series.groupby(group_labels) if len(group[1]) > 0]

    if len(groups) > 1:
        try:
            h_stat, p_value = stats.kruskal(*[group[1] for group in groups])
            print(f"\n--- {metric_name} 差异性检验 (克鲁斯卡尔-沃利斯H检验) ---")
            print(f"H统计量: {h_stat:.2f}")
            print(f"P值: {p_value:.3f}")
            if p_value < 0.05:
                print("结论: 恶劣天气对该指标存在显著影响。")
            else:
                print("结论: 恶劣天气对该指标没有显著影响。")
        except ValueError as e:
            print(f"警告: 无法进行克鲁斯卡尔-沃利斯H检验，因为数据不足或分布不均。错误信息: {e}")
    else:
        print(f"警告: 天气类别数量不足，无法进行 {metric_name} 差异性检验。")


def analyze_weather_impact(df, service_name, pickup_col, dropoff_col):
    """
    分析天气对服务效率和成本的影响
    """
    if df is None or df.empty:
        print(f"--- {service_name} 数据为空，无法进行分析。---")
        return

    print(f"\n--- 正在对 {service_name} 服务进行天气影响分析 ---")

    # 对数据进行天气分类
    df['weather_category'] = df['conditions'].apply(classify_weather_for_efficiency)

    # 效率和成本分析，只对有相关字段的出租车服务进行
    if 'trip_distance' in df.columns and 'total_amount' in df.columns:
        # 计算行程时长和时速
        df['duration_minutes'] = (pd.to_datetime(df[dropoff_col]) - pd.to_datetime(
            df[pickup_col])).dt.total_seconds() / 60
        df['avg_speed_mph'] = df['trip_distance'] / (df['duration_minutes'] / 60)

        # 移除异常值（更严格的过滤）
        df = df[
            (df['duration_minutes'] > 1) & (df['duration_minutes'] < 180) &
            (df['trip_distance'] > 0) &
            (df['avg_speed_mph'] > 0) & (df['avg_speed_mph'] < 60) &
            (df['total_amount'] > 0)
            ]

        # 计算每英里成本
        df['cost_per_mile'] = df['total_amount'] / df['trip_distance']

        # 效率和成本指标按天气分组
        weather_metrics = df.groupby('weather_category').agg(
            avg_duration_minutes=('duration_minutes', 'mean'),
            avg_speed_mph=('avg_speed_mph', 'mean'),
            avg_total_amount=('total_amount', 'mean'),
            avg_cost_per_mile=('cost_per_mile', 'mean')
        ).sort_index()

        # 方案一：效率分析
        print("\n1. 效率分析：")
        print("不同天气下的平均行程时长和平均时速：")
        print(weather_metrics[['avg_duration_minutes', 'avg_speed_mph']])

        # 进行差异性检验
        run_kruskal_test(df['duration_minutes'], df['weather_category'], '行程时长')
        run_kruskal_test(df['avg_speed_mph'], df['weather_category'], '平均时速')

        # 方案二：成本分析
        print("\n2. 成本分析：")
        print("不同天气下的平均总费用和每英里成本：")
        print(weather_metrics[['avg_total_amount', 'avg_cost_per_mile']])

        # 进行差异性检验
        run_kruskal_test(df['total_amount'], df['weather_category'], '总费用')
        run_kruskal_test(df['cost_per_mile'], df['weather_category'], '每英里成本')

    else:
        print(f"警告：{service_name} 服务数据中缺少 'trip_distance' 或 'total_amount' 字段，无法进行效率和成本分析。")


def main():
    """主函数，执行所有分析任务"""
    print("--- 开始进行天气对出行需求影响的分析 ---")

    # 加载数据
    df_yellow = load_data(YELLOW_MERGED_PATH)
    df_green = load_data(GREEN_MERGED_PATH)

    # 对每个数据集分别进行分析
    analyze_weather_impact(df_yellow, 'Yellow Taxi', 'tpep_pickup_datetime', 'tpep_dropoff_datetime')
    analyze_weather_impact(df_green, 'Green Taxi', 'lpep_pickup_datetime', 'lpep_dropoff_datetime')

    print("\n所有分析任务已完成！")


if __name__ == "__main__":
    main()
