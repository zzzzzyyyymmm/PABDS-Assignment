import pandas as pd
import numpy as np
import os
from scipy import stats

# 定义数据文件路径
YELLOW_MERGED_PATH = 'yellow_weather_merged.parquet'
GREEN_MERGED_PATH = 'green_weather_merged.parquet'
FHV_MERGED_PATH = 'fhv_weather_merged.parquet'


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


def classify_time_period(datetime_obj):
    """
    根据日期和时间将数据划分为不同时段
    """
    if pd.isna(datetime_obj):
        return 'Unknown'

    day_of_week = datetime_obj.dayofweek  # 0=Monday, 6=Sunday
    hour = datetime_obj.hour

    if day_of_week < 5:  # Workday
        if hour >= 7 and hour <= 10:
            return 'Workday Morning Rush Hour'
        elif hour >= 16 and hour <= 19:
            return 'Workday Evening Rush Hour'
        else:
            return 'Workday Other'
    else:  # Weekend
        return 'Weekend'


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
                print("结论: 恶劣天气对该时段的订单量存在显著影响。")
            else:
                print("结论: 恶劣天气对该时段的订单量没有显著影响。")
        except ValueError as e:
            print(f"警告: 无法进行克鲁斯卡尔-沃利斯H检验，因为数据不足或分布不均。错误信息: {e}")
    else:
        print(f"警告: 天气类别数量不足，无法进行 {metric_name} 差异性检验。")


def analyze_weather_and_time_interaction(df, service_name, pickup_col):
    """
    分析天气对不同时段出行需求的影响
    """
    if df is None or df.empty:
        print(f"--- {service_name} 数据为空，无法进行分析。---")
        return

    print(f"\n--- 正在对 {service_name} 服务进行天气与时段交互作用分析 ---")

    # 聚合原始数据，得到每小时的订单量
    df[pickup_col] = pd.to_datetime(df[pickup_col], errors='coerce')
    df.dropna(subset=[pickup_col], inplace=True)

    df['date'] = df[pickup_col].dt.date
    df['hour'] = df[pickup_col].dt.hour

    hourly_trips = df.groupby(['date', 'hour']).size().reset_index(name='trip_count')

    # 获取唯一的日期和小时对应的天气数据，并与订单量数据融合
    weather_cols = ['date', 'hour', 'conditions']
    hourly_weather_df = df[weather_cols].drop_duplicates().reset_index(drop=True)

    # 将订单量和天气数据融合到同一个 DataFrame
    analysis_df = pd.merge(hourly_trips, hourly_weather_df, on=['date', 'hour'], how='left')

    # 对数据进行天气和时间分类
    analysis_df['weather_category'] = analysis_df['conditions'].apply(classify_weather_for_efficiency)
    analysis_df['time_period'] = pd.to_datetime(analysis_df['date']).apply(classify_time_period)

    # 按时段和天气分组，并计算平均订单量
    grouped_data = analysis_df.groupby(['time_period', 'weather_category'])['trip_count'].mean().unstack()

    print("\n不同时段和天气下的平均每小时订单量：")
    print(grouped_data)

    # 对每个时段进行差异性检验
    print("\n--- 不同时段下的订单量差异性检验 ---")
    for time_period, group_df in analysis_df.groupby('time_period'):
        print(f"\n**时段: {time_period}**")
        run_kruskal_test(group_df['trip_count'], group_df['weather_category'], f"{time_period}订单量")


def main():
    """主函数，执行所有分析任务"""
    print("--- 开始进行天气对出行需求影响的分析 ---")

    # 加载数据
    df_yellow = load_data(YELLOW_MERGED_PATH)
    df_green = load_data(GREEN_MERGED_PATH)
    df_fhv = load_data(FHV_MERGED_PATH)

    # 对每个数据集分别进行分析，并传入正确的上车时间字段名
    analyze_weather_and_time_interaction(df_yellow, 'Yellow Taxi', 'tpep_pickup_datetime')
    analyze_weather_and_time_interaction(df_green, 'Green Taxi', 'lpep_pickup_datetime')
    analyze_weather_and_time_interaction(df_fhv, 'FHV', 'pickup_datetime')

    print("\n所有分析任务已完成！")


if __name__ == "__main__":
    main()
