import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

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


def classify_weather_by_conditions(conditions_str):
    """
    根据用户指定的精确条件对天气进行分类
    """
    if not isinstance(conditions_str, str):
        return 'Other'

    conditions_str = conditions_str.strip().lower()

    if conditions_str in ['rain, overcast', 'rain, partially cloudy']:
        return 'Rain, Overcast'
    if conditions_str == 'snow, rain, overcast':
        return 'Snow, Rain, Overcast'
    if conditions_str == 'clear':
        return 'Clear'
    if conditions_str == 'overcast':
        return 'Overcast'
    if conditions_str == 'partially cloudy':
        return 'Partially Cloudy'

    return 'Other'


def classify_visibility(visibility_km):
    """
    将能见度（公里）划分为气象学标准等级
    """
    if pd.isna(visibility_km):
        return 'Unknown'

    # 按照气象学定义，从能见度最低的等级开始判断
    if visibility_km < 0.05:
        return '强浓雾 (<0.05km)'
    if visibility_km < 0.2:
        return '浓雾 (<0.2km)'
    if visibility_km < 0.5:
        return '大雾 (<0.5km)'
    if visibility_km < 1.0:
        return '雾 (<1km)'
    if visibility_km < 10.0:
        return '轻雾 (<10km)'

    return '良好能见度 (>=10km)'


def analyze_weather_impact(df, service_name, pickup_col):
    """
    对融合后的数据进行天气影响分析
    """
    if df is None or df.empty:
        print(f"--- {service_name} 数据为空，无法进行分析。---")
        return

    print(f"\n--- 正在对 {service_name} 服务进行天气影响分析 ---")

    # 首先，聚合原始数据，得到每小时的订单量
    df[pickup_col] = pd.to_datetime(df[pickup_col], errors='coerce')
    df.dropna(subset=[pickup_col], inplace=True)
    df['date'] = df[pickup_col].dt.date
    df['hour'] = df[pickup_col].dt.hour

    hourly_trips = df.groupby(['date', 'hour']).size().reset_index(name='trip_count')

    # 获取唯一的日期和小时对应的天气数据，并与订单量数据融合
    weather_cols = ['date', 'hour', 'temp', 'precip', 'snow', 'conditions', 'visibility']
    hourly_weather_df = df[weather_cols].drop_duplicates().reset_index(drop=True)

    # 将订单量和天气数据融合到同一个 DataFrame
    analysis_df = pd.merge(hourly_trips, hourly_weather_df, on=['date', 'hour'], how='left')

    # 方案一: 降水与需求分析
    print("\n1. 降水与需求分析：")
    analysis_df['weather_category'] = analysis_df['conditions'].apply(classify_weather_by_conditions)

    avg_trips_by_weather = analysis_df.groupby('weather_category')['trip_count'].mean().sort_values(ascending=False)
    print("不同天气条件下的平均订单量：")
    print(avg_trips_by_weather)

    # **天气类别差异性检验**
    weather_groups = [group['trip_count'] for name, group in analysis_df.groupby('weather_category') if
                      len(group['trip_count']) > 0]
    if len(weather_groups) > 1:
        f_stat, p_value = stats.f_oneway(*weather_groups)
        print("\n--- 天气类别差异性检验 (ANOVA) ---")
        print(f"F统计量: {f_stat:.2f}")
        print(f"P值: {p_value:.3f}")
        if p_value < 0.05:
            print("结论: 不同天气类别下的订单量存在显著差异。")
        else:
            print("结论: 不同天气类别下的订单量没有显著差异。")
    else:
        print("警告: 天气类别数量不足，无法进行差异性检验。")

    # 方案二: 温度与需求分析
    print("\n2. 温度与需求分析：")
    if 'temp' in analysis_df.columns:
        plt.figure(figsize=(10, 6))
        sns.regplot(x='temp', y='trip_count', data=analysis_df, ci=None, scatter_kws={'alpha': 0.3})
        plt.title(f'{service_name} - 温度与每小时订单量的关系', fontsize=16)
        plt.xlabel('温度 (摄氏度)', fontsize=12)
        plt.ylabel('每小时订单总量', fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("警告：数据中缺少 'temp' 字段，无法进行温度分析。")

    # 方案三: 能见度与需求分析
    print("\n3. 能见度与需求分析：")
    if 'visibility' in analysis_df.columns:
        # 按能见度等级分组并计算平均订单量
        analysis_df['visibility_category'] = analysis_df['visibility'].apply(classify_visibility)
        avg_trips_by_visibility = analysis_df.groupby('visibility_category')['trip_count'].mean().sort_index()

        plt.figure(figsize=(12, 6))
        avg_trips_by_visibility.plot(kind='bar', color='skyblue')
        plt.title(f'{service_name} - 不同能见度等级下的平均每小时订单量', fontsize=16)
        plt.xlabel('能见度等级', fontsize=12)
        plt.ylabel('平均每小时订单量', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

        # **更新：能见度等级差异性检验 (克鲁斯卡尔-沃利斯H检验)**
        visibility_groups = [group['trip_count'] for name, group in analysis_df.groupby('visibility_category') if
                             len(group['trip_count']) > 0]
        if len(visibility_groups) > 1:
            try:
                h_stat, p_value = stats.kruskal(*visibility_groups)
                print("\n--- 能见度等级差异性检验 (克鲁斯卡尔-沃利斯H检验) ---")
                print(f"H统计量: {h_stat:.2f}")
                print(f"P值: {p_value:.3f}")
                if p_value < 0.05:
                    print("结论: 不同能见度等级下的订单量存在显著差异。")
                else:
                    print("结论: 不同能见度等级下的订单量没有显著差异。")
            except ValueError as e:
                print(f"警告: 无法进行克鲁斯卡尔-沃利斯H检验，因为数据不足或分布不均。错误信息: {e}")
        else:
            print("警告: 能见度类别数量不足，无法进行差异性检验。")

    else:
        print("警告：数据中缺少 'visibility' 字段，无法进行能见度分析。")


def main():
    """主函数，执行所有分析任务"""
    print("--- 开始进行天气对出行需求影响的分析 ---")

    # 加载数据
    df_yellow = load_data(YELLOW_MERGED_PATH)
    df_green = load_data(GREEN_MERGED_PATH)
    df_fhv = load_data(FHV_MERGED_PATH)

    # 对每个数据集分别进行分析，并传入正确的上车时间字段名
    analyze_weather_impact(df_yellow, 'Yellow Taxi', 'tpep_pickup_datetime')
    analyze_weather_impact(df_green, 'Green Taxi', 'lpep_pickup_datetime')
    analyze_weather_impact(df_fhv, 'FHV', 'pickup_datetime')

    print("\n所有分析任务已完成！")


if __name__ == "__main__":
    main()