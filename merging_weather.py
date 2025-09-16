import pandas as pd
import numpy as np
import os

# 定义数据文件路径，请将你的数据文件放在相应的路径下
YELLOW_TAXI_PATH = 'yellow_tripdata_2024-12_cleaned.parquet'
GREEN_TAXI_PATH = 'green_tripdata_2024-12_cleaned.parquet'
FHV_TAXI_PATH = 'fhv_tripdata_2024-12_cleaned.parquet'
WEATHER_DATA_PATH = 'New York 2024-12-01 to 2024-12-31.csv'

# 定义融合后数据文件的保存路径
YELLOW_MERGED_PATH = 'yellow_weather_merged.parquet'
GREEN_MERGED_PATH = 'green_weather_merged.parquet'
FHV_MERGED_PATH = 'fhv_weather_merged.parquet'


def load_data(file_path):
    """尝试加载 Parquet 或 CSV 文件，并返回 DataFrame"""
    if not os.path.exists(file_path):
        print(f"警告: 找不到文件 {file_path}")
        return None

    try:
        # 优先尝试加载 Parquet 文件
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        # 尝试加载 CSV 文件
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            print(f"警告: 不支持的文件格式 {file_path}")
            return None
    except Exception as e:
        print(f"加载 {file_path} 失败，请检查文件格式或路径。错误: {e}")
        return None

    print(f"\n{file_path} 数据加载成功，总行数: {len(df)}")
    return df


def merge_taxi_and_weather(taxi_df, weather_df, service_type):
    """
    将出租车数据与天气数据进行融合
    """
    if taxi_df is None or weather_df is None:
        print(f"无法融合 {service_type} 数据，因为其中一个数据源为空。")
        return None

    print(f"\n--- 正在处理和融合 {service_type} 数据 ---")

    # 根据服务类型选择正确的日期时间字段
    if service_type == 'yellow':
        pickup_col = 'tpep_pickup_datetime'
    elif service_type == 'green':
        pickup_col = 'lpep_pickup_datetime'
    else:  # fhv
        pickup_col = 'pickup_datetime'

    if pickup_col not in taxi_df.columns:
        print(f"警告: {service_type} 数据缺少必要的字段 '{pickup_col}'，跳过融合。")
        return None

    # 预处理出租车数据的时间戳
    taxi_df[pickup_col] = pd.to_datetime(taxi_df[pickup_col], errors='coerce')
    taxi_df.dropna(subset=[pickup_col], inplace=True)
    taxi_df['date'] = taxi_df[pickup_col].dt.date
    taxi_df['hour'] = taxi_df[pickup_col].dt.hour

    # 预处理天气数据的时间戳
    weather_df['datetime'] = pd.to_datetime(weather_df['datetime'], errors='coerce')
    weather_df.dropna(subset=['datetime'], inplace=True)
    weather_df['date'] = weather_df['datetime'].dt.date
    weather_df['hour'] = weather_df['datetime'].dt.hour

    # 基于日期和小时进行数据融合，保留所有出租车数据
    merged_df = pd.merge(taxi_df, weather_df, on=['date', 'hour'], how='left')

    print(f"{service_type} 数据融合成功！融合后的数据（部分）：")
    print(merged_df.head())
    print(f"\n融合后的数据总行数: {len(merged_df)}")

    return merged_df


def main():
    """主函数，执行数据准备与融合"""
    print("--- 开始加载数据 ---")

    # 加载所有数据
    df_yellow = load_data(YELLOW_TAXI_PATH)
    df_green = load_data(GREEN_TAXI_PATH)
    df_fhv = load_data(FHV_TAXI_PATH)
    weather_df = load_data(WEATHER_DATA_PATH)

    if weather_df is not None:
        # 分别处理和保存每个服务的数据
        if df_yellow is not None:
            merged_yellow = merge_taxi_and_weather(df_yellow, weather_df, 'yellow')
            if merged_yellow is not None:
                print(f"\n--- 正在将融合后的 Yellow Taxi 数据保存到 {YELLOW_MERGED_PATH} ---")
                merged_yellow.to_parquet(YELLOW_MERGED_PATH, index=False)
                print("保存成功！")

        if df_green is not None:
            merged_green = merge_taxi_and_weather(df_green, weather_df, 'green')
            if merged_green is not None:
                print(f"\n--- 正在将融合后的 Green Taxi 数据保存到 {GREEN_MERGED_PATH} ---")
                merged_green.to_parquet(GREEN_MERGED_PATH, index=False)
                print("保存成功！")

        if df_fhv is not None:
            merged_fhv = merge_taxi_and_weather(df_fhv, weather_df, 'fhv')
            if merged_fhv is not None:
                print(f"\n--- 正在将融合后的 FHV 数据保存到 {FHV_MERGED_PATH} ---")
                merged_fhv.to_parquet(FHV_MERGED_PATH, index=False)
                print("保存成功！")
    else:
        print("天气数据加载失败，无法进行融合。")

    print("\n所有分析任务已完成，请检查生成的三个独立文件。")


if __name__ == "__main__":
    main()