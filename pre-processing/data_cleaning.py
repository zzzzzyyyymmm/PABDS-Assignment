import pandas as pd
import numpy as np
import os

# 定义数据文件路径
YELLOW_TAXI_PATH = 'yellow_tripdata_2024-12.parquet'
GREEN_TAXI_PATH = 'green_tripdata_2024-12.parquet'
FHV_TAXI_PATH = 'fhv_tripdata_2024-12.parquet'

# 定义清洗后文件的保存路径
YELLOW_CLEANED_PATH = 'yellow_tripdata_2024-12_cleaned.parquet'
GREEN_CLEANED_PATH = 'green_tripdata_2024-12_cleaned.parquet'
FHV_CLEANED_PATH = 'fhv_tripdata_2024-12_cleaned.parquet'


def clean_data(file_path, service_type):
    """
    加载数据，检查日期并根据服务类型删除特定缺失值，然后保存修正后的文件。
    """
    print(f"\n--- 正在处理文件: {file_path} ---")
    if not os.path.exists(file_path):
        print(f"警告: 找不到文件 {file_path}")
        return

    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"错误: 加载 {file_path} 失败，无法进行清洗。错误信息: {e}")
        return

    initial_rows = len(df)
    print(f"原始行数: {initial_rows}")

    # 根据服务类型选择正确的日期时间列
    if service_type == 'yellow':
        pickup_col = 'tpep_pickup_datetime'
    elif service_type == 'green':
        pickup_col = 'lpep_pickup_datetime'
    else:  # fhv
        pickup_col = 'pickup_datetime'

    # 将日期时间列转换为正确的格式，并筛选出12月份的记录
    if pickup_col in df.columns:
        df[pickup_col] = pd.to_datetime(df[pickup_col], errors='coerce')

        # 筛选出日期在2024年12月的记录
        df = df[(df[pickup_col].dt.year == 2024) & (df[pickup_col].dt.month == 12)]
        print(f"日期筛选后剩余行数: {len(df)}")
    else:
        print(f"警告: 找不到日期列 '{pickup_col}'，跳过日期筛选。")

    # 根据服务类型执行不同的缺失值处理策略
    if service_type == 'yellow':
        # 黄色出租车：删除任何有缺失值的记录
        df.dropna(inplace=True)
        print("黄色出租车：已删除所有含有缺失值的记录。")
    elif service_type == 'green':
        # 绿色出租车：只删除核心列有缺失值的记录
        required_cols = ['lpep_pickup_datetime', 'lpep_dropoff_datetime', 'PULocationID', 'DOLocationID']
        df.dropna(subset=required_cols, inplace=True)
        print("绿色出租车：已删除核心列含有缺失值的记录。")
    elif service_type == 'fhv':
        # FHV：只删除核心列有缺失值的记录
        required_cols = ['pickup_datetime', 'dropOff_datetime', 'PUlocationID', 'DOlocationID']
        df.dropna(subset=required_cols, inplace=True)
        print("FHV：已删除核心列含有缺失值的记录。")

    cleaned_rows = len(df)
    print(f"缺失值处理后剩余行数: {cleaned_rows}")
    print(f"总共删除了 {initial_rows - cleaned_rows} 行记录。")

    # 保存清洗后的文件
    cleaned_path = file_path.replace('.parquet', '_cleaned.parquet')
    df.to_parquet(cleaned_path, index=False)
    print(f"修正后的文件已成功保存到: {cleaned_path}")


def main():
    """主函数，执行所有文件的清洗任务"""
    clean_data(YELLOW_TAXI_PATH, 'yellow')
    clean_data(GREEN_TAXI_PATH, 'green')
    clean_data(FHV_TAXI_PATH, 'fhv')

    print("\n所有文件处理完毕。")


if __name__ == "__main__":
    main()
