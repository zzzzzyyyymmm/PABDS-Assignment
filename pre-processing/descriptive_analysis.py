import pandas as pd
import numpy as np
import os

# 定义数据文件路径
YELLOW_TAXI_PATH = 'yellow_tripdata_2024-12_cleaned.parquet'
GREEN_TAXI_PATH = 'green_tripdata_2024-12_cleaned.parquet'
FHV_TAXI_PATH = 'fhv_tripdata_2024-12_cleaned.parquet'

# 定义异常值输出文件路径
YELLOW_OUTLIERS_PATH = 'yellow_outliers.parquet'
GREEN_OUTLIERS_PATH = 'green_outliers.parquet'
FHV_OUTLIERS_PATH = 'fhv_outliers.parquet'


def load_data(file_path):
    """尝试加载 Parquet 或 CSV 文件，并返回 DataFrame"""
    if not os.path.exists(file_path):
        print(f"警告: 找不到文件 {file_path}")
        return None

    try:
        # 优先尝试加载 Parquet 文件
        df = pd.read_parquet(file_path)
    except Exception as e:
        # 如果失败，则尝试加载 CSV 文件
        print(f"警告: 加载 {file_path} 失败，尝试作为 CSV 文件加载。错误: {e}")
        df = pd.read_csv(file_path)

    print(f"\n{file_path} 数据加载成功，总行数: {len(df)}")
    return df


def get_datetime_and_duration(df, service_type):
    """根据服务类型返回日期时间字段并计算行程时长"""
    if service_type == 'yellow':
        pickup_col, dropoff_col = 'tpep_pickup_datetime', 'tpep_dropoff_datetime'
    elif service_type == 'green':
        pickup_col, dropoff_col = 'lpep_pickup_datetime', 'lpep_dropoff_datetime'
    else:  # fhv
        pickup_col, dropoff_col = 'pickup_datetime', 'dropOff_datetime'

    # 确保日期时间字段存在
    if pickup_col in df.columns and dropoff_col in df.columns:
        df[pickup_col] = pd.to_datetime(df[pickup_col], errors='coerce')
        df[dropoff_col] = pd.to_datetime(df[dropoff_col], errors='coerce')

        # 计算行程时长（以分钟为单位）
        df['duration_minutes'] = (df[dropoff_col] - df[pickup_col]).dt.total_seconds() / 60
    else:
        df['duration_minutes'] = np.nan

    return df, pickup_col, dropoff_col


def perform_analysis(df_yellow, df_green, df_fhv):
    """
    执行所有要求的分析任务，并处理缺失字段的情况。
    """
    all_datasets = {
        "Yellow Taxi": df_yellow,
        "Green Taxi": df_green,
        "FHV": df_fhv
    }

    # 定义异常值输出文件路径映射
    outlier_paths = {
        "Yellow Taxi": YELLOW_OUTLIERS_PATH,
        "Green Taxi": GREEN_OUTLIERS_PATH,
        "FHV": FHV_OUTLIERS_PATH
    }

    for name, df in all_datasets.items():
        if df is not None:
            print(f"\n--- 正在对 {name} 数据进行分析 ---")
            df, pickup_col, dropoff_col = get_datetime_and_duration(df, name.lower().split()[0])

            # 1. 时间节律分析（所有数据集都有日期时间字段）
            print("\n1. 时间节律分析:")
            if pickup_col in df.columns:
                df['hour'] = df[pickup_col].dt.hour
                df['weekday'] = df[pickup_col].dt.day_name()

                # 获取按小时的需求量，并按照数量从多到少排序
                hourly_demand = df['hour'].value_counts().sort_values(ascending=False)
                print(f"按小时需求量（全部24小时，按订单数降序排列）:\n{hourly_demand}")

                # 获取按星期几的需求量，并按照数量从多到少排序
                weekday_demand = df['weekday'].value_counts().sort_values(ascending=False)
                print(f"\n按星期几需求量（按订单数降序排列）:\n{weekday_demand}")

                # 新增：按天统计需求量，并找出最高和最低
                print("\n5. 按天需求量分析:")
                df['date'] = df[pickup_col].dt.date
                daily_demand = df['date'].value_counts().sort_values(ascending=False)

                most_orders_day = daily_demand.index[0]
                most_orders_count = daily_demand.iloc[0]

                least_orders_day = daily_demand.index[-1]
                least_orders_count = daily_demand.iloc[-1]

                print(f"订单量最多的一天是: {most_orders_day}，订单数: {most_orders_count}")
                print(f"订单量最少的一天是: {least_orders_day}，订单数: {least_orders_count}")

            # 2. 服务画像与效率/成本分析（仅处理有 trip_distance 的数据集）
            if 'trip_distance' in df.columns and 'total_amount' in df.columns:
                print("\n2. 服务画像与效率/成本分析:")

                df_clean = df[
                    (df['trip_distance'] > 0) & (df['total_amount'] > 0) & (df['duration_minutes'] > 0)].copy()

                avg_distance = df_clean['trip_distance'].mean()
                avg_duration = df_clean['duration_minutes'].mean()
                avg_total_cost = df_clean['total_amount'].mean()

                df_clean['cost_per_mile'] = df_clean['total_amount'] / df_clean['trip_distance']
                avg_cost_per_mile = df_clean['cost_per_mile'].mean()

                df_clean['avg_speed_mph'] = df_clean['trip_distance'] / (df_clean['duration_minutes'] / 60)
                avg_speed = df_clean['avg_speed_mph'].mean()

                print(f"平均行程距离: {avg_distance:.2f} 英里")
                print(f"平均行程时长: {avg_duration:.2f} 分钟")
                print(f"平均总费用: ${avg_total_cost:.2f}")
                print(f"平均单位距离成本: ${avg_cost_per_mile:.2f}/英里")
                print(f"平均时速: {avg_speed:.2f} 英里/小时")
            else:
                print("\n无法进行服务画像与效率/成本分析：缺少 'trip_distance' 或 'total_amount' 字段。")

            # 3. 支付模式统计（仅处理有 payment_type 的数据集）
            if 'payment_type' in df.columns:
                print("\n3. 支付模式统计:")
                payment_counts = df['payment_type'].value_counts(normalize=True) * 100
                print(f"支付方式比例:\n{payment_counts}")
            else:
                print("\n无法进行支付模式统计：缺少 'payment_type' 字段。")

            # 4. 异常值分析
            print("\n4. 异常值分析:")
            current_outliers = pd.DataFrame()

            if 'trip_distance' in df.columns:
                # 距离异常值
                q1_dist, q3_dist = df['trip_distance'].quantile(0.25), df['trip_distance'].quantile(0.75)
                iqr_dist = q3_dist - q1_dist
                upper_bound_dist = q3_dist + 1.5 * iqr_dist
                outliers_dist = df[df['trip_distance'] > upper_bound_dist].copy()
                outliers_dist['anomaly_type'] = 'Long Distance Outlier'
                print(f"发现 {len(outliers_dist)} 条超长距离的异常记录。")
                current_outliers = pd.concat([current_outliers, outliers_dist], ignore_index=True)

            if 'duration_minutes' in df.columns:
                # 时长异常值
                q1_dur, q3_dur = df['duration_minutes'].quantile(0.25), df['duration_minutes'].quantile(0.75)
                iqr_dur = q3_dur - q1_dur
                upper_bound_dur = q3_dur + 1.5 * iqr_dur
                outliers_dur = df[df['duration_minutes'] > upper_bound_dur].copy()
                outliers_dur['anomaly_type'] = 'Long Duration Outlier'
                print(f"发现 {len(outliers_dur)} 条超长时间的异常记录。")
                current_outliers = pd.concat([current_outliers, outliers_dur], ignore_index=True)

            if 'total_amount' in df.columns:
                # 费用异常值
                q1_cost, q3_cost = df['total_amount'].quantile(0.25), df['total_amount'].quantile(0.75)
                iqr_cost = q3_cost - q1_cost
                upper_bound_cost = q3_cost + 1.5 * iqr_cost
                outliers_cost = df[df['total_amount'] > upper_bound_cost].copy()
                outliers_cost['anomaly_type'] = 'High Cost Outlier'
                print(f"发现 {len(outliers_cost)} 条超高费用的异常记录。")
                current_outliers = pd.concat([current_outliers, outliers_cost], ignore_index=True)

            # 将当前服务的异常值保存到独立的文件
            if not current_outliers.empty:
                output_path = outlier_paths.get(name)
                if output_path:
                    current_outliers.to_parquet(output_path, index=False)
                    print(f"该服务的所有异常记录已成功保存到 {output_path}。")
            else:
                print(f"{name} 未发现异常记录，未生成异常值文件。")


def main():
    """主函数，执行所有分析任务"""
    print("--- 开始加载数据 ---")
    df_yellow = load_data(YELLOW_TAXI_PATH)
    df_green = load_data(GREEN_TAXI_PATH)
    df_fhv = load_data(FHV_TAXI_PATH)

    # 执行所有分析
    perform_analysis(df_yellow, df_green, df_fhv)

    print("\n所有分析任务已完成，请检查终端输出和生成的异常值文件。")


if __name__ == "__main__":
    main()
