import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- 定义文件路径 ---
WEATHER_MERGED_PATH = {
    'Yellow Taxi': 'yellow_weather_merged.parquet',
    'Green Taxi': 'green_weather_merged.parquet',
    'FHV': 'fhv_weather_merged.parquet'
}
FUSED_POI_PICKUP_PATH = {
    'Yellow Taxi': 'fused_pickup_yellow.parquet',
    'Green Taxi': 'fused_pickup_green.parquet',
    'FHV': 'fused_pickup_fhv.parquet'
}
FUSED_POI_DROPOFF_PATH = {
    'Yellow Taxi': 'fused_dropoff_yellow.parquet',
    'Green Taxi': 'fused_dropoff_green.parquet',
    'FHV': 'fused_dropoff_fhv.parquet'
}
ANOMALY_PATH = {
    'Yellow Taxi': 'yellow_outliers.parquet',
    'Green Taxi': 'green_outliers.parquet',
    'FHV': 'fhv_outliers.parquet'
}


def load_data(file_path, service_name):
    """加载 Parquet 文件"""
    if not os.path.exists(file_path):
        print(f"警告: 找不到文件 {file_path}，跳过 {service_name} 服务。")
        return None
    try:
        df = pd.read_parquet(file_path)
        print(f"\n{service_name}: {file_path} 数据加载成功，总行数: {len(df)}")
        if 'index' in df.columns:
            df = df.set_index('index')
        return df
    except Exception as e:
        print(f"加载 {file_path} 失败: {e}")
        return None


def classify_time_period(datetime_obj):
    """根据日期和时间将数据划分为不同时段"""
    if pd.isna(datetime_obj):
        return 'Unknown'
    day_of_week = datetime_obj.dayofweek
    hour = datetime_obj.hour
    if day_of_week < 5:
        if hour >= 7 and hour <= 10:
            return 'Workday Morning Rush Hour'
        elif hour >= 16 and hour <= 19:
            return 'Workday Evening Rush Hour'
        else:
            return 'Workday Other'
    else:
        return 'Weekend'


def analyze_feature_impact(df, service_name, pickup_col):
    """
    使用逻辑回归模型量化特征对异常值的影响力
    """
    print(f"\n--- 正在使用逻辑回归量化 {service_name} 服务的特征影响力 ---")

    # 1. 特征工程
    df[pickup_col] = pd.to_datetime(df[pickup_col], errors='coerce')
    df.dropna(subset=[pickup_col], inplace=True)
    df['time_period'] = df[pickup_col].apply(classify_time_period)

    df['conditions'] = df['conditions'].fillna('unknown').str.lower()
    df['is_bad_weather'] = (df['conditions'].str.contains('rain') | df['conditions'].str.contains('snow')).astype(int)

    # 定义自变量和因变量
    features = ['is_bad_weather', 'time_period', 'pickup_category', 'dropoff_category']
    X = df[features]
    y = df['is_anomaly']

    if X.empty or y.nunique() < 2:
        print(f"警告: {service_name} 没有足够的有效数据或异常值类别进行逻辑回归分析，跳过。")
        return

    categorical_features = ['time_period', 'pickup_category', 'dropoff_category']
    preprocessor = ColumnTransformer(
        transformers=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)],
        remainder='passthrough'
    )

    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', LogisticRegression(solver='liblinear', random_state=42))])

    model.fit(X, y)

    ohe_feature_names = model.named_steps['preprocessor'].named_transformers_['onehot'].get_feature_names_out(
        categorical_features)
    non_ohe_features = [col for col in features if col not in categorical_features]
    feature_names = list(ohe_feature_names) + non_ohe_features

    coefficients = model.named_steps['classifier'].coef_[0]
    odds_ratios = np.exp(coefficients)

    results_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Odds_Ratio': odds_ratios
    }).sort_values(by='Odds_Ratio', ascending=False)

    print("\n--- 特征影响力排名 (Odds Ratio) ---")
    print(results_df)

    print("\nOdds Ratio > 1 表示特征会增加订单成为异常值的几率。")
    print("Odds Ratio < 1 表示特征会降低订单成为异常值的几率。")


def main():
    """主函数，执行所有分析任务"""
    print("--- 开始进行异常值归因分析 ---")

    services = ['Yellow Taxi', 'Green Taxi', 'FHV']

    for service in services:
        print(f"\n#####################################################")
        print(f"###  正在处理 {service} 数据  ###")

        # 1. 加载所有原始数据
        weather_df = load_data(WEATHER_MERGED_PATH[service], service)
        pickup_poi_df = load_data(FUSED_POI_PICKUP_PATH[service], service)
        dropoff_poi_df = load_data(FUSED_POI_DROPOFF_PATH[service], service)
        anomaly_df = load_data(ANOMALY_PATH[service], service)

        if weather_df is None or pickup_poi_df is None or dropoff_poi_df is None:
            print("数据文件不完整，跳过此服务。")
            continue

        # 2. 核心：数据过滤与融合
        # 先与上车点数据进行 inner join
        merged_df = pd.merge(weather_df, pickup_poi_df[['category']], left_index=True, right_index=True, how='inner')
        merged_df.rename(columns={'category': 'pickup_category'}, inplace=True)

        # 再与下车点数据进行 inner join
        final_df = pd.merge(merged_df, dropoff_poi_df[['category']], left_index=True, right_index=True, how='inner')
        final_df.rename(columns={'category': 'dropoff_category'}, inplace=True)

        # 3. 标记异常值，确保正常和异常订单都存在
        final_df['is_anomaly'] = 0
        if anomaly_df is not None:
            anomaly_indices = anomaly_df.index
            final_df['is_anomaly'] = final_df.index.isin(anomaly_indices).astype(int)
            print(f"\n已成功标记 {final_df['is_anomaly'].sum()} 个异常订单。")

        # 检查是否包含至少两个类别，否则跳过分析
        if final_df['is_anomaly'].nunique() < 2:
            print(f"警告: 过滤后的 {service} 数据只包含单一类别的异常值，无法进行模型训练，跳过。")
            continue

        # 4. 执行分析
        if service == 'Yellow Taxi':
            pickup_col = 'tpep_pickup_datetime'
        elif service == 'Green Taxi':
            pickup_col = 'lpep_pickup_datetime'
        else:  # FHV
            pickup_col = 'pickup_datetime'

        analyze_feature_impact(final_df, service, pickup_col)
        print(f"#####################################################")


if __name__ == "__main__":
    main()