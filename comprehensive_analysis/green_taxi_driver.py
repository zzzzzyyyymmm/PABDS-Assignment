import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from sklearn.impute import SimpleImputer
import geopandas as gpd
import dask.dataframe as dd

# --- 定义文件路径 ---
WEATHER_MERGED_PATH = 'green_weather_merged.parquet'
FUSED_POI_PICKUP_PATH = 'fused_pickup_green.parquet'
FUSED_POI_DROPOFF_PATH = 'fused_dropoff_green.parquet'
ANOMALY_PATH = 'green_outliers.parquet'
NYC_ZONES_PATH = 'taxi_zones/taxi_zones.shp'


def load_data(file_path, service_name):
    """加载 Parquet 文件"""
    if not os.path.exists(file_path):
        print(f"警告: 找不到文件 {file_path}，跳过 {service_name} 服务。")
        return None
    try:
        ddf = dd.read_parquet(file_path)
        df = ddf.compute()
        print(f"\n{service_name}: {file_path} 数据加载成功，总行数: {len(df)}")
        if 'index' in df.columns:
            df = df.set_index('index')
        return df
    except Exception as e:
        print(f"加载 {file_path} 失败: {e}")
        return None


def classify_time_period(datetime_series):
    """完全向量化时间分类"""
    if datetime_series.isna().all():
        return pd.Series(['Unknown'] * len(datetime_series), index=datetime_series.index)
    day_of_week = datetime_series.dt.dayofweek
    hour = datetime_series.dt.hour
    conditions = np.where(day_of_week < 5,
                          np.where((hour >= 7) & (hour <= 10), 'Workday Morning Rush Hour',
                                   np.where((hour >= 16) & (hour <= 19), 'Workday Evening Rush Hour', 'Workday Other')),
                          'Weekend')
    return pd.Series(conditions, index=datetime_series.index)


def haversine_distance(lat1, lon1, lat2, lon2):
    """标量计算哈弗辛距离"""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def load_zone_centroids():
    """加载出租车区域质心"""
    if not os.path.exists(NYC_ZONES_PATH):
        print(f"警告: 找不到 shapefile {NYC_ZONES_PATH}，无法计算距离。")
        return None
    try:
        zones_gdf = gpd.read_file(NYC_ZONES_PATH)
        zones_gdf['centroid'] = zones_gdf.geometry.centroid
        zones_gdf['LocationID'] = zones_gdf['LocationID'].astype(str)
        centroids = {row['LocationID']: (row['centroid'].y, row['centroid'].x) for _, row in zones_gdf.iterrows()}
        print(f"加载 {len(centroids)} 个区域质心成功。")
        return centroids
    except Exception as e:
        print(f"加载 shapefile 失败: {e}")
        return None


def remove_outliers(df, column, threshold=0.98):
    """移除指定列的异常值（基于分位数）"""
    q_high = df[column].quantile(threshold)
    df = df[df[column] <= q_high]
    print(f"移除 {column} 异常值后，数据行数: {len(df)}")
    return df


def prepare_prediction_data(df, service_name, centroids=None, anomaly_df=None):
    """准备预测数据，并排除异常值"""
    if df is None or df.empty:
        return None, None, None

    original_index = df.index.copy()
    print(f"原始数据行数: {len(original_index)}")

    pickup_col = 'lpep_pickup_datetime'
    anomaly_pickup_col = pickup_col
    if anomaly_df is not None and not anomaly_df.empty:
        if anomaly_pickup_col not in anomaly_df.columns:
            print(f"警告: {service_name} 异常值数据中缺少 {anomaly_pickup_col}，跳过异常值排除。")
            anomaly_df = None
        else:
            anomaly_df[anomaly_pickup_col] = pd.to_datetime(anomaly_df[anomaly_pickup_col], errors='coerce')
            anomaly_df['PULocationID'] = anomaly_df['PULocationID'].astype(str)
            df[pickup_col] = pd.to_datetime(df[pickup_col], errors='coerce')
            df['PULocationID'] = df['PULocationID'].astype(str)

            print(f"异常值数据行数: {len(anomaly_df)}")
            df = df[~((df[pickup_col].isin(anomaly_df[anomaly_pickup_col])) &
                      (df['PULocationID'].isin(anomaly_df['PULocationID'])))].copy()
            print(f"排除异常值后，{service_name} 数据行数: {len(df)} (原始: {len(original_index)})")

    df = df.dropna(subset=[pickup_col])
    df['time_period'] = classify_time_period(df[pickup_col])

    df['precipitation'] = df['precipitation'].fillna(0) if 'precipitation' in df else None
    df['thermal_comfort'] = df['thermal_comfort'].fillna(
        df['thermal_comfort'].mean()) if 'thermal_comfort' in df else None
    df['solar_radiation'] = df['solar_radiation'].fillna(
        df['solar_radiation'].mean()) if 'solar_radiation' in df else None
    df['wind_dynamics'] = df['wind_dynamics'].fillna(df['wind_dynamics'].mean()) if 'wind_dynamics' in df else None
    df['snow_accumulation'] = df['snow_accumulation'].fillna(0) if 'snow_accumulation' in df else None

    df['pickup_category'] = df['pickup_category'].fillna('Unknown')
    df['dropoff_category'] = df['dropoff_category'].fillna('Unknown')

    dropoff_col = 'lpep_dropoff_datetime'
    if centroids and 'PULocationID' in df and 'DOLocationID' in df:
        df['PULocationID'] = df['PULocationID'].astype(str)
        df['DOLocationID'] = df['DOLocationID'].astype(str)
        common_ids = set(df['DOLocationID'].unique()) & set(centroids.keys())
        print(
            f"DOLocationID 与 centroids 交集比例: {len(common_ids) / len(df['DOLocationID'].unique()) if df['DOLocationID'].nunique() > 0 else 0:.2%}")
        print(f"转换后 DOLocationID 示例: {df['DOLocationID'].head().tolist()}")
        print(f"Centroids 键示例: {list(centroids.keys())[:5]}")
        valid_pu = df['PULocationID'].isin(centroids.keys())
        valid_do = df['DOLocationID'].isin(centroids.keys())
        print(f"有效 PULocationID 比例: {valid_pu.mean():.2%}, 有效 DOLocationID 比例: {valid_do.mean():.2%}")
        df['approx_distance'] = df.apply(
            lambda row: haversine_distance(*centroids.get(row['PULocationID'], (0, 0)),
                                           *centroids.get(row['DOLocationID'], (0, 0)))
            if row['PULocationID'] in centroids and row['DOLocationID'] in centroids else np.nan,
            axis=1)
        if df['approx_distance'].isna().all():
            print("警告: 所有 approx_distance 为 NaN，使用默认值 0 填充。")
            df['approx_distance'] = 0
        else:
            df['approx_distance'] = df['approx_distance'].fillna(df['approx_distance'].mean())
        print(f"距离计算后，{service_name} 数据行数: {len(df)}")

    df['total_amount'] = df['fare_amount'] + df['extra'] + df['tip_amount'] + df['tolls_amount'] + df[
        'improvement_surcharge'] + df['congestion_surcharge']
    df = df.dropna(subset=['total_amount', 'tip_amount'])

    df = remove_outliers(df, 'total_amount')
    df = remove_outliers(df, 'tip_amount')

    tip_quantile = df['tip_amount'].quantile(0.8)
    total_quantile = df['total_amount'].quantile(0.8)
    df['high_tip_pickup'] = (df['tip_amount'] >= tip_quantile).astype(int)
    df['high_total_pickup'] = (df['total_amount'] >= total_quantile).astype(int)

    df['fare_amount'] = df['fare_amount'].fillna(0)
    df['trip_distance'] = df['trip_distance'].fillna(0)
    df['passenger_count'] = df['passenger_count'].fillna(1)
    df['payment_type'] = df['payment_type'].fillna(2)
    df['RatecodeID'] = df['RatecodeID'].fillna(1)

    features = ['time_period', 'precipitation', 'thermal_comfort', 'solar_radiation', 'wind_dynamics',
                'snow_accumulation', 'dropoff_category', 'trip_distance', 'passenger_count', 'payment_type',
                'RatecodeID', 'approx_distance']
    X = df[features]
    y_tip = df['high_tip_pickup']
    y_total = df['high_total_pickup']

    return X, y_tip, y_total


def build_and_evaluate_model(X, y, model_type, cv=3):
    """构建模型并进行交叉验证"""
    categorical_features = [col for col in X.columns if
                            X[col].dtype == 'object' or col in ['time_period', 'dropoff_category', 'payment_type',
                                                                'RatecodeID']]
    numeric_features = [col for col in X.columns if col not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())]),
             numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )

    # 策略1: 调整类别权重
    if model_type == 'random_forest':
        model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier',
                                 RandomForestClassifier(n_estimators=200, max_depth=15, n_jobs=-1, random_state=42,
                                                        class_weight='balanced'))])
    elif model_type == 'gradient_boosting':
        model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier',
                                 GradientBoostingClassifier(n_estimators=200, max_depth=15, learning_rate=0.05,
                                                            random_state=42))])

    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"\n{model_type.upper()} Model - Cross-validation Accuracy scores: {scores}")
    print(f"Average Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    # 策略2: 手动调整预测阈值
    y_proba = model.predict_proba(X_test)
    threshold = 0.9
    y_pred_adjusted = (y_proba[:, 1] > threshold).astype(int)

    # 评估调整后的结果
    precision_adj = precision_score(y_test, y_pred_adjusted, zero_division=0)
    recall_adj = recall_score(y_test, y_pred_adjusted, zero_division=0)

    print(f"\n--- 使用调整阈值 ({threshold:.2f}) 的评估结果 ---")
    print(f"查准率 (Precision): {precision_adj:.4f}")
    print(f"查全率 (Recall): {recall_adj:.4f}")
    print(f"完整分类报告:")
    print(classification_report(y_test, y_pred_adjusted))

    return model


def main():
    """主函数，执行预测模型训练"""
    print("--- 开始进行高价值地点预测分析 ---")

    centroids = load_zone_centroids()

    service = 'Green Taxi'
    print(f"\n#####################################################")
    print(f"###  正在处理 {service} 数据  ###")

    weather_df = load_data(WEATHER_MERGED_PATH, service)
    pickup_poi_df = load_data(FUSED_POI_PICKUP_PATH, service)
    dropoff_poi_df = load_data(FUSED_POI_DROPOFF_PATH, service)
    anomaly_df = load_data(ANOMALY_PATH, service)

    if weather_df is None or pickup_poi_df is None or dropoff_poi_df is None:
        print("数据文件不完整，跳过此服务。")
        return

    merged_df = pd.merge(weather_df, pickup_poi_df[['category']], left_index=True, right_index=True, how='left')
    merged_df.rename(columns={'category': 'pickup_category'}, inplace=True)
    final_df = pd.merge(merged_df, dropoff_poi_df[['category']], left_index=True, right_index=True, how='left')
    final_df.rename(columns={'category': 'dropoff_category'}, inplace=True)

    X, y_tip, y_total = prepare_prediction_data(final_df, service, centroids, anomaly_df)

    if X is None or y_tip is None or y_total is None or X.empty:
        print(f"警告: {service} 数据为空或无效，跳过模型训练。")
        return

    print(f"\n--- 预测 {service} 高小费地点 ---")
    rf_tip_model = build_and_evaluate_model(X, y_tip, 'random_forest')

    print(f"\n--- 预测 {service} 高总收入地点 ---")
    rf_total_model = build_and_evaluate_model(X, y_total, 'random_forest')

    print(f"#####################################################")


if __name__ == "__main__":
    main()
