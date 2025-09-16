import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd

# --- 定義檔案路徑 ---
yellow_file = 'yellow_tripdata_2024-12_cleaned.parquet'
green_file = 'green_tripdata_2024-12_cleaned.parquet'
fhv_file = 'fhv_tripdata_2024-12_cleaned.parquet'
taxi_zones_file = 'taxi_zones/taxi_zones.shp'

# --- 定義服務顏色 ---
SERVICE_COLORS = {
    'yellow': '#FFD700',  # 金黃色
    'green': '#32CD32',  # 亮綠色
    'fhv': '#FF4500'  # 橙紅色
}


def analyze_single_service(service_name, file_path):
    """
    對單一服務模式進行描述性統計分析。
    """
    print(f"--- 正在分析 {service_name} 服務 ---")

    if not os.path.exists(file_path):
        print(f"錯誤：找不到檔案 {file_path}，請檢查路徑。")
        return None, None, None, None

    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"錯誤：讀取檔案 {file_path} 時發生問題：{e}")
        return None, None, None, None

    # 統一日期時間欄位名稱
    if service_name == 'yellow':
        df = df.rename(columns={'tpep_pickup_datetime': 'pickup_datetime', 'tpep_dropoff_datetime': 'dropoff_datetime'})
    elif service_name == 'green':
        df = df.rename(columns={'lpep_pickup_datetime': 'pickup_datetime', 'lpep_dropoff_datetime': 'dropoff_datetime'})
    elif service_name == 'fhv':
        df = df.rename(columns={'PUlocationID': 'PULocationID', 'dropOff_datetime': 'dropoff_datetime'})

    # 確保日期時間格式正確
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
    df = df.dropna(subset=['pickup_datetime'])

    # --- 時間節律分析 (Temporal Dynamics) ---
    print(f"  > 進行時間節律分析...")
    df['hour'] = df['pickup_datetime'].dt.hour
    df['weekday'] = df['pickup_datetime'].dt.dayofweek

    hourly_counts = df.groupby('hour').size().reset_index(name='trip_count')
    weekly_counts = df.groupby('weekday').size().reset_index(name='trip_count')

    print("  > 時間節律分析完成。")

    # --- 服務畫像分析 (Service Profile) ---
    print(f"  > 進行服務畫像分析...")
    # 這裡我們只在 dropoff_datetime 存在時才計算 trip_duration_minutes
    if 'dropoff_datetime' in df.columns:
        df['trip_duration_minutes'] = (pd.to_datetime(df['dropoff_datetime']) - df[
            'pickup_datetime']).dt.total_seconds() / 60
    else:
        df['trip_duration_minutes'] = None  # 或者你可以選擇一個默認值，例如 0

    profile = {
        'service': service_name,
        'avg_trip_distance': df.get('trip_distance').mean() if 'trip_distance' in df.columns else None,
        'avg_trip_duration_minutes': df['trip_duration_minutes'].mean() if 'trip_duration_minutes' in df.columns and df[
            'trip_duration_minutes'].notna().any() else None,
        'total_trips': len(df)
    }

    print("  > 服務畫像分析完成。")
    print("----------------------------------\n")

    return hourly_counts, weekly_counts, profile, df


def plot_temporal_dynamics(all_hourly_data):
    """
    生成動態的時間節律對比圖 (每小時)。
    """
    print("\n生成每小時動態圖...")
    fig = go.Figure()

    # 確保按照特定的順序繪製，以便疊加圖層（如果需要）
    # 這裡對於線圖，順序不是關鍵，但統一處理會更好
    for service_name in ['yellow', 'fhv', 'green']:  # 例如，讓Green最後繪製，確保其線條可見
        if service_name in all_hourly_data:
            df = all_hourly_data[service_name]
            fig.add_trace(go.Scatter(
                x=df['hour'],
                y=df['trip_count'],
                mode='lines+markers',
                name=service_name.capitalize(),
                line=dict(color=SERVICE_COLORS.get(service_name, '#000000')),  # 設置線條顏色
                marker=dict(color=SERVICE_COLORS.get(service_name, '#000000'))  # 設置標記顏色
            ))

    fig.update_layout(
        title='各服務模式每小時訂單量動態對比',
        xaxis_title='小時 (Hour)',
        yaxis_title='訂單量 (Trip Count)',
        hovermode='x unified'  # 懸停時顯示所有數據點資訊
    )

    fig.write_html("temporal_dynamics_hourly.html")  # 檔名區分一下
    fig.show()
    print("每小時動態圖表已生成，請在瀏覽器中查看。")
    print(f"檔案位置: {os.path.join(os.getcwd(), 'temporal_dynamics_hourly.html')}")


def plot_weekly_dynamics(all_weekly_data):
    """
    生成每週的時間節律對比圖 (每日)。
    """
    print("\n生成每週動態圖...")
    fig = go.Figure()

    # 定義週幾標籤
    weekday_labels = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日']

    # 為了讓數量較少的 Green 服務在堆疊柱狀圖中可見，我們將其最後添加
    # 這樣它的柱子會在最頂層
    sorted_services = ['yellow', 'fhv', 'green']

    for service_name in sorted_services:
        if service_name in all_weekly_data:
            df = all_weekly_data[service_name]
            # 確保順序正確
            df['weekday'] = df['weekday'].astype('category').cat.set_categories(range(7))
            df = df.sort_values('weekday')

            fig.add_trace(go.Bar(
                x=[weekday_labels[i] for i in df['weekday']],
                y=df['trip_count'],
                name=service_name.capitalize(),
                marker_color=SERVICE_COLORS.get(service_name, '#000000')  # 設置柱子顏色
            ))

    fig.update_layout(
        title='各服務模式每週每日訂單量動態對比',
        xaxis_title='週幾 (Weekday)',
        yaxis_title='訂單量 (Trip Count)',
        barmode='stack',  # 將柱狀圖改為堆疊模式，讓小數量的Green也能看見
        hovermode='x unified'  # 懸停時顯示所有數據點資訊
    )

    fig.write_html("temporal_dynamics_weekly.html")  # 檔名區分一下
    fig.show()
    print("每週動態圖表已生成，請在瀏覽器中查看。")
    print(f"檔案位置: {os.path.join(os.getcwd(), 'temporal_dynamics_weekly.html')}")


def analyze_airport_trips(yellow_df, green_df, fhv_df):
    """
    分析三種服務在機場的上車行為差異。
    """
    print("\n--- 正在進行機場服務分析 ---")

    if not os.path.exists(taxi_zones_file):
        print(f"錯誤：找不到檔案 {taxi_zones_file}，無法進行機場分析。")
        return

    try:
        gdf_zones = gpd.read_file(taxi_zones_file)
    except Exception as e:
        print(f"錯誤：讀取 {taxi_zones_file} 時發生問題：{e}")
        return

    airport_zones = gdf_zones[gdf_zones['zone'].str.contains('JFK|LaGuardia|Newark', case=False, na=False)]
    airport_ids = airport_zones['LocationID'].tolist()

    if not airport_ids:
        print("警告：未找到機場區域ID，請檢查 taxi_zones.shp 的 'zone' 欄位。")
        return

    airport_trips_data = {}

    for service_name, df in [('yellow', yellow_df), ('green', green_df), ('fhv', fhv_df)]:
        # 檢查 df 是否為 None
        if df is None:
            print(f"警告：{service_name} 服務的 DataFrame 為空，跳過機場分析。")
            continue

        trips = df[df['PULocationID'].isin(airport_ids)]
        airport_trips_data[service_name] = trips

        total_trips = len(trips)
        avg_distance = trips['trip_distance'].mean() if 'trip_distance' in trips.columns else None

        print(f"\n服務：{service_name.capitalize()}")
        print(f"  > 機場上車總行程數：{total_trips}")
        if avg_distance is not None:
            print(f"  > 平均行程距離：{avg_distance:.2f} 英里")
        else:
            print(f"  > 平均行程距離：不適用")
    print("----------------------------------\n")


def main():
    """
    主函式，執行所有服務的獨立分析、繪圖與深入分析。
    """
    all_hourly_data = {}
    all_weekly_data = {}
    all_profiles = []

    all_dfs = {}  # 用於儲存原始 DataFrame 以供後續分析

    # 獨立分析每個服務
    # Yellow
    hourly, weekly, profile, df = analyze_single_service('yellow', yellow_file)
    if hourly is not None:
        all_hourly_data['yellow'] = hourly
        all_weekly_data['yellow'] = weekly
        all_profiles.append(profile)
        all_dfs['yellow'] = df

    # Green
    hourly, weekly, profile, df = analyze_single_service('green', green_file)
    if hourly is not None:
        all_hourly_data['green'] = hourly
        all_weekly_data['green'] = weekly
        all_profiles.append(profile)
        all_dfs['green'] = df

    # FHV
    hourly, weekly, profile, df = analyze_single_service('fhv', fhv_file)
    if hourly is not None:
        all_hourly_data['fhv'] = hourly
        all_weekly_data['fhv'] = weekly
        all_profiles.append(profile)
        all_dfs['fhv'] = df

    # 檢查是否所有必要的服務數據都已加載
    if not all_dfs:
        print("沒有任何服務數據成功加載，分析結束。")
        return

    # --- 列印所有服務的結果 ---
    print("### 所有服務的時間節律分析結果 ###")
    for service, df_hourly in all_hourly_data.items():
        print(f"\n服務: {service.capitalize()}")
        print("> 每小時訂單量:")
        print(df_hourly.to_string(index=False))

    for service, df_weekly in all_weekly_data.items():
        print(f"\n服務: {service.capitalize()}")
        print("> 每週每天訂單量:")
        print(df_weekly.to_string(index=False))

    print("\n\n### 所有服務的服務畫像結果 ###")
    profiles_df = pd.DataFrame(all_profiles)
    print(profiles_df.to_string(index=False))

    # --- 繪製動態圖表 ---
    if all_hourly_data:
        plot_temporal_dynamics(all_hourly_data)
    else:
        print("\n沒有足夠的每小時數據來繪製時間節律圖。")

    if all_weekly_data:
        plot_weekly_dynamics(all_weekly_data)
    else:
        print("\n沒有足夠的每週數據來繪製時間節律圖。")

    # --- 進行更深入的分析 ---
    # 只有當所有三個服務的DataFrame都成功加載時才進行機場分析
    if 'yellow' in all_dfs and 'green' in all_dfs and 'fhv' in all_dfs:
        analyze_airport_trips(all_dfs['yellow'], all_dfs['green'], all_dfs['fhv'])
    else:
        print("\n無法進行機場分析，因為部分或所有數據文件未找到或加載失敗。")

    print("\n分析程式運行結束。")


if __name__ == "__main__":
    main()