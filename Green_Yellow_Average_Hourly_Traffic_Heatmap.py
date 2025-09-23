# 引入需要的函式庫
import pandas as pd
import folium
from folium.plugins import HeatMapWithTime
import geopandas as gpd
from pyproj import Transformer

# --- 步驟一：資料讀取與準備 ---

# 設定檔案路徑
yellow_path = "/Users/rosechiu/Desktop/大学/大三上/大數據原理與系統/data/yellow_tripdata_2024-12.parquet"
green_path = "/Users/rosechiu/Desktop/大学/大三上/大數據原理與系統/data/green_tripdata_2024-12_cleaned.parquet"
shapefile_path = "/Users/rosechiu/Desktop/大学/大三上/大數據原理與系統/data/taxi_zones/taxi_zones.shp"

# 載入資料
df_yellow = pd.read_parquet(yellow_path)
df_green = pd.read_parquet(green_path)
print("Yellow and Green taxi data loaded.")

# 讀入地理區域 Shapefile
zones = gpd.read_file(shapefile_path)
zones['centroid_lon'] = zones.geometry.centroid.x
zones['centroid_lat'] = zones.geometry.centroid.y
zone_coords = zones.groupby('LocationID')[['centroid_lon', 'centroid_lat']].mean().to_dict('index')
transformer = Transformer.from_crs("EPSG:2263", "EPSG:4326", always_xy=True)
print("Taxi zone coordinates loaded and processed.")

# --- 步驟二：處理黃色計程車資料 ---
df_yellow_prep = df_yellow[['tpep_pickup_datetime', 'PULocationID']].copy()
df_yellow_prep.rename(columns={'tpep_pickup_datetime': 'pickup_datetime'}, inplace=True)
df_yellow_prep['pickup_lon'] = df_yellow_prep['PULocationID'].map(lambda x: zone_coords.get(x, {}).get('centroid_lon'))
df_yellow_prep['pickup_lat'] = df_yellow_prep['PULocationID'].map(lambda x: zone_coords.get(x, {}).get('centroid_lat'))
df_yellow_prep.dropna(subset=['pickup_lat', 'pickup_lon'], inplace=True)
df_yellow_prep['pickup_lon'], df_yellow_prep['pickup_lat'] = transformer.transform(df_yellow_prep['pickup_lon'].values, df_yellow_prep['pickup_lat'].values)
df_yellow_prep['hour'] = df_yellow_prep['pickup_datetime'].dt.hour
print("Yellow taxi data prepared for heatmap.")

# --- 步驟三：處理綠色計程車資料 ---
df_green_prep = df_green[['lpep_pickup_datetime', 'PULocationID']].copy()
df_green_prep.rename(columns={'lpep_pickup_datetime': 'pickup_datetime'}, inplace=True)
df_green_prep['pickup_lon'] = df_green_prep['PULocationID'].map(lambda x: zone_coords.get(x, {}).get('centroid_lon'))
df_green_prep['pickup_lat'] = df_green_prep['PULocationID'].map(lambda x: zone_coords.get(x, {}).get('centroid_lat'))
df_green_prep.dropna(subset=['pickup_lat', 'pickup_lon'], inplace=True)
df_green_prep['pickup_lon'], df_green_prep['pickup_lat'] = transformer.transform(df_green_prep['pickup_lon'].values, df_green_prep['pickup_lat'].values)
df_green_prep['hour'] = df_green_prep['pickup_datetime'].dt.hour
print("Green taxi data prepared for heatmap.")

# --- 步驟四：生成黃色計程車熱力圖 ---
data_by_time_yellow = []
time_index = [f'{hour}:00' for hour in range(24)]
for hour in range(24):
    df_hour_yellow = df_yellow_prep[df_yellow_prep['hour'] == hour]
    locations_yellow = df_hour_yellow[['pickup_lat', 'pickup_lon']].values.tolist()
    data_by_time_yellow.append([[loc[0], loc[1], 1] for loc in locations_yellow])

m_yellow = folium.Map(location=[40.75, -73.98], zoom_start=12, tiles='cartodb positron')
hm_yellow = HeatMapWithTime(data_by_time_yellow, index=time_index, auto_play=False, name='黃色計程車載客熱力圖', radius=30)
hm_yellow.add_to(m_yellow)
m_yellow.save('./yellow_taxi_heatmap.html')
print("Yellow taxi heatmap saved.")

# --- 步驟五：生成綠色計程車熱力圖 ---
data_by_time_green = []
for hour in range(24):
    df_hour_green = df_green_prep[df_green_prep['hour'] == hour]
    locations_green = df_hour_green[['pickup_lat', 'pickup_lon']].values.tolist()
    data_by_time_green.append([[loc[0], loc[1], 1] for loc in locations_green])

m_green = folium.Map(location=[40.75, -73.98], zoom_start=12, tiles='cartodb positron')
hm_green = HeatMapWithTime(data_by_time_green, index=time_index, auto_play=False, name='綠色計程車載客熱力圖', radius=30)
hm_green.add_to(m_green)
m_green.save('./green_taxi_heatmap.html')
print("Green taxi heatmap saved.")