import pandas as pd
import geopandas as gpd
from pyproj import Transformer

# 設定 Pandas 顯示的最大欄位數為 None（無限制）
pd.set_option('display.max_columns', None)

# -----------------------------
# 1️⃣ 讀入地理區域 Shapefile
# -----------------------------
shapefile_path = "/Users/rosechiu/Desktop/大学/大三上/大數據原理與系統/data/taxi_zones/taxi_zones.shp"
zones = gpd.read_file(shapefile_path)

# 計算每個區域中心點（centroid）
zones['centroid_lon'] = zones.geometry.centroid.x
zones['centroid_lat'] = zones.geometry.centroid.y

# 對每個 LocationID 取 centroid 坐標
zone_coords = zones.groupby('LocationID')[['centroid_lon', 'centroid_lat']].mean().to_dict('index')

# -----------------------------
# 2️⃣ 讀入綠色計程車 parquet
# -----------------------------
# 修正後的檔案路徑，確保與你電腦上的路徑一致
green_path = "/Users/rosechiu/Desktop/大学/大三上/大數據原理與系統/data/原/green_tripdata_2024-12.parquet"
# 變數名稱也改為 df_green，更具可讀性
df_green = pd.read_parquet(green_path)

# 篩選單日，例如 2024-12-25
# 注意：這裡的欄位名稱已經從 tpep_ 改為 lpep_
df_green['lpep_pickup_datetime'] = pd.to_datetime(df_green['lpep_pickup_datetime'])
df_green['lpep_dropoff_datetime'] = pd.to_datetime(df_green['lpep_dropoff_datetime'])
df_green = df_green[
    (df_green['lpep_pickup_datetime'].dt.date == pd.to_datetime('2024-12-12').date())
]

# -----------------------------
# 3️⃣ 加入區域經緯度（EPSG:2263 → WGS84 EPSG:4326）
# -----------------------------
# 修正後的變數名稱
df_green['pickup_x'] = df_green['PULocationID'].map(lambda x: zone_coords.get(x, {}).get('centroid_lon'))
df_green['pickup_y'] = df_green['PULocationID'].map(lambda x: zone_coords.get(x, {}).get('centroid_lat'))
df_green['dropoff_x'] = df_green['DOLocationID'].map(lambda x: zone_coords.get(x, {}).get('centroid_lon'))
df_green['dropoff_y'] = df_green['DOLocationID'].map(lambda x: zone_coords.get(x, {}).get('centroid_lat'))

# 去掉缺失值
df_green = df_green.dropna(subset=['pickup_x','pickup_y','dropoff_x','dropoff_y'])

# EPSG:2263 → WGS84 EPSG:4326
transformer = Transformer.from_crs("EPSG:2263", "EPSG:4326", always_xy=True)
df_green['pickup_lon'], df_green['pickup_lat'] = transformer.transform(df_green['pickup_x'].values, df_green['pickup_y'].values)
df_green['dropoff_lon'], df_green['dropoff_lat'] = transformer.transform(df_green['dropoff_x'].values, df_green['dropoff_y'].values)

# -----------------------------
# 4️⃣ 選擇欄位輸出 CSV
# -----------------------------
# 修正後的欄位名稱
df_out = df_green[['lpep_pickup_datetime','lpep_dropoff_datetime',
                    'PULocationID','DOLocationID',
                    'pickup_lon','pickup_lat','dropoff_lon','dropoff_lat']]

# 修正後的輸出檔名，讓它與讀入的資料集一致
df_out.to_csv("/Users/rosechiu/Desktop/大学/大三上/大數據原理與系統/data/green_2024-12-12_wgs84.csv", index=False)

print("單日 CSV 輸出完成 ✅")