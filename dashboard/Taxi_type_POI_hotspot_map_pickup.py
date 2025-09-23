# 引入需要的函式庫
import pandas as pd
import folium
from folium.plugins import MarkerCluster
import geopandas as gpd
from pyproj import Transformer

# --- 步驟一：資料讀取與準備 ---

# 設定檔案路徑
yellow_path = "/Users/rosechiu/Desktop/大学/大三上/大數據原理與系統/data/yellow_tripdata_2024-12.parquet"
green_path = "/Users/rosechiu/Desktop/大学/大三上/大數據原理與系統/data/green_tripdata_2024-12_cleaned.parquet"
fhv_path = "/Users/rosechiu/Desktop/大学/大三上/大數據原理與系統/data/fhv_tripdata_2024-12_cleaned.parquet"
shapefile_path = "/Users/rosechiu/Desktop/大学/大三上/大數據原理與系統/data/taxi_zones/taxi_zones.shp"

# 載入黃色、綠色和FHV出租車資料
df_yellow = pd.read_parquet(yellow_path)
df_green = pd.read_parquet(green_path)
df_fhv = pd.read_parquet(fhv_path)
print("All taxi data loaded.")

# 讀入地理區域 Shapefile
zones = gpd.read_file(shapefile_path)
zones['centroid_lon'] = zones.geometry.centroid.x
zones['centroid_lat'] = zones.geometry.centroid.y
zone_coords = zones.groupby('LocationID')[['centroid_lon', 'centroid_lat']].mean().to_dict('index')
transformer = Transformer.from_crs("EPSG:2263", "EPSG:4326", always_xy=True)
print("Taxi zone coordinates loaded and processed.")

# --- 步驟二：處理各服務類型資料 (新增了抽樣步驟) ---

def process_data(df, service_type, pickup_col, sample_size=50000):
    """
    處理單一數據集，轉換經緯度並抽樣，以優化效能。
    """
    df_prep = df.copy()
    
    # 進行隨機抽樣
    if len(df_prep) > sample_size:
        df_prep = df_prep.sample(n=sample_size, random_state=42) # 使用固定種子確保每次運行結果一致
    
    df_prep['service_type'] = service_type
    
    # 將 PULocationID 轉換為經緯度
    df_prep['pickup_lon'] = df_prep[pickup_col].map(lambda x: zone_coords.get(x, {}).get('centroid_lon'))
    df_prep['pickup_lat'] = df_prep[pickup_col].map(lambda x: zone_coords.get(x, {}).get('centroid_lat'))
    df_prep.dropna(subset=['pickup_lat', 'pickup_lon'], inplace=True)
    df_prep['pickup_lon'], df_prep['pickup_lat'] = transformer.transform(df_prep['pickup_lon'].values, df_prep['pickup_lat'].values)
    
    # 這裡我們只關注上車點
    return df_prep[['pickup_lat', 'pickup_lon', 'service_type']]

# 分別處理三種服務的資料
df_yellow_prep = process_data(df_yellow, 'Yellow', 'PULocationID')
df_green_prep = process_data(df_green, 'Green', 'PULocationID')
df_fhv_prep = process_data(df_fhv, 'FHV', 'PUlocationID')
print("All data prepared and coordinates converted.")
print(f"Sampled data sizes: Yellow={len(df_yellow_prep)}, Green={len(df_green_prep)}, FHV={len(df_fhv_prep)}")

# --- 步驟三：建立地圖與聚類圖層 ---

nyc_coords = [40.75, -73.98]
m = folium.Map(location=nyc_coords, zoom_start=12, tiles='cartodb positron')

# 創建三個 FeatureGroup，每個代表一個服務類型
fg_yellow = folium.FeatureGroup(name='黃色計程車 (Yellow Taxi)').add_to(m)
fg_green = folium.FeatureGroup(name='綠色計程車 (Green Taxi)').add_to(m)
fg_fhv = folium.FeatureGroup(name='FHV (Uber/Lyft)').add_to(m)

# 為每個服務類型建立一個 MarkerCluster
mc_yellow = MarkerCluster(name='黃色計程車載客點').add_to(fg_yellow)
mc_green = MarkerCluster(name='綠色計程車載客點').add_to(fg_green)
mc_fhv = MarkerCluster(name='FHV載客點').add_to(fg_fhv)

# 將各服務類型的載客點添加到對應的 MarkerCluster 中
for lat, lon in zip(df_yellow_prep['pickup_lat'], df_yellow_prep['pickup_lon']):
    folium.Marker([lat, lon]).add_to(mc_yellow)
for lat, lon in zip(df_green_prep['pickup_lat'], df_green_prep['pickup_lon']):
    folium.Marker([lat, lon]).add_to(mc_green)
for lat, lon in zip(df_fhv_prep['pickup_lat'], df_fhv_prep['pickup_lon']):
    folium.Marker([lat, lon]).add_to(mc_fhv)

print("Service type markers added to their respective layers.")

# --- 步驟四：添加圖層控制 ---
# 這將生成一個下拉式選單或複選框，讓使用者切換圖層
folium.LayerControl().add_to(m)
print("Layer control added.")

# --- 步驟五：儲存地圖為 HTML 檔案 ---
output_path = './service_type_distribution_cluster_optimized.html'
m.save(output_path)
print(f"Interactive map with optimized clustering saved to {output_path}")
