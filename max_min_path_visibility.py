import pandas as pd
from keplergl import KeplerGl
import json

# --------------------------
# 1. 載入資料集
# --------------------------
print("正在載入四個資料集...")
try:
    df_yellow_12 = pd.read_csv("/Users/rosechiu/Desktop/大学/大三上/大數據原理與系統/data/yellow_2024-12-12_wgs84.csv")
    df_yellow_12['tpep_pickup_datetime'] = pd.to_datetime(df_yellow_12['tpep_pickup_datetime'])

    df_yellow_25 = pd.read_csv("/Users/rosechiu/Desktop/大学/大三上/大數據原理與系統/data/yellow_2024-12-25_wgs84.csv")
    df_yellow_25['tpep_pickup_datetime'] = pd.to_datetime(df_yellow_25['tpep_pickup_datetime'])
    
    df_green_12 = pd.read_csv("/Users/rosechiu/Desktop/大学/大三上/大數據原理與系統/data/green_2024-12-12_wgs84.csv")
    df_green_12['lpep_pickup_datetime'] = pd.to_datetime(df_green_12['lpep_pickup_datetime'])

    df_green_25 = pd.read_csv("/Users/rosechiu/Desktop/大学/大三上/大數據原理與系統/data/green_2024-12-25_wgs84.csv")
    df_green_25['lpep_pickup_datetime'] = pd.to_datetime(df_green_25['lpep_pickup_datetime'])
except FileNotFoundError as e:
    print(f"錯誤：找不到檔案，請確認路徑是否正確。錯誤訊息：{e}")
    exit()
print("資料載入完成！")

# --------------------------
# 2. 定義生成單張地圖的 function
# --------------------------
def create_map_html(df, data_id, arc_color, pickup_col, file_name):
    config = {
        "version": "v1",
        "config": {
            "visState": {
                "filters": [
                    {
                        "id": f"filter_{data_id}_time",
                        "dataId": data_id,
                        "name": pickup_col,
                        "type": "timeRange",
                        "value": [
                            df[pickup_col].min().timestamp()*1000,
                            df[pickup_col].min().timestamp()*1000 + 20*60*1000
                        ],
                        "enlarged": False,
                        "plotType": "histogram"
                    }
                ],
                "layers": [
                    {
                        "id": f"arc_{data_id}",
                        "type": "arc",
                        "config": {
                            "dataId": data_id,
                            "label": data_id,
                            "color": arc_color,
                            "columns": {
                                "lat0": "pickup_latitude",
                                "lng0": "pickup_longitude",
                                "lat1": "dropoff_latitude",
                                "lng1": "dropoff_longitude"
                            },
                            "isVisible": True,
                            "size": 2
                        },
                        "visualChannels": {
                            "color": {"field": {"name":"trip_distance","type":"real"}, "scale":"quantile"}
                        }
                    }
                ],
                "layerOrder": [f"arc_{data_id}"]
            },
            "mapState": {
                "bearing": 0,
                "latitude": 40.7128,
                "longitude": -74.0060,
                "pitch": 0,
                "zoom": 10,
                "is3d": False
            },
            "mapStyle": {
                "styleType": "dark"
            }
        }
    }

    map_obj = KeplerGl(height=400, data={data_id: df}, config=config)
    map_obj.save_to_html(file_name=file_name)

# --------------------------
# 3. 生成四張地圖 HTML
# --------------------------
print("正在生成四張地圖...")
create_map_html(df_yellow_12, "yellow_12_12", [255,203,153], "tpep_pickup_datetime", "map_yellow_12.html")
create_map_html(df_yellow_25, "yellow_12_25", [255,203,153], "tpep_pickup_datetime", "map_yellow_25.html")
create_map_html(df_green_12, "green_12_12", [153,255,203], "lpep_pickup_datetime", "map_green_12.html")
create_map_html(df_green_25, "green_12_25", [153,255,203], "lpep_pickup_datetime", "map_green_25.html")
print("四張地圖生成完成！")

# --------------------------
# 4. 創建總覽儀表板 HTML（iframe 格式）
# --------------------------
with open("multi_map_dashboard.html", "w", encoding="utf-8") as f:
    f.write(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>紐約計程車比較儀表板</title>
        <meta charset="UTF-8">
        <style>
            body {{ margin: 0; padding: 0; font-family: sans-serif; }}
            .map-container {{ display: flex; flex-wrap: wrap; justify-content: space-around; }}
            .map-item {{ width: 49%; margin: 5px; border: 1px solid #ddd; }}
            h2 {{ text-align: center; }}
        </style>
    </head>
    <body>
        <div style="text-align: center; margin: 20px;">
            <h1>紐約計程車服務比較</h1>
            <p>比較 12月12日 (高峰日) 與 12月25日 (節日低谷日) 的出行模式</p>
        </div>
        <div class="map-container">
            <div class="map-item">
                <h2>黃色計程車 (12月12日)</h2>
                <iframe src="map_yellow_12.html" width="100%" height="400px" frameborder="0"></iframe>
            </div>
            <div class="map-item">
                <h2>黃色計程車 (12月25日)</h2>
                <iframe src="map_yellow_25.html" width="100%" height="400px" frameborder="0"></iframe>
            </div>
            <div class="map-item">
                <h2>綠色計程車 (12月12日)</h2>
                <iframe src="map_green_12.html" width="100%" height="400px" frameborder="0"></iframe>
            </div>
            <div class="map-item">
                <h2>綠色計程車 (12月25日)</h2>
                <iframe src="map_green_25.html" width="100%" height="400px" frameborder="0"></iframe>
            </div>
        </div>
    </body>
    </html>
    """)

print("完成！請打開 'multi_map_dashboard.html' 查看四個地圖儀表板。")