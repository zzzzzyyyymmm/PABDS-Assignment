from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Define feature lists before loading models
PASSENGER_NUM_FEATURES = ['trip_distance', 'passenger_count', 'approx_distance', 'hour']
PASSENGER_CAT_FEATURES = ['time_period', 'precipitation', 'thermal_comfort', 'solar_radiation', 'wind_dynamics', 'snow_accumulation', 'pickup_category', 'dropoff_category', 'payment_type', 'RatecodeID']
DRIVER_NUM_FEATURES = ['trip_distance', 'passenger_count', 'approx_distance']
DRIVER_CAT_FEATURES = ['time_period', 'precipitation', 'thermal_comfort', 'solar_radiation', 'wind_dynamics', 'snow_accumulation', 'dropoff_category', 'payment_type', 'RatecodeID']

PASSENGER_FEATURES = [
    'time_period', 'precipitation', 'thermal_comfort', 'solar_radiation', 'wind_dynamics',
    'snow_accumulation', 'pickup_category', 'dropoff_category', 'trip_distance',
    'passenger_count', 'payment_type', 'RatecodeID', 'approx_distance', 'hour'
]
PASSENGER_DTYPES = {
    'time_period': str, 'pickup_category': str, 'dropoff_category': str,
    'payment_type': float, 'RatecodeID': float, 'passenger_count': float,
    'precipitation': float, 'thermal_comfort': float, 'solar_radiation': float, 'wind_dynamics': float,
    'snow_accumulation': float, 'trip_distance': float, 'approx_distance': float, 'hour': float
}

DRIVER_FEATURES = [
    'time_period', 'precipitation', 'thermal_comfort', 'solar_radiation', 'wind_dynamics',
    'snow_accumulation', 'dropoff_category', 'trip_distance', 'passenger_count',
    'payment_type', 'RatecodeID', 'approx_distance'
]
DRIVER_DTYPES = {
    'time_period': str, 'dropoff_category': str,
    'payment_type': float, 'RatecodeID': float, 'passenger_count': float,
    'precipitation': float, 'thermal_comfort': float, 'solar_radiation': float, 'wind_dynamics': float,
    'snow_accumulation': float, 'trip_distance': float, 'approx_distance': float
}

# --- 模型加载 ---
try:
    gb_duration_model = pickle.load(open('gb_duration_model.pkl', 'rb'))
    gb_fare_model = pickle.load(open('gb_fare_model.pkl', 'rb'))
    gb_tip_model = pickle.load(open('gb_tip_model.pkl', 'rb'))
    gb_total_model = pickle.load(open('gb_total_model.pkl', 'rb'))
    print("✅ 所有模型加载成功！")

    # Replace the num pipeline with the fitted scaler (remove imputer)
    for model in [gb_duration_model, gb_fare_model, gb_tip_model, gb_total_model]:
        preprocessor = model.named_steps['preprocessor']
        num_transformer = preprocessor.named_transformers_['num']
        fitted_scaler = num_transformer.named_steps['scaler']
        new_num_pipeline = Pipeline([('scaler', fitted_scaler)])
        preprocessor.transformers_[0] = ('num', new_num_pipeline, preprocessor.transformers_[0][2])
        preprocessor.named_transformers_['num'] = new_num_pipeline

    # Modify ColumnTransformer to use indices instead of names for passenger models
    for model in [gb_duration_model, gb_fare_model]:
        preprocessor = model.named_steps['preprocessor']
        num_indices = [PASSENGER_FEATURES.index(f) for f in PASSENGER_NUM_FEATURES]
        cat_indices = [PASSENGER_FEATURES.index(f) for f in PASSENGER_CAT_FEATURES]
        preprocessor.transformers_ = [
            ('num', preprocessor.named_transformers_['num'], num_indices),
            ('cat', preprocessor.named_transformers_['cat'], cat_indices)
        ]

    # Modify ColumnTransformer to use indices for driver models
    for model in [gb_tip_model, gb_total_model]:
        preprocessor = model.named_steps['preprocessor']
        num_indices = [DRIVER_FEATURES.index(f) for f in DRIVER_NUM_FEATURES]
        cat_indices = [DRIVER_FEATURES.index(f) for f in DRIVER_CAT_FEATURES]
        preprocessor.transformers_ = [
            ('num', preprocessor.named_transformers_['num'], num_indices),
            ('cat', preprocessor.named_transformers_['cat'], cat_indices)
        ]
except FileNotFoundError as e:
    print(f"❌ 模型文件加载失败: {e}")
    exit()

# --- 映射字典：处理界面输入的中文到模型期望的英文/数值 (核心修正以避免类型不匹配和NaN) ---
TIME_PERIOD_MAP = {
    '工作日早高峰': 'Workday Morning Rush Hour',
    '工作日晚高峰': 'Workday Evening Rush Hour',
    '工作日其他': 'Workday Other',
    '周末': 'Weekend'
}

PAYMENT_TYPE_MAP = {
    '信用卡': 1,
    '现金': 2,
    '无费用': 3,
    '争议': 4
}

# 假设模型训练中POI category为中文（基于提供的POI_TYPES），无需映射；若为英文，可在此添加映射
POI_CATEGORY_MAP = {  # 如果训练数据是英文，这里添加映射，如 '交通': 'Transportation'
    '交通': '交通',
    '食品': '食品',
    '购物': '购物',
    '生活服务': '生活服务',
    '医疗': '医疗',
    '文娱': '文娱',
    '教育': '教育',
    '住宿': '住宿',
    '旅游景点': '旅游景点',
    '基础设施': '基础设施',
    '未知': 'Unknown'
}

# --- 全局配置与辅助函数 (无变动) ---
POI_TYPES = ['交通', '食品', '购物', '生活服务', '医疗', '文娱', '教育', '住宿', '旅游景点', '基础设施']
POI_LOCATIONS = {
    '交通': [(40.6413, -73.7781), (40.7769, -73.8740)], '食品': [(40.7580, -73.9855), (40.7128, -74.0060)],
    '购物': [(40.7128, -74.0134), (40.7484, -73.9857)], '生活服务': [(40.7831, -73.9712), (40.6782, -73.9442)],
    '医疗': [(40.7580, -73.9855), (40.7128, -74.0060)], '文娱': [(40.7128, -74.0134), (40.7484, -73.9857)],
    '教育': [(40.7831, -73.9712), (40.6782, -73.9442)], '住宿': [(40.7580, -73.9855), (40.7128, -74.0060)],
    '旅游景点': [(40.7128, -74.0134), (40.7484, -73.9857)], '基础设施': [(40.7831, -73.9712), (40.6782, -73.9442)],
    'Unknown': [(40.7128, -74.0060)]
}


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371;
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1;
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a));
    return R * c


def find_nearest_poi_coords(poi_type, current_lat, current_lon):
    if poi_type not in POI_LOCATIONS: return POI_LOCATIONS['Unknown'][0]
    locations = POI_LOCATIONS[poi_type]
    distances = [haversine_distance(current_lat, current_lon, lat, lon) for lat, lon in locations]
    return locations[np.argmin(distances)]


def get_hour_from_time_period(time_period):
    return {'Workday Morning Rush Hour': 8, 'Workday Evening Rush Hour': 17, 'Workday Other': 12, 'Weekend': 14}.get(
        time_period, 12)


# --- 主路由 ---
@app.route('/', methods=['GET', 'POST'])
def index():
    passenger_result, driver_result = None, None
    driver_lat, driver_lon = 40.7128, -74.0060  # 预设固定坐标
    driver_poi_type = '旅游景点'  # 预设固定POI类型（演示用，实际查询不使用）
    driver_coords_str = f"({driver_lat:.4f}, {driver_lon:.4f})，POI类型: {driver_poi_type}"

    if request.method == 'POST':
        if 'passenger_submit' in request.form:
            try:
                form_data = request.form
                # 映射中文输入到模型期望值
                time_period_en = TIME_PERIOD_MAP.get(form_data['time_period'], 'Workday Other')
                payment_type_int = PAYMENT_TYPE_MAP.get(form_data['payment_type'], 1)
                pickup_category = POI_CATEGORY_MAP.get(form_data['pickup_category'], 'Unknown')
                dropoff_category = POI_CATEGORY_MAP.get(form_data['dropoff_category'], 'Unknown')

                feature_dict = {
                    'time_period': time_period_en,
                    'precipitation': float(form_data.get('precipitation', 0.0)),
                    'thermal_comfort': float(form_data.get('thermal_comfort', 20.0)),
                    'solar_radiation': float(form_data.get('solar_radiation', 100.0)),
                    'wind_dynamics': float(form_data.get('wind_dynamics', 5.0)),
                    'snow_accumulation': float(form_data.get('snow_accumulation', 0.0)),
                    'pickup_category': pickup_category,
                    'dropoff_category': dropoff_category,
                    'trip_distance': 5.0,  # 固定（未输入）
                    'passenger_count': float(form_data.get('passenger_count', 1)),
                    'payment_type': float(payment_type_int),
                    'RatecodeID': 1.0,  # 固定（未输入）
                    'approx_distance': 5.0,  # 固定（未输入）
                    'hour': float(get_hour_from_time_period(time_period_en))
                }

                input_df = pd.DataFrame([feature_dict])
                input_df = input_df.astype(PASSENGER_DTYPES)[PASSENGER_FEATURES]

                # 排查NaN：打印输入以调试
                print("乘客输入DataFrame:\n", input_df)

                input_array = input_df.to_numpy()

                duration_pred = gb_duration_model.predict(input_array)[0]
                fare_pred = gb_fare_model.predict(input_array)[0]
                passenger_result = f"✅ 查询成功！\n预估行程时间: {max(0, duration_pred - 3):.1f} - {duration_pred + 3:.1f} 分钟\n预估行程费用: ${max(0, fare_pred - 5):.2f} - ${fare_pred + 5:.2f} 美元"
            except Exception as e:
                print(f"乘客查询发生错误: {e}")
                passenger_result = f"❌ 乘客查询出错，请检查后台日志。"

        elif 'driver_submit' in request.form:
            try:
                form_data = request.form
                # 映射中文输入到模型期望值
                time_period_en = TIME_PERIOD_MAP.get(form_data['driver_time_period'], 'Workday Other')

                base_feature_dict = {
                    'time_period': time_period_en,
                    'precipitation': float(form_data.get('driver_precipitation', 0.0)),
                    'thermal_comfort': float(form_data.get('driver_thermal_comfort', 20.0)),
                    'solar_radiation': float(form_data.get('driver_solar_radiation', 100.0)),
                    'wind_dynamics': float(form_data.get('driver_wind_dynamics', 5.0)),
                    'snow_accumulation': float(form_data.get('driver_snow_accumulation', 0.0)),
                    'trip_distance': 5.0,  # 固定
                    'passenger_count': 1.0,  # 固定（司机查询未输入乘客数），改为float
                    'payment_type': 1.0,  # 固定（假设信用卡）
                    'RatecodeID': 1.0,  # 固定
                    'approx_distance': 5.0  # 固定
                }
                tip_probabilities, total_probabilities = {}, {}

                for poi in POI_TYPES:
                    current_features = base_feature_dict.copy()
                    current_features['dropoff_category'] = POI_CATEGORY_MAP.get(poi, 'Unknown')

                    input_df = pd.DataFrame([current_features])
                    input_df = input_df.astype(DRIVER_DTYPES)[DRIVER_FEATURES]

                    # 排查NaN：打印输入以调试
                    print(f"司机输入DataFrame for {poi}:\n", input_df)

                    input_array = input_df.to_numpy()

                    tip_probabilities[poi] = gb_tip_model.predict_proba(input_array)[0][1]
                    total_probabilities[poi] = gb_total_model.predict_proba(input_array)[0][1]

                best_poi_for_tip = max(tip_probabilities, key=tip_probabilities.get)
                best_poi_for_total = max(total_probabilities, key=total_probabilities.get)
                nearest_tip_coords = find_nearest_poi_coords(best_poi_for_tip, driver_lat, driver_lon)
                nearest_total_coords = find_nearest_poi_coords(best_poi_for_total, driver_lat, driver_lon)
                driver_result = (
                    f"⭐ **高小费推荐**\n   - **最佳POI类型**: {best_poi_for_tip} (模型预测概率: {tip_probabilities[best_poi_for_tip]:.2%})\n   - **最近地点坐标**: ({nearest_tip_coords[0]:.4f}, {nearest_tip_coords[1]:.4f})\n\n"
                    f"⭐ **高总收入推荐**\n   - **最佳POI类型**: {best_poi_for_total} (模型预测概率: {total_probabilities[best_poi_for_total]:.2%})\n   - **最近地点坐标**: ({nearest_total_coords[0]:.4f}, {nearest_total_coords[1]:.4f})"
                )
            except Exception as e:
                print(f"司机查询发生错误: {e}")
                driver_result = f"❌ 司机查询出错，请检查后台日志。"

    return render_template('index.html', passenger_result=passenger_result, driver_result=driver_result,
                           driver_coords=driver_coords_str)


if __name__ == "__main__":
    app.run(debug=True)
