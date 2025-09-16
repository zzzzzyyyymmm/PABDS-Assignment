import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os
import numpy as np

# 定义文件路径
taxi_data_path = 'green_tripdata_2024-12_cleaned.parquet'
taxi_zones_path = 'taxi_zones/taxi_zones.shp'
osm_map_path = 'new-york-250907-free/gis_osm_pois_free_1.shp'
output_path = 'fused_pickup_green.parquet'  # 新的输出文件路径

# 全面的 POI 分类字典
FCLASS_TO_CATEGORY_MAPPING = {
    # 交通 (Transportation)
    'airport': '交通',
    'bicycle_rental': '交通',
    'bus_station': '交通',
    'car_rental': '交通',
    'car_sharing': '交通',
    'ferry_terminal': '交通',
    'light_rail_station': '交通',
    'parking': '交通',
    'railway_station': '交通',
    'subway_station': '交通',
    'taxi': '交通',
    'track': '交通',

    # 食品 (Food)
    'bakery': '食品',
    'beverages': '食品',
    'butcher': '食品',
    'cafe': '食品',
    'fast_food': '食品',
    'food_court': '食品',
    'greengrocer': '食品',
    'ice_cream': '食品',
    'restaurant': '食品',
    'supermarket': '食品',

    # 购物 (Shopping)
    'bookshop': '购物',
    'bicycle_shop': '购物',
    'car_dealership': '购物',
    'clothes': '购物',
    'computer_shop': '购物',
    'convenience': '购物',
    'department_store': '购物',
    'doityourself': '购物',
    'florist': '购物',
    'furniture_shop': '购物',
    'general': '购物',
    'gift_shop': '购物',
    'jeweller': '购物',
    'mall': '购物',
    'market_place': '购物',
    'mobile_phone_shop': '购物',
    'outdoor_shop': '购物',
    'retail': '购物',
    'shoe_shop': '购物',
    'sports_shop': '购物',
    'stationery': '购物',
    'toy_shop': '购物',
    'video_shop': '购物',

    # 生活服务 (Daily Services)
    'beauty_shop': '生活服务',
    'car_wash': '生活服务',
    'fuel': '生活服务',
    'hairdresser': '生活服务',
    'laundry': '生活服务',
    'newsagent': '生活服务',
    'optician': '生活服务',
    'travel_agent': '生活服务',
    'vending_any': '生活服务',
    'vending_machine': '生活服务',
    'vending_parking': '生活服务',
    'veterinary': '生活服务',
    'nursing_home': '生活服务',

    # 医疗（Medical）
    'chemist': '医疗',
    'clinic': '医疗',
    'dentist': '医疗',
    'doctors': '医疗',
    'hospital': '医疗',
    'pharmacy': '医疗',

    # 文娱 (Entertainment)
    'arts_centre': '文娱',
    'artwork': '文娱',
    'bar': '文娱',
    'biergarten': '文娱',
    'cinema': '文娱',
    'community_centre': '文娱',
    'library': '文娱',
    'ice_rink': '文娱',
    'museum': '文娱',
    'nightclub': '文娱',
    'playground': '文娱',
    'picnic_site': '文娱',
    'pitch': '文娱',
    'pub': '文娱',
    'sports_centre': '文娱',
    'stadium': '文娱',
    'swimming_pool': '文娱',
    'theatre': '文娱',
    'zoo': '文娱',

    # 教育 (Education)
    'college': '教育',
    'kindergarten': '教育',
    'school': '教育',
    'university': '教育',

    # 住宿 (Accommodation)
    'alpine_hut': '住宿',
    'camp_site': '住宿',
    'caravan_site': '住宿',
    'chalet': '住宿',
    'guesthouse': '住宿',
    'hostel': '住宿',
    'hotel': '住宿',
    'motel': '住宿',

    # 旅游景点 (Tourist Attractions)
    'archaeological': '旅游景点',
    'attraction': '旅游景点',
    'battlefield': '旅游景点',
    'castle': '旅游景点',
    'fountain': '旅游景点',
    'fort': '旅游景点',
    'garden_centre': '旅游景点',
    'hunting_stand': '旅游景点',
    'lighthouse': '旅游景点',
    'memorial': '旅游景点',
    'monument': '旅游景点',
    'observation_tower': '旅游景点',
    'public_building': '旅游景点',
    'ruins': '旅游景点',
    'shelter': '旅游景点',
    'theme_park': '旅游景点',
    'tourist_info': '旅游景点',
    'tower': '旅游景点',
    'viewpoint': '旅游景点',
    'wayside_cross': '旅游景点',
    'wayside_shrine': '旅游景点',
    'windmill': '旅游景点',

    # 基础设施 (Infrastructure)
    'embassy': '基础设施',
    'government': '基础设施',
    'office': '基础设施',
    'town_hall': '基础设施',
    'atm': '基础设施',
    'bank': '基础设施',
    'bench': '基础设施',
    'camera_surveillance': '基础设施',
    'comms_tower': '基础设施',
    'water_tower': '基础设施',
    'dog_park': '基础设施',
    'drinking_water': '基础设施',
    'fire_station': '基础设施',
    'graveyard': '基础设施',
    'park': '基础设施',
    'police': '基础设施',
    'post_box': '基础设施',
    'post_office': '基础设施',
    'prison': '基础设施',
    'recycling': '基础设施',
    'recycling_clothes': '基础设施',
    'recycling_glass': '基础设施',
    'recycling_metal': '基础设施',
    'recycling_paper': '基础设施',
    'telephone': '基础设施',
    'toilet': '基础设施',
    'waste_basket': '基础设施',
    'wastewater_plant': '基础设施',
    'water_works': '基础设施',
    'water_well': '基础设施'
}


def load_and_prepare_data():
    """
    加载所有必要的数据文件,并进行初步的预处理。
    """
    try:
        # 1. 加载出租车数据
        df_taxi = pd.read_parquet(taxi_data_path)
        print("出租车数据加载成功，前5行:")
        print(df_taxi.head())

        # 2. 加载出租车区域(Taxi Zones) Shapefile
        if not os.path.exists(taxi_zones_path):
            print(f"错误: 找不到文件 - {taxi_zones_path}")
            return None, None, None

        gdf_zones = gpd.read_file(taxi_zones_path)
        print("\n出租车区域数据加载成功，列名:")
        print(gdf_zones.columns)

        # 3. 加载 OpenStreetMap POI 数据
        gdf_osm_pois = gpd.read_file(osm_map_path)
        print("\nOpenStreetMap POI 数据加载成功，前5行:")
        print(gdf_osm_pois.head())

        # 确保所有地理数据使用相同的坐标系
        gdf_zones = gdf_zones.to_crs("EPSG:4326")  # 使用通用 WGS84 坐标系
        gdf_osm_pois = gdf_osm_pois.to_crs("EPSG:4326")

        # 对 POI 进行分类和提取经纬度
        gdf_osm_pois['category'] = gdf_osm_pois['fclass'].map(FCLASS_TO_CATEGORY_MAPPING)
        # 提取经纬度
        gdf_osm_pois['longitude'] = gdf_osm_pois.geometry.x
        gdf_osm_pois['latitude'] = gdf_osm_pois.geometry.y
        print("\nPOI 数据已分类并提取经纬度，前5行:")
        print(gdf_osm_pois[['fclass', 'category', 'longitude', 'latitude']].head())

        return df_taxi, gdf_zones, gdf_osm_pois

    except FileNotFoundError as e:
        print(f"错误: 找不到文件 - {e}")
        return None, None, None


def perform_data_fusion(df_taxi, gdf_zones, gdf_osm_pois):
    """
    执行出租车数据与地理数据的融合。
    """
    print("\n开始进行数据融合...")

    # 将出租车数据与区域数据进行合并，获取每个行程的上车点信息
    merged_taxi_zones = df_taxi.merge(gdf_zones, left_on='PUlocationID', right_on='LocationID', how='inner')

    # 将合并后的数据转换为 GeoDataFrame
    gdf_taxi_trips = gpd.GeoDataFrame(
        merged_taxi_zones,
        geometry='geometry',
        crs=gdf_zones.crs
    )

    # 进一步筛选 POI 数据，只保留有分类的 POI
    gdf_poi_filtered = gdf_osm_pois[gdf_osm_pois['category'].notna()]

    # 空间连接：找到每个出租车上车区域附近的 POI
    gdf_taxi_with_pois = gpd.sjoin(
        gdf_poi_filtered,
        gdf_taxi_trips.drop_duplicates(subset=['PUlocationID']),
        how='inner',
        predicate='within'
    )

    print("\n空间连接完成，结果预览:")
    # 打印新添加的 category, longitude, latitude 栏位
    print(gdf_taxi_with_pois[['name', 'fclass', 'category', 'longitude', 'latitude', 'PUlocationID']].head())

    return gdf_taxi_with_pois


def analyze_pickup_poi_types(df):
    """
    统计上车点POI的类别数量和占比。
    """
    print("\n--- 开始统计上车点POI类型 ---")
    if 'category' not in df.columns:
        print("错误: 数据中缺少 'category' 列，无法进行统计。")
        return

    # 统计每个大类的数量
    poi_counts = df['category'].value_counts()
    print("\n上车点POI类别数量:")
    print(poi_counts.sort_values(ascending=False))

    # 统计每个大类的占比
    poi_proportions = df['category'].value_counts(normalize=True).mul(100)
    print("\n上车点POI类别占比 (%):")
    print(poi_proportions.sort_values(ascending=False).round(2).astype(str) + '%')

    print("\n--- 上车点POI类型统计完成 ---")


def main():
    """
    主函数，执行整个数据处理和分析流程。
    """
    if os.path.exists(output_path):
        print("已检测到处理好的数据文件，直接加载...")
        try:
            gdf_fused = gpd.read_parquet(output_path)
            if 'category' not in gdf_fused.columns or 'longitude' not in gdf_fused.columns:
                print("旧的融合数据不包含 'category' 或 'longitude' 列，正在重新生成...")
                os.remove(output_path)
                main()
                return
        except Exception as e:
            print(f"加载旧数据文件失败 ({e})，正在重新生成...")
            os.remove(output_path)
            main()
            return
    else:
        # 1. 加载和准备数据
        df_taxi, gdf_zones, gdf_osm_pois = load_and_prepare_data()
        if df_taxi is None:
            return

        # 2. 执行数据融合
        gdf_fused = perform_data_fusion(df_taxi, gdf_zones, gdf_osm_pois)

        # 3. 将处理好的数据保存到 GeoParquet 文件
        if gdf_fused is not None:
            gdf_fused.to_parquet(output_path)
            print(f"\n数据已成功保存至 {output_path}。")

    # 4. 执行统计分析
    if 'gdf_fused' in locals() and gdf_fused is not None:
        analyze_pickup_poi_types(gdf_fused)

    print("\n代码运行结束。数据已成功融合，可以继续进行更深入的分析和可视化。")


if __name__ == "__main__":
    main()
