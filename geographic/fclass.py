import pandas as pd
from shapely.geometry import Point
import numpy as np
import geopandas as gpd
import os

# 定义文件路径
osm_map_path = 'new-york-250907-free/gis_osm_pois_free_1.shp'
output_file_path = 'all_fclass_types.txt'


def extract_and_save_fclass():
    """
    加载 POI 数据，提取所有独特的 'fclass' 类型，并保存到文件中。
    """
    print("--- 开始提取所有独特的 POI 类型 ---")

    if not os.path.exists(osm_map_path):
        print(f"错误: 找不到文件 - {osm_map_path}")
        return

    try:
        # 1. 加载 OpenStreetMap POI 数据
        print("正在加载 OpenStreetMap POI 数据...")
        gdf_osm_pois = gpd.read_file(osm_map_path)

        # 2. 提取所有独特的 fclass 类型
        unique_fclass = gdf_osm_pois['fclass'].unique()

        # 3. 将结果排序并保存到文件中
        unique_fclass_sorted = sorted([f for f in unique_fclass if pd.notna(f)])

        with open(output_file_path, 'w', encoding='utf-8') as f:
            for item in unique_fclass_sorted:
                f.write(f"{item}\n")

        print(f"\n✅ 提取完成！总共找到 {len(unique_fclass_sorted)} 种独特的 POI 类型。")
        print(f"✅ 所有类型已成功保存至文件: {output_file_path}")

    except Exception as e:
        print(f"在处理过程中发生错误: {e}")


if __name__ == "__main__":
    extract_and_save_fclass()
