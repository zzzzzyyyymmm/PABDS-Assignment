import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
import plotly.express as px
import os

# --- 定义文件路径 ---
fused_data_path = 'fused_pickup_fhv.parquet'
output_dir = 'visualizations'

# 确保输出文件夹存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def create_interactive_geographic_map(df):
    """
    创建具有单选分层功能的地理分布图。
    """
    print("生成具有单选分层功能的地理分布图...")

    if 'longitude' not in df.columns or 'latitude' not in df.columns:
        print("错误: 融合数据中缺少 'longitude' 或 'latitude' 列，无法生成地图。")
        return

    # 初始化 Plotly 图形对象
    fig = go.Figure()

    # 获取所有独特的 POI 类别
    categories = sorted(df['category'].unique().tolist())

    # 使用 Plotly 的 Vivid 配色方案
    colors = px.colors.qualitative.Vivid

    # 为每个 POI 类别创建独立的图层（trace）
    for i, category in enumerate(categories):
        df_category = df[df['category'] == category].copy()

        fig.add_trace(go.Scattermapbox(
            lat=df_category['latitude'],
            lon=df_category['longitude'],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=8,
                color=colors[i % len(colors)]
            ),
            hovertext=df_category['name'],
            hoverinfo='text+lat+lon',
            name=category,
            customdata=df_category[['fclass', 'category']],
            hovertemplate='<b>名称:</b> %{hovertext}<br><b>经度:</b> %{lon}<br><b>纬度:</b> %{lat}<br><b>详细类别:</b> %{customdata[0]}<br><b>大类:</b> %{customdata[1]}<extra></extra>'
        ))

    # 创建下拉菜单按钮
    buttons = []
    # 添加一个 "全部显示" 按钮
    buttons.append(dict(
        label="全部显示",
        method="update",
        args=[{"visible": [True] * len(categories)}],
    ))

    # 为每个类别创建单独的显示按钮
    for i, category in enumerate(categories):
        visibility = [False] * len(categories)
        visibility[i] = True
        buttons.append(dict(
            label=category,
            method="update",
            args=[{"visible": visibility}],
        ))

    # 地图布局，添加下拉菜单
    fig.update_layout(
        mapbox_style="carto-positron",
        title='POI 类别地理分布与出租车上车点关系图',
        mapbox_zoom=10,
        mapbox_center={"lat": 40.7, "lon": -74.0},
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        showlegend=False,
        updatemenus=[go.layout.Updatemenu(
            type="dropdown",
            direction="down",
            showactive=True,
            active=0,
            x=0.0,
            xanchor="left",
            y=1.1,
            yanchor="top",
            buttons=buttons,
        )]
    )

    output_path = os.path.join(output_dir, 'poi_pickup_map_fhv.html')
    fig.write_html(output_path)
    print(f"交互式地理分布图已生成！文件位置: {output_path}")
    fig.show()


def main():
    """
    主函数，执行可视化流程。
    """
    if not os.path.exists(fused_data_path):
        print(f"错误：找不到 {fused_data_path} 文件。请确保你已成功运行过数据融合代码。")
        return

    try:
        # 加载已经处理好的数据
        df_fused = gpd.read_parquet(fused_data_path)
        print("成功加载融合后的数据。")

        # 执行可化函数
        create_interactive_geographic_map(df_fused)

    except Exception as e:
        print(f"可视化过程中发生错误：{e}")


if __name__ == "__main__":
    main()
