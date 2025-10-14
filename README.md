# PABDS-Assignment
南京大学23级信息管理与信息系统本科生 大数据系统原理与应用核心课程期初作业  
231820016-邱怡玮 231820288-张亦萌

## 项目简介
本项目旨在通过对纽约市出租车数据的分析，结合天气、地理位置等多维度数据，提供全面的可视化和预测功能。项目分为多个模块，涵盖数据预处理、综合分析、仪表盘展示和查询接口等。

## 项目结构

### 1. 数据预处理模块 (`pre-processing`)
- **功能**：
  - 数据清洗与异常值处理。
  - 描述性统计分析。
  - 数据可视化。
- **主要文件**：
  - `basic_visualization.py`：生成时间动态图和机场服务分析。
  - `descriptive_analysis.py`：执行描述性统计分析。

### 2. 综合分析模块 (`comprehensive_analysis`)
- **功能**：
  - 深入分析不同出租车服务的特性。
  - 异常值检测与处理。
- **主要文件**：
  - `green_taxi_driver.py`：预测高小费&总收入的POI类型并返回最近坐标（面向司机）。
  - `green_taxi_passenger.py`：预测订单的时间及价格（面向乘客）。
  - `outliers_analysis.py`：异常值分析。

### 3. 气候分析模块 (`climate`)
- **功能**：
  - 分析天气对出行需求的影响，并进行差异性检验。
  - 计算效率与成本。
- **主要文件**：
  - `weather_temp_visibility.py`：分析订单数与能见度的关系。
  - `efficiency_cost.py`：计算天气对服务效率的影响。
  - `peakhours_week.py`：分析高峰时段的天气影响。

### 4. 数据合并模块 (`merge`)
- **功能**：
  - 合并不同来源的数据。
  - 生成统一的数据集。
- **主要文件**：
  - `merging_weather.py`：合并天气数据。

### 5. 地理分析模块 (`geography`)
- **功能**：
  - 分析上下车点的POI类型分布。
  - 生成热力图。
- **主要文件**：
  - `Green_Yellow_Average_Hourly_Traffic_Heatmap.py`：生成平均流量热力图。

### 6. 仪表盘模块 (`dashboard`)
- **功能**：
  - 提供交互式可视化界面。
  - 展示多维度分析结果。
- **主要文件**：
  - `app.py`：仪表盘主程序。
  - `templates/`：HTML模板文件。

### 7. 查询接口模块 (`query_interface`)
- **功能**：
  - 提供基于用户输入的查询功能。
  - 返回预测结果和推荐。
- **主要文件**：
  - `app.py`：查询接口主程序。

## 使用方法

### 环境配置
1. 确保已安装Python 3.8及以上版本。
2. 安装依赖包：
   ```bash
   pip install -r requirements.txt
   ```

### 数据准备
1. 将原始数据放置在`原始数据/`文件夹中。
2. 运行数据预处理脚本：
   ```bash
   python pre-processing/basic_visualization.py
   ```


### 查询接口
1. 进入`query_interface`目录。
2. 启动查询服务：
   ```bash
   python app.py
   ```
3. 在浏览器中访问`http://127.0.0.1:5000`。


