#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HazeGuard-KL 全局配置与站点注册表

设计原则
--------
1. 单一数据源：只用 Open-Meteo 的 CAMS 空气质量 + ERA5 气象，不引入第二份数据集。
2. 空间单元是 CAMS 格点，不是行政站点。0.4° 约 45 km，因此候选点位若落在同一格点，
   序列会逐位相同——本项目按序列相等去重，只保留 26 个真实互异的空间单元
   （见 docs/node_discovery.json 的发现过程）。
3. 受体（receptor）与源区（source）分离：受体是预测目标，源区只作上风向特征输入。
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 时间范围：CAMS 全球档案自 2022-08-01 起（首月覆盖率 89%，其后 100%）
# ---------------------------------------------------------------------------
START_DATE = "2022-08-01"
END_DATE = "2026-10-31"
TIMEZONE = "Asia/Kuala_Lumpur"

# ---------------------------------------------------------------------------
# 预测目标
# ---------------------------------------------------------------------------
HORIZON_H = 3                                   # 提前 3 小时
LEVEL_THRESHOLDS = (15.0, 35.0, 55.0)           # µg/m³，四级有序
LEVEL_NAMES = {0: "Low", 1: "Moderate", 2: "High", 3: "Very High"}

# ---------------------------------------------------------------------------
# 时序切分（逐节点内部按时间顺序）
# ---------------------------------------------------------------------------
SPLIT_RATIOS = (0.70, 0.15, 0.15)               # train / val / test

# ---------------------------------------------------------------------------
# 数据质量规则
# ---------------------------------------------------------------------------
MAX_INTERP_GAP_H = 3                            # 连续缺失 <=3h 线性插值
BOUNDS = {                                      # 物理越界 → NaN
    "pm2_5": (0.0, 900.0),
    "pm10": (0.0, 1200.0),
    "no2": (0.0, 500.0),
    "so2": (0.0, 500.0),
    "o3": (0.0, 500.0),
    "co": (0.0, 20000.0),
    "temperature_2m": (-20.0, 50.0),
    "relative_humidity_2m": (0.0, 100.0),
    "wind_speed_ms": (0.0, 60.0),
    "precipitation": (0.0, 300.0),
}

# --- 质量控制阈值（v2.1 新增）---------------------------------------------
# 模式封顶：CAMS 对 PM2.5 的输出上限约 900 µg/m³，贴近上限的值是被截断的，
# 不是真实观测，必须单独标记而不是当极值使用。
CENSORED_PM25 = 895.0
# 静风阈值：风速低于此值时风向是数值噪声，上风向权重不可信
CALM_WIND_MS = 0.5
# 基线支撑度下限：低于此值的 (station,month,hour) 单元格回退到 (station,month)
MIN_BASELINE_SUPPORT = 30

# ---------------------------------------------------------------------------
# 空气质量字段映射（Open-Meteo 原字段 → 项目标准字段）
# ---------------------------------------------------------------------------
AQ_VARS = {
    "pm2_5": "pm2_5",
    "pm10": "pm10",
    "nitrogen_dioxide": "no2",
    "sulphur_dioxide": "so2",
    "ozone": "o3",
    "carbon_monoxide": "co",       # 注意：该接口 CO 单位为 µg/m³，非 ppm
}
WEATHER_VARS = {
    "temperature_2m": "temperature_2m",
    "relative_humidity_2m": "relative_humidity_2m",
    "wind_speed_10m": "wind_speed_kmh",   # 原单位 km/h，另派生 m/s
    "wind_direction_10m": "wind_direction_10m",
    "precipitation": "precipitation",
}

# ---------------------------------------------------------------------------
# 26 个真实互异的 CAMS 空间单元
# role: receptor = 预测目标站点；source = 仅作上风向特征；corridor = 两者兼有
# 同格点归并关系见 members（保留代表点，不重复建站）
# ---------------------------------------------------------------------------
NODES = [
    # ---- 大吉隆坡（受体，3 个真实格点）----
    dict(id="KL_City",       lat=3.107, lon=101.726, region="KL",        role="receptor",
         members=["KL_City", "Batu_Muda", "Petaling_Jaya", "Shah_Alam"]),
    dict(id="Port_Klang",    lat=3.000, lon=101.392, region="KL",        role="receptor",
         members=["Port_Klang"]),
    dict(id="Sepang",        lat=2.828, lon=101.703, region="KL",        role="receptor",
         members=["Sepang", "Putrajaya", "Banting"]),
    # ---- 马来半岛走廊（既作目标也作源）----
    dict(id="Seremban",      lat=2.729, lon=101.938, region="Peninsula", role="corridor", members=["Seremban"]),
    dict(id="Malacca",       lat=2.189, lon=102.250, region="Peninsula", role="corridor", members=["Malacca", "Muar"]),
    dict(id="Johor_Bahru",   lat=1.492, lon=103.741, region="Peninsula", role="corridor", members=["Johor_Bahru"]),
    dict(id="Kuantan",       lat=3.807, lon=103.326, region="Peninsula", role="corridor", members=["Kuantan"]),
    dict(id="Temerloh",      lat=3.450, lon=102.417, region="Peninsula", role="corridor", members=["Temerloh"]),
    dict(id="Kota_Bharu",    lat=6.125, lon=102.238, region="Peninsula", role="corridor", members=["Kota_Bharu"]),
    dict(id="Kuala_Terengganu", lat=5.330, lon=103.140, region="Peninsula", role="corridor", members=["Kuala_Terengganu"]),
    dict(id="Penang",        lat=5.414, lon=100.329, region="Peninsula", role="corridor", members=["Penang"]),
    dict(id="Alor_Setar",    lat=6.118, lon=100.368, region="Peninsula", role="corridor", members=["Alor_Setar"]),
    dict(id="Ipoh",          lat=4.597, lon=101.090, region="Peninsula", role="corridor",
         members=["Ipoh", "Cameron_Highland"]),
    # ---- 苏门答腊火点区（上风向源）----
    dict(id="Banda_Aceh",    lat=5.548, lon=95.324,  region="Sumatra",   role="source", members=["Banda_Aceh"]),
    dict(id="Medan",         lat=3.595, lon=98.672,  region="Sumatra",   role="source", members=["Medan"]),
    dict(id="Dumai",         lat=1.666, lon=101.450, region="Sumatra",   role="source", members=["Dumai"]),
    dict(id="Pekanbaru",     lat=0.507, lon=101.447, region="Sumatra",   role="source", members=["Pekanbaru"]),
    dict(id="Padang",        lat=-0.949, lon=100.354, region="Sumatra",  role="source", members=["Padang"]),
    dict(id="Jambi",         lat=-1.610, lon=103.613, region="Sumatra",  role="source", members=["Jambi"]),
    dict(id="Palembang",     lat=-2.976, lon=104.775, region="Sumatra",  role="source", members=["Palembang"]),
    dict(id="Bengkulu",      lat=-3.800, lon=102.265, region="Sumatra",  role="source", members=["Bengkulu"]),
    # ---- 加里曼丹（西段烟霾源）----
    dict(id="Pontianak",     lat=-0.026, lon=109.342, region="Borneo",   role="source", members=["Pontianak"]),
    dict(id="Palangkaraya",  lat=-2.209, lon=113.917, region="Borneo",   role="source", members=["Palangkaraya"]),
    dict(id="Banjarmasin",   lat=-3.319, lon=114.590, region="Borneo",   role="source", members=["Banjarmasin"]),
    dict(id="Kuching",       lat=1.553, lon=110.359, region="Borneo",    role="source", members=["Kuching"]),
    dict(id="Bintulu",       lat=3.171, lon=113.041, region="Borneo",    role="source", members=["Bintulu"]),
]

NODE_IDS = [n["id"] for n in NODES]
RECEPTORS = [n["id"] for n in NODES if n["role"] == "receptor"]
COORDS = {n["id"]: (n["lat"], n["lon"]) for n in NODES}

# ---------------------------------------------------------------------------
# 非对称代价矩阵  C[i][j] = 真实为 i、预测为 j 的代价
# 左下三角（漏报高危）为普通误报的 8–15 倍
# ---------------------------------------------------------------------------
DEFAULT_COST_MATRIX = [
    [0, 1, 3, 5],
    [2, 0, 1, 3],
    [8, 4, 0, 1],
    [15, 10, 5, 0],
]

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
DATA_DIR = "data"
RAW_DIR = "data/raw"
MODEL_DIR = "models"
FIGURE_DIR = "figures"
REPORT_DIR = "reports"

RANDOM_STATE = 42
