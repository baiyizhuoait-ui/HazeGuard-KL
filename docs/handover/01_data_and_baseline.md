# 01 模块交接 — 数据工程、气候基线解耦与基线模型

**负责人**：组长
**状态**：已完成，已并入 `main`
**下一棒**：2号（02 EDA 与时空特征），3号（03）请一并参考第 4 节
**日期**：2026-11-07

---

## 1. 我产出了什么

### 数据表（你们主要要用的）

| 文件 | 形状 | 大小 | SHA256 前 16 位 |
|---|---|---|---|
| `data/haze_panel_clean.parquet` | **966,677 行 × 35 列** | 25.1 MB | `97b5db123352861b` |
| `data/manifest.json` | 清单 + 11 项校验结果 | 6 KB | 每次重跑会变 |
| `data/data_dictionary.json` | 字段口径 | 4 KB | — |

- 时间：**2022-08-04 08:00 → 2026-10-31 20:00**（UTC+8，逐小时）
- 空间：**26 个真实互异的 CAMS 格点**
- 切分：train 676,884 / val 145,054 / test 144,739

**读之前先核对 SHA256**，不然你可能是拿旧表算的，指标对不上别人：

```python
import json, hashlib
m = json.load(open("data/manifest.json", encoding="utf-8"))
assert m["sha256"] == hashlib.sha256(open("data/haze_panel_clean.parquet","rb").read()).hexdigest()
```

### 代码

| 文件 | 作用 |
|---|---|
| `src/config.py` | 26 节点注册表、阈值、代价矩阵、路径 |
| `src/01_data_pipeline.py` | 采集 → 清洗 → 目标 → 切分 → 基线 → QC 标记 → Gate 0 |
| `src/utils_metrics.py` | **六指标唯一实现**，全组 import |
| `src/01_baseline_dt.py` | M1 基线模型 |
| `src/01_verify_data.py` | 完整性校验 + 生成 manifest |
| `tools/discover_nodes.py` | 格点去重（证明为什么是 26 个不是 33 个） |

### 模型与指标

| 文件 | 内容 |
|---|---|
| `models/baseline_dt.joblib` | dict，键 `model`（模型对象）/ `features`（特征名列表） |
| `models/metrics_stage1.json` | 六指标 + 超参 + 分节点结果 + 类别分布 |
| `models/gate0_report.json` | 四项数据可行性闸门 |
| `models/cleaning_report.json` | 清洗计数、QC 计数、逐节点完整性 |

### 基线成绩（test 段，只取受体节点，n = 16,728，其中高危 6,865）

| 指标 | 值 |
|---|---|
| Macro_F1 | **0.6620** |
| Balanced_Accuracy | 0.6443 |
| High_Risk_Recall | **0.5707** |
| MAOE | 0.3220 |
| Severe_Jumping_Error_Rate | 0.0218 |
| Avg_Asymmetric_Cost | 0.8905 |

超参 `gini / max_depth=9 / min_samples_leaf=20`，384 叶，训练-验证 gap 0.015。

---

## 2. 字段说明

### 标识与时空

| 列 | 说明 |
|---|---|
| `station_id` | **CAMS 格点 ID，不是监测站**（见第 4 节第 1 条） |
| `timestamp` | 整小时，UTC+8 |
| `lat` `lon` | 格点代表坐标 |
| `region` | `KL` / `Peninsula` / `Sumatra` / `Borneo` |
| `role` | `receptor` / `corridor` / `source`。**决定谁能当预测目标** |
| `split` | `train` / `val` / `test`，严格时序，不可重切 |

### 观测变量

- 污染物：`pm2_5` `pm10` `no2` `so2` `o3` `co`（µg/m³，CAMS 原样；注意 **CO 不是 ppm**）
- 气象：`temperature_2m`（°C）、`relative_humidity_2m`（%）、
  `wind_speed_ms`（**建模用这个，m/s**）、`wind_speed_kmh`（原始值，只用来核对）、
  `wind_direction_10m`（度，0=北，顺时针）、`precipitation`（mm）

### 质量标记（做敏感性分析时会用到）

| 列 | 语义 | 条数 | 建议怎么处理 |
|---|---|---|---|
| `is_missing` | 仍有变量没插上值 | 277 | 建议丢掉或降权 |
| `qc_interpolated` | 这一行有单元格是 ≤3h 线性插值补的 | 356 | 能用；做敏感性分析时建议剔掉再跑一次 |
| `qc_pm25_gt_pm10` | PM2.5 > PM10，物理上不可能 | 303（0.031%） | 建议丢掉，或 `pm2_5 = min(pm2_5, pm10)` 截断 |
| `qc_censored` | PM2.5 ≥ 895，贴着模式输出上限 | 4 | **不能当真实极值用** |
| `qc_calm_wind` | 风速 < 0.5 m/s，风向是噪声 | 48,621（5.0%） | 上风向权重必须置零 |

### 时间与基线

| 列 | 说明 |
|---|---|
| `month` `hour` `dayofweek` | 由 timestamp 派生 |
| `b_proto_pm25` | 气候态基线，**只在 train 上按 (station, month, hour) 求均值** |
| `b_proto_temp` | 同上，温度版 |
| `b_proto_n` | 该单元格用了多少训练样本（最小 85，中位 90） |
| `b_proto_src` | 现在全是 `station_month_hour`；支撑不足时会退回 `station_month` |
| `pm2_5_anomaly` | `pm2_5 − b_proto_pm25`，去掉季节和日周期背景 |
| `temp_anomaly` | `temperature_2m − b_proto_temp` |

### 目标

| 列 | 说明 |
|---|---|
| `target_pm25_3h` | t+3h 的 PM2.5，`groupby(station).shift(-3)` 构造，**无 NaN** |
| `target_level_3h` | int8，0–3，阈值 (15, 35, 55) |

---

## 3. 关键决策和理由

### 3.1 为什么是 26 个格点，不是 8 个监测站

原来打算按大吉隆坡 8 个监测站做。实际拉数据发现：**8 个坐标只落在 3 个 CAMS 格点上，
同格点内的序列逐位完全相同**——因为格距 45 km 比整个巴生谷（约 40 km）还大。

照原计划做会有两个致命问题：上风向传导特征里的"邻居"其实就是它自己；
留一站点验证（LOSO）留掉一个站，同格还有 2–3 个兄弟带着一模一样的标签。

所以撒了 33 个候选点，按序列是否完全相同去重，得到 26 个真互异的格点，
并往西扩到苏门答腊、往南扩到加里曼丹（跨境烟霾的物理源头）。证据在 `docs/node_discovery.json`。

### 3.2 为什么时间范围是 51 个月

CAMS 全球档案从 2022-08-01 开始（首月前 80 小时是空的，整段剔掉，没插值）。
既然能用就用到边界——多出来的 27 个月把 2024、2025 两个完整季风期装进训练集，
直接缓解了测试期的分布偏移。

### 3.3 为什么基线只在 train 上算

用全量数据算 `b_proto`，验证期和测试期的未来信息会通过分组均值漏进训练特征。
`01_verify_data.py` 的第 7 项检查会用 train 重算一遍再逐行比对，
现在 966,677 行全部可复现，说明没漏。

### 3.4 为什么按时间跨度切，不按行数切

按行数切会在缺失处让边界漂移。按 timestamp 的 70/15/15 分位切，
边界稳定在 2025-07-24 02:00 和 2026-03-13 13:00，26 个节点完全一致。

### 3.5 为什么 QC 异常只标记不修改

303 行 PM2.5 > PM10 如果直接截断，等于把高值压成中值，会系统性低估烟霾强度。
所以我们只打标记，把处理权交给用的人，并且要求他在报告里说明怎么处理的——
而不是偷偷改掉当没发生过。

---

## 4. 已知缺陷和坑（**一定要看**）

### 4.1 `station_id` 不是监测站

它是 0.4°（约 45 km）的 CAMS 格点。**图表和报告里不要写** "8 个监测站"、
"ground station"、"measured"。正确说法是 *grid-cell estimates from the CAMS global
composition product*。这句话写错，答辩时会被直接问住。

### 4.2 数据整体偏高

`KL_City` 两年均值约 41 µg/m³，明显高于马来西亚官方常年水平。跨数据源比较时必须说明。
也别把"Level 3 占 21.6%"直接翻译成"一年有五分之一时间空气危险"。

### 4.3 测试集里有真实烟霾事件

2026 年 9、10 月，`KL_City` 单月 Level 3 占比 46.9% / 51.0%，PM2.5 的 p95 是 151.8。
受体整体 Level 3：train 11.13% vs test 18.85%，差 1.69 倍。

**含义**：test 比 train 难。任何"测试不如训练"的落差不能全甩给过拟合。
做对照实验时必须用同一个 test 段，不要临时改切分。

### 4.4 源区节点的 Level 3 比例几乎是 0

`Banda_Aceh` 0.0%、`Bengkulu` 0.0%、`Kota_Bharu` 0.1%、`Kuala_Terengganu` 0.1%。

**含义**：把 26 个节点混在一起算 Macro-F1，会被这些"永远低浓度"的节点拉高，
看起来漂亮但没意义。**主指标必须在受体节点上报**（`role == "receptor"`）。

### 4.5 `Palangkaraya` 有真实缺口

37 小时（2026-08-30 前后）+ 若干短缺口，一共 44 处断点，最长步长 19 小时。
这些行因为没有当前时刻的 PM2.5 已经被剔掉了，所以这个节点是 36,952 行（其他节点 37,189）。

**不要插值补上。** 补 37 小时等于编造一次烟霾过程。

### 4.6 静风时段的风向不能信

`wind_speed_ms < 0.5` 时风向是数值噪声。算上风向权重时 `max(0, v·e)` 在 v≈0 时虽然会自动趋近 0，
但如果先把风向编码成 sin/cos，噪声就会被当成真实方向。
**建议**：`weight = weight * (1 - qc_calm_wind)`。

### 4.7 我之前写错了一个假设，这里更正

一开始以为 CAMS 是 3 小时原生、Open-Meteo 做了线性插值，所以 1 小时滞后没意义。
**实测不成立**：二阶差分接近 0 的比例只有 0.6%，不存在 3 小时分段线性的结构，
序列本来就是逐小时输出的。

但平滑度确实偏高（模式产品的通病），所以你在做滞后特征时会发现 lag1 极其重要——
这是数据特性，**不是泄露**，别看到 lag1 权重高就以为写错了。

### 4.8 `qc_pm25_gt_pm10` 集中在那几个节点

做敏感性分析如果要剔掉这 303 行，**一定要同时给出剔除前后的指标对比**，
否则会被质疑"选择性清洗"。

### 4.9 列名是接口，不是建议

`pm2_5_anomaly` 不是 `pm25_anomaly`，`target_level_3h` 不是 `y`。
最后一棒合成报告时是按列名批量取数的，拼错不报错，只会少一列。

---

## 5. 下一棒怎么用

### 最小可运行片段（02 的起点）

```python
import pandas as pd

df = pd.read_parquet("data/haze_panel_clean.parquet")
df["split"] = df["split"].astype(str)      # category 转 str，比较方便

train = df[df["split"] == "train"]          # 所有统计量只能从这里拟合

# 做时序特征必须先分组，否则站点 A 的末尾会接到站点 B 的开头
df = df.sort_values(["station_id", "timestamp"])
df["pm25_lag1"] = df.groupby("station_id")["pm2_5"].shift(1)

import sys; sys.path.insert(0, "src")
from utils_metrics import evaluate_predictions, print_metrics   # 指标统一走这里
```

### 受体 vs 全 panel

```python
receptor = df[df["role"] == "receptor"]    # 主指标在这里报
panel    = df                              # 训练可以用全量，评估别用
```

训练用全 panel、评估用受体，是目前推荐的配置。但**报告里要写清楚**，
不然读者会以为模型只在 KL 上训过。

### 校验

```bash
python src/01_verify_data.py --check     # 应该 11/11 通过
```

每次产出新数据表之后都跑一次，确认没把上游的东西弄坏。

---

## 6. 我没做的事

| 没做 | 归谁 | 备注 |
|---|---|---|
| 滞后、滚动、周期编码特征 | 2号 | 我只给了当前时刻的观测 |
| 上风向传导 / 风矢量分解 | 2号 | 但 `wind_speed_ms` 和静风标记已经准备好了 |
| 模糊信息粒化（FIG） | 3号 | — |
| 特征选择与消融 | 3号 | — |
| M2–M6 五个模型 | 4 / 5 / 6 号 | 我只做了 M1 基线 |
| 分位数区间、PICP、MPIW | 5号 | — |
| 代价敏感优化、SHAP、泛化实验 | 6号 | 数据已经支持（26 个节点 + 51 个月） |
| Streamlit 界面 | 7号 | — |

另外提醒一句：`README.md` 和 `CONTRIBUTING.md` 里写的模块划分是最终目标，
我这一棒只完成了 01。别以为 02 之后的东西已经有了。

---

## 附：我踩过的三个坑，你们可以直接跳过

1. **`shift()` 忘了 groupby** —— 会造出跨城市的伪标签，程序不报错，指标莫名其妙变好。
2. **风速单位** —— Open-Meteo 默认 km/h，直接拿去算上风向权重会整体放大 3.6 倍，同样不报错。
3. **基线用了全量数据** —— 最常见的泄露，指标会虚高，而且很难发现。
   我的校验脚本第 7 项就是专门查这个的。
