# 03 模块交接 多尺度趋势粒化与特征选择

| | |
|---|---|
| 负责人 | 3 号 |
| 任务依据 | 仓库现行规则：`CONTRIBUTING.md`（3 号职责与产物）、`docs/syllabus_coverage.md` §3（只做区间粒化与趋势粒化；互信息 + RFE + permutation importance）、2 号交接文档（白名单、折内拟合）、1 号 11-13 的修补要求（周期编码成对、邻域对照） |
| 上游输入 | `data/feat_spatiotemporal.parquet`（2 号，4 个有序分片），清单摘要 `208cf00942b404aead0ffe913497d34157ec01f3e1a4ccf593b10f71fb61a55e` |
| 下一棒 | 4 号、5 号（读 `features_final.parquet` 和 `selected_features.json`） |
| 分支 | `feat/03-granulation`，已同步 `main`（02 已于 `8e7c847` 并入）；合并顺序 02 → 03 → 04 |
| 版本 | v2（2026-11-13，对应质量通告 QN-002 的 Q3-1~Q3-3）：sin/cos 成对重选 Top-25，**比 v1 换了一列**（加 `hour_cos`，去 `gran_low_12h`）；消融增加 `hour_cos` 与邻域 50 列原始值两个因素的对照 |

---

## 1. 我产出了什么

### 数据表

| 文件 | 形状 | 大小 | SHA256 |
|---|---|---|---|
| `data/features_final.parquet` | **111,056 行 × 32 列** | 4.7 MB | `cadbe163b43917c421070298f1de9f978358274989362e80a258043fd360c78a` |

行 = 受体格点（`role == "receptor"`）且 `audit_model_eligible` 为真的行，三段行数与 2 号审计 JSON 的 `eligibility_by_receptor_split` 一致，也与 1 号在 `PANEL_CONTRACT` 登记的可建模行数一致：

| `split` | 行数 |
|---|---:|
| `train` | 77,753 |
| `val` | 16,656 |
| `test` | 16,647 |

中间表 `data/feat_granulation.parquet`（全面板 966,677 行 × 12 列，14.7 MB）**不入库**，`.gitignore` 已登记。它由粒化脚本 1 秒内生成，溯源信息写在文件的 parquet 元数据里。

### 模型文件与图

| 文件 | 内容 |
|---|---|
| `models/selected_features.json` | 交付的成对 Top-25、不成对的旧版 Top-25、三种策略各自的排名、并集、成对规则、RFE 排名、行筛选与填补规则、粒化溯源 |
| `models/metrics_stage3.json` | 十二组消融 × 3 个种子在 val 上的六项指标，均值 ± 标准差；M0 持续性基线；邻域特征的定义、每个受体的邻居排序、缺测统计和排列方式检查 |
| `figures/fig_granulation_bands.png` | 粒化行为：KL_City 在 train 期的一次烟霾脉冲前后各 5 天，12h、6h 粒化区间与中枢，以及两种窗口的斜率，300 dpi |
| `figures/fig_feature_ablation.png` | 十二组消融结果：Macro-F1、高危召回、平均非对称代价三联图，带 M0 参考线，300 dpi |
| `reports/sec2_4_2_and_2_5.md` | 报告 §2.4.2 与 §2.5 |

### 代码

| 文件 | 作用 | 耗时 |
|---|---|---|
| `src/03_granulation_features.py` | 区间粒化与趋势粒化，画 `fig_granulation_bands.png` | 约 1 秒 |
| `src/03_feature_selection.py` | 三种选择策略 → 并集 → 成对 RFE → Top-25；邻域均值；消融；最终表 | 约 3 分钟 |

两份代码在仓库 `src/` 里直接运行，输入和输出都是仓库根目录。所有 JSON 都带 `generated_at`、`git_commit`、`code_sha256`、`input_sha256`。同一输入重跑，结果逐字节相同。

---

## 2. 字段/接口说明

### 2.1 `features_final.parquet`

**行顺序是 `station_id, timestamp`（沿用面板落盘顺序），不是时间序。** 做 `TimeSeriesSplit` 之前必须先按 `timestamp` 排序（协作规范 §3 第 8 条）。

| 列 | 类型 | 说明 |
|---|---|---|
| `station_id` | str | CAMS 格点 ID，不是监测站 |
| `timestamp` | `datetime64[us]` | 吉隆坡本地时间，无时区 |
| `region`、`role` | str | `role` 全是 `receptor` |
| `split` | category | `train` / `val` / `test`，沿用 1 号的划分 |
| `target_pm25_3h` | float64 | t+3h PM2.5，µg/m³，无缺失 |
| `target_level_3h` | **int8** | 0–3，阈值 15 / 35 / 55，无缺失（int8 是契约类型，见第 4 节第 5 条） |
| 25 个特征 | float | 顺序与 `selected_features.json → selected_features` 相同，**全部为有限值，无 NaN、无 inf** |

Top-25（按最终 RF 重要性从高到低；「策略」列表示被 a 互信息 / b 方差+相关性过滤 / c 置换重要性 中的哪几种选中）：

| # | 特征 | 来源 | 单位 | 策略 |
|---:|---|---|---|---|
| 1 | `pm2_5` | 1 号原始 | µg/m³ | abc |
| 2 | `pm10` | 1 号原始 | µg/m³ | ac |
| 3 | `pm2_5_mean_3h` | 2 号滚动 | µg/m³ | ac |
| 4 | `hour_sin` | 2 号周期（与 `hour_cos` 成对） | 无量纲 | bc |
| 5 | `pm2_5_lag_1` | 2 号滞后 | µg/m³ | ac |
| 6 | `no2` | 1 号原始 | µg/m³ | abc |
| 7 | `pm10_lag_1` | 2 号滞后 | µg/m³ | abc |
| 8 | `pm2_5_change_3h` | 2 号变化率 | µg/m³ | abc |
| 9 | `gran_up_6h` | 3 号区间粒化 | µg/m³ | ac |
| 10 | `so2` | 1 号原始 | µg/m³ | ab |
| 11 | `spatial_pm25_gradient` | 2 号近场空间（**缺失已填补**） | µg/m³ | ab |
| 12 | `no2_lag_12` | 2 号滞后 | µg/m³ | b |
| 13 | `pm2_5_slope_6h` | 2 号趋势（= 6h 趋势粒化） | µg/m³/h | abc |
| 14 | `gran_low_6h` | 3 号区间粒化 | µg/m³ | ac |
| 15 | `gran_slope_12h` | 3 号趋势粒化 | µg/m³/h | ab |
| 16 | `pm2_5_mean_6h` | 2 号滚动（= 6h 粒化中枢） | µg/m³ | abc |
| 17 | `no2_lag_6` | 2 号滞后 | µg/m³ | b |
| 18 | `relative_humidity_2m_lag_3` | 2 号滞后 | % | bc |
| 19 | `o3` | 1 号原始 | µg/m³ | bc |
| 20 | `temperature_2m` | 1 号原始 | °C | b |
| 21 | `co` | 1 号原始 | µg/m³ | ab |
| 22 | `no2_lag_24` | 2 号滞后 | µg/m³ | ab |
| 23 | `far_source_pm25_mean_24h` | 2 号远场背景 | µg/m³ | b |
| 24 | `no2_lag_2` | 2 号滞后 | µg/m³ | b |
| 25 | `hour_cos` | 2 号周期（与 `hour_sin` 成对） | 无量纲 | bc |

按来源统计：1 号原始 7 列、2 号时间特征 13 列、2 号空间 2 列、3 号粒化 3 列。与 v1 相比只换了一列：加 `hour_cos`，去 `gran_low_12h`。

**空间列缺失填补**（只填上游为 NaN 的格，其余值与上游逐值相等；不拟合任何参数）：

| 列 | 缺失含义 | 填补值 | 填补行数 train / val / test |
|---|---|---|---|
| `upwind_pm25_advection_1h` / `_2h` / `_3h` | 该时刻没有有效的近场上风向邻居 | 本格点 `pm2_5_lag_1` / `_2` / `_3`（邻域等于本格点） | 约 7,585 / 1,623 / 753（每列差 1–2 行） |
| `spatial_pm25_gradient` | 同上 | 0 | 7,587 / 1,623 / 753 |
| `far_source_pm25_mean_48h` | 分段开头不足 48h 历史 | 同刻 `far_source_pm25_mean_24h`（最接近的更短窗口） | 70 / 72 / 72 |

近场空间列在合格受体 train 行里缺 9.76%。五列都在选择和消融之前填补，最终只有 `spatial_pm25_gradient` 进了 Top-25。没有把缺失当作零浓度（2 号交接 §2）。

### 2.2 `selected_features.json`

下游只需要 `selected_features`（有序列表，25 个），按这个键名读即可。其余键：

| 键 | 内容 |
|---|---|
| `id_columns`、`target_columns` | 最终表里特征以外的列 |
| `row_filter`、`rows_by_split` | 行筛选条件与三段行数 |
| `spatial_fill` | 上面的填补规则与逐段行数 |
| `granulation` | 粒化脚本的溯源：生成时间、上游摘要、代码哈希、缺失统计、作图时段 |
| `candidates`（68）、`exact_duplicates_removed`（5） | 候选清单；与 2 号列完全相同而被去掉的粒化列 |
| `strategy_a_mutual_info`、`strategy_b_variance_spearman`、`strategy_c_permutation` | 三种策略的完整排名和各自选出的集合 |
| `union`（64）、`pairing` | 并集；成对规则、成对后的并集（本次与并集相同，六个周期列本来都在） |
| `rfe_ranking`、`final_rf_importance`、`selected_in_strategies` | 成对 RFE 的淘汰顺序（1 = 入选，数字越大越早淘汰）、最终重要性、每个入选特征来自哪几种策略 |
| `selected_features_unpaired`、`rfe_ranking_unpaired` | v1 的不成对 Top-25 及其 RFE 排名，只作消融对照 |

### 2.3 粒化特征定义（10 列，float32）

窗口 x = [x(t−W+1), …, x(t)]，即含当前时刻的 PM2.5 尾随窗口，W ∈ {6, 12}。

| 列 | 定义 | 单位 |
|---|---|---|
| `gran_low_{W}h`、`gran_up_{W}h` | 窗口最小值、最大值（区间粒的下界、上界） | µg/m³ |
| `gran_core_{W}h` | 窗口均值（中枢） | µg/m³ |
| `gran_width_{W}h` | 上界 − 下界 | µg/m³ |
| `gran_slope_{W}h` | 窗口内最小二乘斜率（趋势粒） | µg/m³/h |

窗口里任一小时缺失（或被标为插值/缺失），该行全部粒化特征为 NaN：全表 6h 约 0.155% 的行、12h 约 0.27% 的行；**所有合格行（26 个节点）都没有 NaN**。

### 2.4 邻域特征（QN-002 Q3-2/Q3-3；只用于消融，不进候选、不在最终表里）

**主实现：邻域 50 列原始值**（列名 `nbr01_pm2_5`、`nbr01_change_3h` … `nbr25_pm2_5`、`nbr25_change_3h`）

| 项 | 定义 |
|---|---|
| 取哪些格点 | 每个受体取**另外 25 个格点**（全部 26 个格点里**不含自身**），按大圆距离从近到远排成 `nbr01` … `nbr25` |
| 取什么值 | 每个格点两个原始值：t 时刻的 PM2.5；该格点的 `pm2_5_change_3h`（2 号的列，x(t) − x(t−3)）。**不做平均，不用上风向权重**，逐列进模型 |
| 缺测怎么办 | 被标为插值（`qc_interpolated`）或缺失（`is_missing`）的小时记为 NaN（与 2 号的屏蔽规则一致）。**不丢列、不按时间插值（插值会用到未来端点）、不填 0、也不用自身滞后替代**；保留 NaN，交给随机森林原生的缺失值处理（每次分裂时为缺失样本选择更优的一侧）。这样不会像上风向特征那样被填补成自回归项 |
| 实际排序 | KL_City：Sepang 31.1 km、Port_Klang 38.9 km、Seremban 48.2 km …… 最远 Banjarmasin 1,598.3 km；Port_Klang：KL_City 38.9 km、Sepang 39.5 km、Seremban 67.7 km …… 最远 Banjarmasin 1,626.4 km；Sepang：Seremban 28.3 km、KL_City 31.1 km、Port_Klang 39.5 km …… 最远 Banjarmasin 1,587.0 km（全表见 JSON） |
| 缺测统计（受体合格行） | 50 列的缺失比例：train 0.0105%（241 行至少缺一个值）、val 0、test 0.54%（2,349 行至少缺一个值） |

**排列方式检查**：同样的原始值改成"每个格点固定一列"（26 个格点 × 2 = 52 列，受体自身那两列为 NaN），加到 v1 Top-25 上再跑一次，用来确认结果不是按距离排列造成的。结果在 `neighbour_feature.layout_check`，不画进图。

**对照（压缩版）：最近 3 格点均值** `nbr_pm25_mean_k3`：`nbr01`~`nbr03` 三个 PM2.5 的简单平均；对可用值取平均，全缺时用本格点当时的 PM2.5（实际没有全缺的行）。按 QN-002，这只是对照，不是主实现，在消融里是第 ⑦、⑧ 组。

出处：`metrics_stage3.json → neighbour_feature`（`main`、`layout_check`、`control`、`neighbours_by_receptor`、`row_counts_eligible_receptor`）。

### 2.5 `metrics_stage3.json`

`ablation.<组名>` 下有 `features`、`runs`（每个种子的完整指标字典）、`mean`、`std`；`metrics_persistence_M0_val` 是同一批 val 行上的 M0；`factor_rows` 记录对照用的旧版 Top-25 和补上的周期项；`neighbour_feature` 记录邻域特征的定义、排序、缺测和排列检查结果。指标全部由 `src/utils_metrics.py` 的 `evaluate_predictions` 产出，`High_Risk_Recall` 是 union 口径。

---

## 3. 关键决策和理由

**只做区间粒化与趋势粒化，不做 FCM。** 依据 `docs/syllabus_coverage.md` §3：模糊 C 均值、隶属度和解粒化重构属于课程外内容，且与滚动统计高度重叠。

**候选只用 2 号白名单。** `model_feature_columns` 是唯一预测白名单，排除了上游气候态和异常量。候选 = 白名单 63 列 + 去重后的 5 列粒化特征 = 68 列。

**折内拟合。** 粒化特征是窗口统计量，填补规则和邻域特征也不拟合参数，RF 不需要标准化。因此特征选择是唯一用到标签的步骤，只在受体合格 train 行上拟合，val 和 test 没参与。

**窗口规则照搬 02。** 窗口按 `station_id × split` 在完整小时网格上取；`qc_interpolated` 或 `is_missing` 的行置空；窗口不完整就记 NaN。与 2 号同定义的 5 列逐值相等，NaN 位置也相同。

**和 02 完全重复的粒化列不进候选。**
- `gran_core_6h` / `gran_core_12h` = `pm2_5_mean_6h` / `pm2_5_mean_12h`
- `gran_width_6h` / `gran_width_12h` = `pm2_5_range_6h` / `pm2_5_range_12h`
- `gran_slope_6h` = `pm2_5_slope_6h`

**特征选择**（受体合格 train 行，按 `timestamp, station_id` 排序）：
- a) `mutual_info_classif`，标签用离散等级，取前 30。
- b) 准常量过滤（众数占比 ≥ 0.99，实际删掉 0 列）；再用 Spearman |ρ| > 0.92 贪心剪枝，按与 `target_pm25_3h` 的 |ρ| 优先保留。剪掉 21 列，留下 47 列。
- c) 置换重要性：以 train 最后 20% 的时间（2024-12-19 05:00 起，15,606 行）作内部留出，留出前 3 小时的训练行删掉，在其余 62,138 行上训练默认 RF，评分 Macro-F1，重复 5 次，取均值大于 0 的前 30。
- 三者并集共 64 列。

**周期编码成对（v2，1 号要求的缺陷修补）。** v1 只选了 `hour_sin`。单独一个正弦值无法区分对称的时刻（例如 3 点和 9 点的正弦值相同），所以只保留一半周期编码是缺陷。现在的规则：
- `hour`、`dayofweek`、`month` 的 sin/cos 各自作为一个单元；任一个进并集，两个都进（本次六个周期列本来都在并集里）。
- RFE 每轮淘汰重要性最低的单元，成对单元的重要性是两列之和；如果去掉一对会少于 25 列，就跳过这一对，去掉下一个最弱的单列。最后恰好 25 列，周期项都是成对出现。
- 结果：`hour_sin` 和 `hour_cos` 都入选，星期和月份的两对都没入选；与 v1 相比加了 `hour_cos`、去了 `gran_low_12h`。

**消融的设定**（默认 `RandomForestClassifier`，种子 42、43、44，在 val 上评估）：
- ① 白名单里 12 个原始列 → ② 加滞后、滚动、变化率和周期编码 → ③ 加风分量和近场、远场空间 → ④ 加 5 列粒化。
- 两个因素分开看（QN-002 Q3-2）：以 v1 的不成对 Top-25 为基准 ⑤，加 `hour_cos`（⑥）；加最近 3 格点均值（⑦，压缩版对照）；两者都加（⑧）。
- ⑨ 交付的成对 Top-25；**⑩ 在 ⑨ 上加邻域 50 列**，用来判断邻域特征要不要并入交付版。
- ⑪、⑫ 在 ⑤ 上加邻域 50 列（不加 / 加 `hour_cos`），与 7 号的拆分口径对应。
- 其余设置（模型、种子、样本、指标）与 ①~④ 完全相同。

**最终表只交可直接训练的行**（3 号的决定）：下游模型要求每一行 Top-25 都是有限值，而全面板里有大量非受体行和边界行带 NaN。

---

## 4. 已知缺陷和坑

1. **消融结果。**

   | 组 | 特征数 | Macro_F1 | High_Risk_Recall | MAOE | Avg_Asymmetric_Cost |
   |---|---:|---|---|---|---|
   | M0 持续性 | — | 0.6095 | 0.7051 | 0.3811 | 0.9266 |
   | ① Raw only | 12 | 0.6666 ± 0.0012 | 0.6597 ± 0.0029 | 0.3053 | 0.8547 |
   | ② + Lag/Rolling | 53 | 0.7125 ± 0.0013 | 0.7642 ± 0.0022 | 0.2566 | 0.6419 |
   | ③ + Spatial | 63 | 0.7103 ± 0.0006 | 0.7555 ± 0.0010 | 0.2579 | 0.6545 |
   | ④ + Granulation | 68 | 0.7129 ± 0.0018 | 0.7623 ± 0.0006 | 0.2561 | 0.6419 |
   | ⑤ v1 Top-25（不成对） | 25 | 0.7107 ± 0.0013 | 0.7664 ± 0.0019 | 0.2597 | 0.6497 |
   | ⑥ ⑤ + `hour_cos` | 26 | 0.7142 ± 0.0021 | 0.7687 ± 0.0013 | 0.2557 | 0.6399 |
   | ⑦ ⑤ + 最近 3 格点均值（压缩版，对照） | 26 | 0.7131 ± 0.0011 | 0.7665 ± 0.0019 | 0.2565 | 0.6479 |
   | ⑧ ⑤ + `hour_cos` + 最近 3 格点均值 | 27 | 0.7129 ± 0.0016 | 0.7669 ± 0.0035 | 0.2563 | 0.6459 |
   | ⑨ **成对 Top-25（交付）** | 25 | **0.7157 ± 0.0010** | **0.7712 ± 0.0008** | **0.2546** | **0.6369** |
   | ⑩ ⑨ + 邻域 50 列 | 75 | 0.7078 ± 0.0018 | 0.7319 ± 0.0009 | 0.2593 | 0.6821 |
   | ⑪ ⑤ + 邻域 50 列 | 75 | 0.7018 ± 0.0022 | 0.7245 ± 0.0025 | 0.2656 | 0.7032 |
   | ⑫ ⑤ + `hour_cos` + 邻域 50 列 | 76 | 0.7072 ± 0.0015 | 0.7312 ± 0.0032 | 0.2602 | 0.6862 |
   | 排列检查：⑤ + 每格点一列（52 列） | 77 | 0.7077 ± 0.0012 | 0.7405 ± 0.0027 | 0.2589 | 0.6742 |

   - ① 到 ② 是主要增益。
   - ② 到 ③：加入 2 号的上风向空间特征后三项都略差（召回 −0.0087，代价 +0.0126）。
   - ③ 到 ④：粒化把空间带来的下降补回来，但相对 ② 只多 0.0004 Macro-F1。
   - **`hour_cos`（⑤→⑥）**：三项都变好（Macro-F1 +0.0035，召回 +0.0023，代价 −0.0098）。补它首先是修编码缺陷。
   - **邻域 50 列：在本模块的默认 RF 下三项都明显变差。** 加在交付版上（⑨→⑩）：Macro-F1 0.7157 → 0.7078，召回 0.7712 → 0.7319，代价 0.6369 → 0.6821；加在 v1 上（⑤→⑪）也一样（Macro-F1 −0.0089，召回 −0.0419，代价 +0.0535）。这和 7 号用他的模型测出的增益方向相反，离 1 号给的并入门槛（Macro-F1 ≥ 0.73、代价 ≤ 0.53）很远。已排除两种可能的错误：抽样核对邻居列与该格点同一时刻的值，0 处不一致；改成每格点固定一列（排列检查）仍然变差，只是幅度小一些。
   - 可能的原因（未验证）：默认 RF 每次分裂只随机抽 √p 个特征，从 25 列变成 75 列后，本格点自己的强特征被抽中的机会变少；50 列里大多数是几百公里外的格点，在 3 小时步长里对受体几乎没有信息。7 号的基础 25 特征 Macro-F1 是 0.7227，高于这里的 0.7107，说明模型设置不同，增益可能依赖模型。
   - **最近 3 格点均值（⑦，压缩版）**：Macro-F1 +0.0024，召回几乎不变；和 `hour_cos` 一起加（⑧）不如只加 `hour_cos`（⑥）。
   - **⑨ 成对重选的 Top-25** 在所有组里三项都最好。因此交付版**不加入**邻域 50 列。结合 ⑦、⑩、⑪ 看，这批格点的邻居信息在 3 小时步长里基本已被受体自身的滞后吸收。
   - 标准差只反映 RF 随机种子，不反映样本波动，组间差在 0.01 以内的不要下强结论（⑩、⑪ 的召回下降约 0.04，远大于这个范围）。

2. **v1 → v2 换了一列，下游要重跑。** `features_final.parquet` 和 `selected_features.json` 都变了（加 `hour_cos`，去 `gran_low_12h`）。规范 v3 §3.3 贴的 25 列清单要由 1 号更新；4、5 号用旧表算出的结果需要用新表重跑。

3. **三种策略取并集几乎没有起筛选作用**：并集 64 列，候选只有 68 列。真正把特征数压到 25 的是 RFE。

4. **行数要对照"可建模行数"，不是面板行数。** 最终表 train 是 77,753 行，比面板受体行数 78,102 少 349 行（2 号合格标记筛掉的）。1 号已在 `PANEL_CONTRACT` 登记 `receptor_{train,val,test}_eligible_rows` = 77,753 / 16,656 / 16,647，下游核对时请对照这一组。

5. **标签 `target_level_3h` 是 int8（契约类型）。** pandas 3 下不能用 `frame.loc[:, "target_level_3h"] = ...` 把 int64 整列写回 int8 列，会报 `TypeError`；要转类型请直接写 `frame["target_level_3h"] = ...`（第 5 节示例就是这样写的）。

6. **Top-25 是用全部 train 选出来的。** 下游在 train 内部做交叉验证调参时，选择这一步用过全部 train 的标签，内部折的分数会略偏乐观；val 和 test 上的评估不受影响。

7. **上游再变时怎么办。**
   - 白名单、合格标记、单文件或分片目录都已兼容，不用改代码。
   - 填补规则只认三类列：`upwind_pm25_advection_{K}h`、`spatial_pm25_gradient`、`far_source_pm25_mean_{W}h`；别的列带 NaN 并被选进 Top-25 时脚本会停下报出列名。
   - 选择脚本会核对粒化表是用当前特征表生成的，对不上就报错。

8. **`feat_granulation.parquet` 是运行时生成的中间文件，不要提交**（已在 `.gitignore` 里）。

---

## 5. 下一棒怎么用

```python
import json
from pathlib import Path

import pandas as pd

sel = json.loads(Path("models/selected_features.json").read_text(encoding="utf-8"))
feats = sel["selected_features"]                      # 25 个，只按这个清单取列
df = pd.read_parquet("data/features_final.parquet")   # 已是受体合格行，Top-25 无 NaN
df["target_level_3h"] = df["target_level_3h"].astype(int)   # pandas 3 不要用 .loc[:, col] 整列写回 int8
train = df[df["split"].eq("train")].sort_values(["timestamp", "station_id"]).reset_index(drop=True)
X, y = train[feats], train["target_level_3h"]
print(df.shape, X.shape)                              # (111056, 32) (77753, 25)
```

- 训练与交叉验证只用 `split == "train"`，`TimeSeriesSplit` 之前必须先按 `timestamp` 排序（上面已排）。
- 超参和阈值在 `val` 上选，`test` 只在最后评估一次。
- 本模块的消融用的是默认 RF；调参后的结果应与 `metrics_stage3.json` 里的 ⑨ 组对照，同一批 val 行（16,656 行，高危 4,842 行）。
- 每张对比表都带上 M0（`metrics_persistence_M0_val`）。

重跑（仓库根目录，2 号特征表在位）：

```sh
python src/03_granulation_features.py
python src/03_feature_selection.py
```

---

## 6. 我没做的事

- 没有把邻域 50 列放进候选或最终表：在默认 RF 下它让三项指标都变差（第 4 节第 1 条）。如果之后改用别的模型（例如提升法）并测出增益，需要另行决定是否加入，下游也要一起重跑。
- 没有用 7 号的模型复现他的邻域增益；两边模型设置不同，结果不能直接比较。
- 选择流程没有按 train 内部的时间折分别重选特征（见第 4 节第 6 条）。
- 风分量 `u_wind`、`v_wind` 没有按成对处理（这次的成对规则只针对 sin/cos 周期编码）。
- 没有加 24h 及以上的粒化窗口。
- 2 号交接建议的误报率和分时段稳定性没有单独评估；消融只报了 `utils_metrics` 的六项指标。
- 没有在 `test` 上做任何评估或选择（按规定留给最后）。
- 规范 v3 §3.3 的 25 列清单是 1 号的文件，没有改；需要 1 号按本表更新。
