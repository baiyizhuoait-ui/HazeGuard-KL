# 03 模块交接 多尺度趋势粒化与特征选择

| | |
|---|---|
| 负责人 | 3 号 |
| 任务依据 | 仓库现行规则：`CONTRIBUTING.md`（3 号职责与产物）、`docs/syllabus_coverage.md` §3（只做区间粒化与趋势粒化；互信息 + RFE + permutation importance）、2 号交接文档（白名单、折内拟合） |
| 上游输入 | `feat/02-eda` 提交 `2165c4c`，`data/feat_spatiotemporal.parquet`（4 个有序分片），清单摘要 `208cf00942b404aead0ffe913497d34157ec01f3e1a4ccf593b10f71fb61a55e` |
| 下一棒 | 4 号、5 号（读 `features_final.parquet` 和 `selected_features.json`） |
| 分支 | `feat/03-granulation`，基于 `feat/02-eda`；**合并顺序必须是 02 先、03 后** |
| 日期 | 2026-11-10 |

---

## 1. 我产出了什么

### 数据表

| 文件 | 形状 | 大小 | SHA256 |
|---|---|---|---|
| `data/features_final.parquet` | **111,056 行 × 32 列** | 4.8 MB | `64285f40e81c72ab4872d282d52be68401078e8c33dec50134d93e7877d2f55b` |

行 = 受体格点（`role == "receptor"`）且 `audit_model_eligible` 为真的行，三段行数与 2 号审计 JSON 的 `eligibility_by_receptor_split` 一致：

| `split` | 行数 |
|---|---:|
| `train` | 77,753 |
| `val` | 16,656 |
| `test` | 16,647 |

中间表 `data/feat_granulation.parquet`（全面板 966,677 行 × 12 列，14.7 MB）**不入库**。它由粒化脚本 1 秒内生成，溯源信息写在文件的 parquet 元数据里。

### 模型文件与图

| 文件 | 内容 |
|---|---|
| `models/selected_features.json` | 最终 Top-25 清单（按最终 RF 重要性排序）、三种策略各自的排名、并集、RFE 排名、行筛选与填补规则、粒化溯源 |
| `models/metrics_stage3.json` | 五组消融 × 3 个种子在 val 上的六项指标，均值 ± 标准差，附 M0 持续性基线 |
| `figures/fig_granulation_bands.png` | 粒化行为：KL_City 在 train 期的一次烟霾脉冲前后各 5 天，12h、6h 粒化区间与中枢，以及两种窗口的斜率，300 dpi |
| `figures/fig_feature_ablation.png` | 消融结果：Macro-F1、高危召回、平均非对称代价三联图，带 M0 参考线，300 dpi |

### 代码

| 文件 | 作用 | 耗时 |
|---|---|---|
| `src/03_granulation_features.py` | 区间粒化与趋势粒化，画 `fig_granulation_bands.png` | 约 1 秒 |
| `src/03_feature_selection.py` | 三种选择策略 → 并集 → RF-RFE → Top-25；消融；最终表 | 约 2.5 分钟 |

两份代码在仓库 `src/` 里直接运行，输入和输出都是仓库根目录。所有 JSON 都带 `generated_at`、`git_commit`、`code_sha256`、`input_sha256`。同一输入连跑两次，Top-25、15 次消融结果和最终表都逐字节相同。

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
| 4 | `hour_sin` | 2 号周期 | 无量纲 | bc |
| 5 | `pm2_5_lag_1` | 2 号滞后 | µg/m³ | ac |
| 6 | `no2` | 1 号原始 | µg/m³ | abc |
| 7 | `pm2_5_change_3h` | 2 号变化率 | µg/m³ | abc |
| 8 | `pm10_lag_1` | 2 号滞后 | µg/m³ | abc |
| 9 | `so2` | 1 号原始 | µg/m³ | ab |
| 10 | `no2_lag_12` | 2 号滞后 | µg/m³ | b |
| 11 | `spatial_pm25_gradient` | 2 号近场空间（**缺失已填补**） | µg/m³ | ab |
| 12 | `pm2_5_slope_6h` | 2 号趋势（= 6h 趋势粒化） | µg/m³/h | abc |
| 13 | `gran_slope_12h` | 3 号趋势粒化 | µg/m³/h | ab |
| 14 | `gran_low_12h` | 3 号区间粒化 | µg/m³ | abc |
| 15 | `relative_humidity_2m_lag_3` | 2 号滞后 | % | bc |
| 16 | `gran_low_6h` | 3 号区间粒化 | µg/m³ | ac |
| 17 | `pm2_5_mean_6h` | 2 号滚动（= 6h 粒化中枢） | µg/m³ | abc |
| 18 | `gran_up_6h` | 3 号区间粒化 | µg/m³ | ac |
| 19 | `no2_lag_6` | 2 号滞后 | µg/m³ | b |
| 20 | `o3` | 1 号原始 | µg/m³ | bc |
| 21 | `temperature_2m` | 1 号原始 | °C | b |
| 22 | `no2_lag_2` | 2 号滞后 | µg/m³ | b |
| 23 | `co` | 1 号原始 | µg/m³ | ab |
| 24 | `no2_lag_24` | 2 号滞后 | µg/m³ | ab |
| 25 | `far_source_pm25_mean_24h` | 2 号远场背景 | µg/m³ | b |

按来源统计：1 号原始 7 列、2 号时间特征 12 列、2 号空间 2 列、3 号粒化 4 列。

**空间列缺失填补**（只填上游为 NaN 的格，其余值与上游逐值相等；不拟合任何参数）：

| 列 | 缺失含义 | 填补值 | 填补行数 train / val / test |
|---|---|---|---|
| `upwind_pm25_advection_1h` / `_2h` / `_3h` | 该时刻没有有效的近场上风向邻居 | 本格点 `pm2_5_lag_1` / `_2` / `_3`（邻域等于本格点） | 约 7,585 / 1,623 / 753（每列差 1–2 行） |
| `spatial_pm25_gradient` | 同上 | 0 | 7,587 / 1,623 / 753 |
| `far_source_pm25_mean_48h` | 分段开头不足 48h 历史 | 同刻 `far_source_pm25_mean_24h`（最接近的更短窗口） | 70 / 72 / 72 |

近场空间列在合格受体 train 行里缺 9.76%。五列都在选择和消融之前填补，最终只有 `spatial_pm25_gradient` 进了 Top-25。没有把缺失当作零浓度（2 号交接 §2）。

### 2.2 `selected_features.json`

下游只需要 `selected_features`（有序列表，25 个），4 号代码就按这个键名读。其余键：

| 键 | 内容 |
|---|---|
| `id_columns`、`target_columns` | 最终表里特征以外的列 |
| `row_filter`、`rows_by_split` | 行筛选条件与三段行数 |
| `spatial_fill` | 上面的填补规则与逐段行数 |
| `granulation` | 粒化脚本的溯源：生成时间、上游摘要、代码哈希、缺失统计、作图时段 |
| `candidates`（68）、`exact_duplicates_removed`（5） | 候选清单；与 2 号列完全相同而被去掉的粒化列 |
| `strategy_a_mutual_info`、`strategy_b_variance_spearman`、`strategy_c_permutation` | 三种策略的完整排名和各自选出的集合 |
| `union`（64）、`rfe_ranking`、`final_rf_importance`、`selected_in_strategies` | 并集、RFE 淘汰顺序、最终重要性、每个入选特征来自哪几种策略 |

### 2.3 粒化特征定义（10 列，float32）

窗口 x = [x(t−W+1), …, x(t)]，即含当前时刻的 PM2.5 尾随窗口，W ∈ {6, 12}。

| 列 | 定义 | 单位 |
|---|---|---|
| `gran_low_{W}h`、`gran_up_{W}h` | 窗口最小值、最大值（区间粒的下界、上界） | µg/m³ |
| `gran_core_{W}h` | 窗口均值（中枢） | µg/m³ |
| `gran_width_{W}h` | 上界 − 下界 | µg/m³ |
| `gran_slope_{W}h` | 窗口内最小二乘斜率（趋势粒） | µg/m³/h |

窗口里任一小时缺失（或被标为插值/缺失），该行全部粒化特征为 NaN：全表 6h 约 0.155% 的行、12h 约 0.27% 的行；**所有合格行（26 个节点）都没有 NaN**。

### 2.4 `metrics_stage3.json`

`ablation.<组名>` 下有 `features`、`runs`（每个种子的完整指标字典）、`mean`、`std`；`metrics_persistence_M0_val` 是同一批 val 行上的 M0。指标全部由 `src/utils_metrics.py` 的 `evaluate_predictions` 产出，`High_Risk_Recall` 是 union 口径。

---

## 3. 关键决策和理由

**只做区间粒化与趋势粒化，不做 FCM。** 依据 `docs/syllabus_coverage.md` §3：模糊 C 均值、隶属度和解粒化重构属于课程外内容，且与滚动统计高度重叠。早期按技术规范 v2.0 做过 FCM 版本，在 val 上隶属度和重构误差都没有带来提升（第 4 节第 2 条），这个结论与删掉它的决定一致。

**候选只用 2 号白名单。** 2 号交接写明 `model_feature_columns` 是唯一预测白名单，排除了上游气候态和异常量，所以 1 号的 `pm2_5_anomaly`、`temp_anomaly` 不进候选。候选 = 白名单 63 列 + 去重后的 5 列粒化特征 = 68 列。

**折内拟合。** 2 号交接要求粒化、填补、标准化和特征选择都只在折内训练数据上拟合：
- 粒化特征是窗口统计量，不拟合参数；
- 填补规则也不拟合参数；
- RF 不需要标准化。
因此特征选择是唯一用到标签的步骤，只在受体合格 train 行上拟合，val 和 test 没参与。

**窗口规则照搬 02。** 窗口按 `station_id × split` 在完整小时网格上取；`qc_interpolated` 或 `is_missing` 的行置空；窗口不完整就记 NaN。核对结果：与 2 号同定义的 5 列逐值相等，NaN 位置也相同，覆盖全部 966,677 行。

**和 02 完全重复的粒化列不进候选。** 有 5 列粒化特征与 2 号的列完全相同：
- `gran_core_6h` / `gran_core_12h` = `pm2_5_mean_6h` / `pm2_5_mean_12h`
- `gran_width_6h` / `gran_width_12h` = `pm2_5_range_6h` / `pm2_5_range_12h`
- `gran_slope_6h` = `pm2_5_slope_6h`

这几列保留 2 号的列名；粒化表里仍然有它们，方便按定义画图和核对。真正新增的是 `gran_low_*`、`gran_up_*` 和 `gran_slope_12h`。

**两个窗口长度。** 6h 和 12h 对应短时脉冲和半日尺度。24h 窗口在合格行上可能不完整（合格标记只要求 24h 滞后这一个点），会引入新的缺失，没有加。

**特征选择**（受体合格 train 行，按 `timestamp, station_id` 排序）：
- a) `mutual_info_classif`，标签用离散等级，取前 30。
- b) 先做准常量过滤（众数占比 ≥ 0.99，实际删掉 0 列）；再用 Spearman |ρ| > 0.92 贪心剪枝，按与 `target_pm25_3h` 的 |ρ| 从高到低优先保留。剪掉 21 列，留下 47 列。
- c) 置换重要性：以 train 最后 20% 的时间（2024-12-19 05:00 起）作内部留出，留出前 3 小时的训练行删掉（防目标重叠）。评分用 `utils_metrics` 的 Macro_F1，重复 5 次，取均值大于 0 的前 30。
- 三者并集共 64 列，再用默认 RF 做 RFE（每次删 1 列），收敛到 25 列。

Top-K 取 25：现行规则没有另定数字，协作规范的提交示例写的是 Top-25。

**消融的设定**（默认 `RandomForestClassifier`，种子 42、43、44，在 val 上评估）：
- ① Raw = 白名单里 12 个原始列。
- ② 加滞后、滚动、变化率和周期编码。
- ③ 加风分量和 2 号的近场、远场空间特征。
- ④ 加去重后的 5 列粒化特征。
- ⑤ 选择后的 Top-25，用来对比选择前后。

**最终表只交可直接训练的行**（3 号的决定）。4 号的读取函数要求每一行 Top-25 都是有限值，而全面板里有大量非受体行和边界行带 NaN。代价是：其他 23 个节点不在最终表里，需要时读粒化表和 2 号的特征表；RF 原生的缺失值处理也用不上了。

---

## 4. 已知缺陷和坑

1. **消融结果：增益主要来自滞后和滚动特征。**

   | 组 | 特征数 | Macro_F1 | High_Risk_Recall | Avg_Asymmetric_Cost |
   |---|---:|---|---|---|
   | M0 持续性 | — | 0.6095 | 0.7051 | 0.9266 |
   | ① Raw only | 12 | 0.6666 ± 0.0012 | 0.6597 ± 0.0029 | 0.8547 ± 0.0025 |
   | ② + Lag/Rolling | 53 | 0.7125 ± 0.0013 | 0.7642 ± 0.0022 | 0.6419 ± 0.0038 |
   | ③ + Spatial | 63 | 0.7103 ± 0.0006 | 0.7555 ± 0.0010 | 0.6545 ± 0.0013 |
   | ④ + Granulation | 68 | 0.7129 ± 0.0018 | 0.7623 ± 0.0006 | 0.6419 ± 0.0035 |
   | ⑤ Top-25 | 25 | 0.7107 ± 0.0013 | 0.7664 ± 0.0019 | 0.6497 ± 0.0042 |

   - ① 到 ② 是主要增益。
   - ② 到 ③：加入 2 号的空间特征后，三项指标都略差（召回 −0.0087，代价 +0.0126），差距大于种子标准差。
   - ③ 到 ④：加入粒化特征后恢复到 ② 的水平（召回 +0.0068，代价 −0.0126）。
   - ⑤ 只用 25 列，高危召回在五组里最高（0.7664），Macro-F1 和代价略差于 ④。
   - 标准差只反映 RF 随机种子，不反映样本波动，组间差在 0.01 以内的不要下强结论。
   - 只用原始变量（①）时高危召回 0.6597，低于 M0 的 0.7051。

2. **早期 FCM 版本的结论（存档，不再交付）。** 按技术规范 v2.0 做过 8 原型 FCM：隶属度和重构误差在 val 上没有提升，置换重要性在 0 附近，没有一列进入 Top-25。这支持课程对照表删掉 FCM 的判断，报告里可以作为补充说明。

3. **三种策略取并集几乎没有起筛选作用**：并集 64 列，候选只有 68 列。策略 b 只剪共线、不按重要性截断，一下留下 47 列。真正把特征数压到 25 的是 RF-RFE。

4. **行数契约差 349 行。** 1 号 `src/config.py` 的 `PANEL_CONTRACT["receptor_train_rows"]` 是 78,102；质量通告 QN-001 让 4 号加的核对容差是 ±200；而最终表 train 是 77,753 行。协作规范 §3 第 7 条已写明以审计登记的合格行数为准，**需要 1 号在 `PANEL_CONTRACT` 里登记合格行数**，否则 4 号加上这道核对后会报错。

5. **4 号现有读取函数在 pandas 3 下读不了 int8 标签。** `branch4` 上 `src/04_standard_and_reg_rf.py` 第 132 行 `frame.loc[:, "target_level_3h"] = levels.astype(int)` 会报 `TypeError: Invalid value … for dtype 'int8'`。1 号自己的 `haze_panel_clean.parquet` 也一样报错，原因不在 3 号的表。改成 `frame["target_level_3h"] = levels.astype(int)` 即可（第 123、133 行同理），改后读入完整通过。int8 是契约类型，3 号不改。

6. **Top-25 是用全部 train 选出来的。** 4 号在 train 内部做交叉验证调参时，选择这一步用过全部 train 的标签，内部折的分数会略偏乐观；val 和 test 上的评估不受影响。

7. **上游再变时怎么办。**
   - 白名单、合格标记（`audit_model_eligible`，或拆到 `data/audit_stage2.parquet`）、单文件或分片目录，都已兼容，不用改代码。
   - 填补规则只认三类列：`upwind_pm25_advection_{K}h`、`spatial_pm25_gradient`、`far_source_pm25_mean_{W}h`。如果以后有别的列带 NaN 并且被选进 Top-25，脚本会停下来报出列名。
   - 选择脚本会核对粒化表是用当前特征表生成的，对不上就报错，提醒先重跑粒化脚本。

8. **`feat_granulation.parquet` 是运行时生成的中间文件，不在登记的产物里，不要提交。**

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
- 本模块的消融用的是默认 RF；4 号调参后的结果应与 `metrics_stage3.json` 里的 ⑤ 组对照，同一批 val 行（16,656 行，高危 4,842 行）。
- 每张对比表都带上 M0（`metrics_persistence_M0_val`）。

重跑（仓库根目录，2 号特征表在位）：

```sh
python src/03_granulation_features.py
python src/03_feature_selection.py
```

---

## 6. 我没做的事

- 报告正文（`reports/` 下的英文章节）由 3 号本人写，本模块只提供数字和图。
- 选择流程没有按 train 内部的时间折分别重选特征（见第 4 节第 6 条）。
- 没有加 24h 及以上的粒化窗口。
- 2 号交接建议的误报率和分时段稳定性没有单独评估；消融只报了 `utils_metrics` 的六项指标。
- 没有在 `test` 上做任何评估或选择（按规定留给最后）。
- 远场列只处理了 `far_source_pm25_mean_48h` 的缺失；其余远场列在合格受体行里本来就没有缺失。
- 4 号代码和 1 号的行数登记都没有改（不是 3 号的文件），问题和改法见第 4 节第 4、5 条。
