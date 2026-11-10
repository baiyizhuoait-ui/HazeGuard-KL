# 02 模块交接 EDA 与时空特征

成员 3 从现有 `feat/02-eda` 取得第二阶段成果。正式读表入口是 `data/feat_spatiotemporal.parquet` 目录。共享 README 由组长维护，本模块的运行与交接以本页为入口。 [协作规范](../../CONTRIBUTING.md) · [26 格点技术契约](../technical_spec_26grid.md) · [质量通告 QN-001](../quality_notice_001.md)

## 1. 我产出了什么

| 文件 | 用途 |
|---|---|
| [02_eda_analysis.py](../../src/02_eda_analysis.py)、[02_spatiotemporal_features.py](../../src/02_spatiotemporal_features.py)、[stage2_storage.py](../../src/stage2_storage.py) | EDA、特征生成及有序 Parquet 分片。 |
| [feat_spatiotemporal.parquet](../../data/feat_spatiotemporal.parquet) | **成员 3 正式输入目录**。契约为 966,677 行 × 98 列，包含 35 个原始列和 63 个新增列。 |
| [haze_panel_clean.parquet](../../data/haze_panel_clean.parquet)、[data_dictionary.json](../../data/data_dictionary.json) | 清洗输入及原字段字典。输入为 966,677 行 × 35 列。 |
| [日内与季风](../../figures/fig_eda_diurnal_seasonal.png)、[相关矩阵](../../figures/fig_eda_correlation_matrix.png)、[等级分布](../../figures/fig_eda_class_imbalance.png)、[风玫瑰](../../figures/fig_eda_wind_rose_pollution.png)、[空间互相关](../../figures/fig_eda_spatial_crosscorr.png) | 五张英文 300 dpi 图。 |
| [EDA 统计](../../models/metrics_stage2_eda.json)、[特征审计](../../models/metrics_stage2_features.json)、[验收记录](../../models/metrics_stage2_verification.json) | 图表数字、特征白名单、缺失和覆盖率、来源校验及实跑记录。 |
| [英文章节](../../reports/sec2_3_and_2_4_1.md)、[仓库审计](../repository_audit_stage2.md) | 科学结果及工程决策的核对材料。课程提交正文由学生本人完成。 |
| [测试](../../tests/)、[验证器](../../tools/verify_stage2.py)、[打包工具](../../tools/package_stage2.py) | 检验正式分片、图表和测试，按固定提交导出并从 ZIP 复验。 |

清洗输入 SHA256 为 `97b5db123352861b6cd41224daf1ce278c90d6d528031b9bdc2744317a5ead93`。输出为四个按原行顺序写入的 Zstandard 分片，单片上限 50,000,000 字节，按普通 Git 文件交付。实际总字节数及每片的文件名、行数、大小和 SHA256 见特征审计的 `output_bytes`、`output_parts`。`output_sha256` 是按分片名排序、规范序列化后的清单摘要。验收成绩见验证记录中的 `checks` 和 `all_exit_codes_zero`。

## 2. 字段/接口说明

每行是一个格点在一个小时的预测时刻 `t`，联合主键为 `station_id`、`timestamp`。26 个节点分为 3 个受体、10 个走廊和 13 个源区。时间按 `Asia/Kuala_Lumpur` 本地钟解释，清洗输入落盘为无时区 `timestamp[us]`。需要 UTC 时先本地化，再转换。

`target_pm25_3h` 为 `float64` 的 `t+3h` 浓度目标，单位 µg/m³。`target_level_3h` 为 `int8`，阈值为 15、35、55 µg/m³。全表 train/val/test 分别为 676,884 / 145,054 / 144,739 行，三个受体的原始行数分别为 78,102 / 16,737 / 16,728。各段离线合格行数以 `eligibility_by_receptor_split` 为准。

| 新字段 | 含义、单位与可空性 |
|---|---|
| `pm2_5_lag_*`、`pm10_lag_*`、`no2_lag_*`、`relative_humidity_2m_lag_*` | 四个变量各有 1、2、3、6、12、24 小时精确滞后，沿用原单位。可空。 |
| `pm2_5_mean_{3,6,12}h`、`pm2_5_std_{3,6,12}h`、`pm2_5_range_{3,6,12}h` | 含 `t` 的尾随均值、样本标准差（`ddof=1`）和极差，µg/m³。缺完整窗口时为空。 |
| `pm2_5_change_3h`、`pm2_5_slope_6h` | `x(t)−x(t−3h)`（µg/m³）和含 `t` 的六小时最小二乘斜率（µg/m³/h）。可空。 |
| `{hour,dayofweek,month}_{sin,cos}` | 本地周期 24、7、12，月份从 0 编码。无量纲。 |
| `u_wind`、`v_wind` | 东向、北向风分量，m/s。已知静风记为零，来源无效时为空。 |
| `upwind_pm25_advection_{1,2,3}h`、`spatial_pm25_gradient` | 同分区近场源在 `t−1h`、`t−2h`、`t−3h` 的方向加权浓度，以及当前目标减去同刻近场加权浓度，µg/m³。无支持时为空。 |
| `upwind_weight_sum_{1,2,3}h`、`upwind_neighbor_count_{1,2,3}h` | 三档权重和（m/s）及有效邻居数。有可判定历史但无正权重时为零，缺历史或候选源输入全部缺失时为空。仅用于诊断。 |
| `far_source_pm25_mean_{12,24,48}h`、`far_source_pm25_lag_24h` | 源区逐小时背景均值的严格历史滚动量及精确 24 小时滞后，µg/m³。可空。 |
| `audit_source_clean`、`audit_history_complete` | 当前源质量、所需时间滞后和滚动历史的完整性。布尔值。 |
| `audit_target_within_split_3h`、`audit_target_aligned_3h`、`audit_target_safe_3h`、`audit_model_eligible` | 核对同格点同分区的未来标签、源质量和离线选样条件。布尔值。 |

依 QN-001 Q2-2，六个审计字段统一为 `audit_*`，下游应更新字段引用。当前 63 个新增列中，51 个是预测量，6 个是权重诊断，6 个是布尔审计。`model_feature_columns` 是唯一预测白名单，含 51 个新增预测量和 12 个原始量，共 63 列。白名单排除 `target_*`、`audit_*`、六个权重诊断列、ID、时间、角色、分区、原始风向角、km/h 风速及上游气候态和异常量。

逐列缺失及受体覆盖率见 `nan_counts`、`nan_fractions`、`spatial_coverage_by_receptor` 和 `eligible_receptor_spatial_missing_fractions`。离线合格标记检查当前质量、历史和未来标签，可保留空间缺失样本。填补只在训练折拟合。空间浓度缺失与浓度为零有不同含义。

## 3. 关键决策和理由

- 时间特征在 `station_id × split` 的小时网格上生成，原始列、行顺序、值、类型、标签和分区保留。插值或缺失标记行从新历史及空间统计中屏蔽，避免上游插值的未来端点影响新特征。
- 近场使用 Haversine 距离 ≤150 km、同区域组和 `σ=60 km`。KL 与 Peninsula 合为一组，Sumatra 和 Borneo 各自成组。源风和源浓度同小时对齐，正向风投影乘以 `exp(−D²/(2σ²))`。同刻梯度沿用近场定义。50/80 km 灵敏度只在训练集报告，15 km 旧口径只保留为诊断参照。
- 远场把 Sumatra 8 个、Borneo 5 个格点的质量合格 PM2.5 按小时等权平均。每个分区单独构造过去 12、24、48 小时窗口及精确 24 小时滞后。窗口从 `t−1h` 回看，所有小时须有可用区域均值。
- 图 1、2、4 使用训练统计，图 3、5 的全时段面板用于描述。图 5 预选固定坐标和角色的配对。训练扩展诊断使用相同配对，比较原序列和去训练月小时气候态结果。所有峰值及配对样本数留在 EDA 统计中。

## 4. 已知缺陷和坑

- 固定坐标的无序对在 50、60、80、150 km 内分别有 5、5、8、25 对。这是加区域约束前的几何计数。`σ=60 km` 的 5 对达不到 QN-001 要求的 10 对。仅用 150 km 半径筛选就有 10 个节点没有几何候选。增加区域限制后也无法在全网原分母下达到零支持 ≤10%。用相同分母比较旧尺度和新近场的全网覆盖，受体结果另列。实测零支持、原因及受体缺失见特征审计。
- 训练扩展互相关里，6 对近场原序列中有 1 对在候选带内取全窗最大峰（+2 h）。6 对远场原序列中有 3 对在 +18 至 +36 h 候选带内取全窗最大峰，均为 Pontianak 配对。去气候态后这三对的带内峰值减弱，`r` 约为 0.068 至 0.114。峰值表示时间关联，跨境来源、旅行时间和因果传输需要独立证据。
- 26 个序列来自模式采样坐标。325 对训练 PM2.5 序列无完全相同对，最大绝对 Pearson 相关约 0.9086。实际网格中心、模式误差独立性、地面代表性及归档数据在 `t` 的发布时间仍需验证。
- `qc_pm25_gt_pm10`、`qc_censored` 等上游标记可用于敏感性分析。复用气候态或异常量时应逐训练折拟合。更换输入前核对注册表、地区、坐标和风速单位。相同坐标的源会对自身或同坐标目标排除，向第三目标供值仍可能重复计权。
- 分片目录必须完整保留。单片、旧版本指针及旧单文件实体均不满足当前交付契约。课程作业 PDF 的本地副本未提供可核实的 C.2 原文，命名验收依据为项目 QN-001。§D 的课程提交要求与本模块工程交接分别执行。

## 5. 下一棒怎么用

使用 Python 3.13.9。已下载本版本源码 ZIP 时，从解压的根目录开始安装依赖。使用 Git 时取得现有分支，再在仓库根目录创建独立环境。

```sh
git clone --branch feat/02-eda --single-branch https://github.com/baiyizhuoait/HazeGuard-KL.git
cd HazeGuard-KL
python -m venv .venv
```

PowerShell 激活命令为 `.\.venv\Scripts\Activate.ps1`，Bash 为 `source .venv/bin/activate`。用锁文件安装后，从随项目提供的清洗输入生成特征和图表，再运行完整验收。

```sh
python -m pip install -r requirements-stage2.lock.txt
python -m pip check
python src/02_spatiotemporal_features.py
python src/02_eda_analysis.py
python tools/verify_stage2.py
```

已取得完整生成产物时可先运行 `python tools/verify_stage2.py`。验证器会写本次 `models/metrics_stage2_verification.json`，建议在自己的工作副本运行。打包工具从 Git 检出目录按固定提交导出，并在解压后重新生成和验收。已核实 HEAD 时可运行 `python tools/package_stage2.py --commit HEAD --output ../HazeGuard-KL-stage2-verified.zip`。

```python
import json
from pathlib import Path

import pandas as pd

audit = json.loads(Path("models/metrics_stage2_features.json").read_text(encoding="utf-8"))
panel = pd.read_parquet("data/feat_spatiotemporal.parquet")
assert panel.shape == (966677, 98)
feature_columns = audit["model_feature_columns"]
assert len(feature_columns) == 63
eligible = panel.loc[panel["role"].eq("receptor") & panel["audit_model_eligible"]]
train = eligible.loc[eligible["split"].eq("train")].sort_values(["timestamp", "station_id"])
X = train[feature_columns]
y_level = train["target_level_3h"]
y_pm25 = train["target_pm25_3h"]
```

FIG 尾随窗口先在完整小时网格按格点与分区构造，再应用离线合格条件。内部验证以全局唯一 `timestamp` 向前分折，同一时间的受体同折。训练折删除满足 `timestamp + 3h >= 首个验证时刻` 的样本。FIG/FCM、填补、标准化和特征选择均仅在折内训练数据拟合，最后才用 `test` 评价。留一格点时在拟合中排除留出格点。

成员 3 按协作规范使用 `src/03_granulation_features.py`、`src/03_feature_selection.py`，交付 `data/features_final.parquet`、`models/selected_features.json` 及两张指定图。保持受体、样本、分区、步长和模型设置一致，沿用 `src/utils_metrics.py` 及 M0 基线。`High_Risk_Recall` 使用 L2∪L3 的 union 口径。

## 6. 我没做的事

- FIG 原型、特征筛选、最终建模表及模型增益验证由成员 3 继续。
- 独立地面验证、归档可用性、上线延迟、火点归因及因果传输评估仍待后续研究。当前成果支持回顾性实验。
- 正式课程提交正文由学生本人完成。现有英文章节是供核对结果和出处的技术材料。
