# 02 模块交接 EDA 与时空特征

成员 3 从已有的 **`feat/02-eda`** 读取 `data/feat_spatiotemporal.parquet`。2026-11-09 核查的成果基准提交为 `bbf2f51820bb77cac6a396673a8515c68f36ca80`。历史 PR #1–#3 的合并内容已在 `main` 撤回，核查时 `main` 为 `24ceeb6617ab12b03350bf6f974aa730243f15e4`。本轮按确认的分支分工在现有 02 分支交付，采用协作规范第 7 节的六段模板。 [协作规范](../../CONTRIBUTING.md) · [26 格点技术契约](../technical_spec_26grid.md)

协作规范第 2 节的合并及分支清理流程适用于另行安排的项目整合。成员 3 本轮可直接使用 02 分支的正式数据开始工作。

## 1. 我产出了什么

| 工作 | 状态和证据 |
|---|---|
| EDA | 已生成五张英文 300 dpi PNG，含季风、时段分布、变量关联、风场和空间互相关。各图的 SHA256 已通过验收器核对。 |
| 时间滞后 | 已完成四个变量的 1、2、3、6、12、24 小时滞后。 |
| 滚动统计与趋势 | 已完成 PM2.5 的 3、6、12 小时均值、样本标准差与极差，以及 3 小时变化和 6 小时趋势。 |
| 时间周期 | 已完成小时、星期和月份各一对正弦余弦。 |
| 风场 | 已完成东向与北向风分量和静风、缺失处理。 |
| 动态上风向 | 已完成两档方向加权浓度、当前梯度、权重与邻居数诊断。 |
| 26 格点适配 | 已用正式数据核对唯一 ID、坐标、角色、序列差异、原始列与主键。 |

| 文件 | 用途 |
|---|---|
| [02_eda_analysis.py](../../src/02_eda_analysis.py)、[02_spatiotemporal_features.py](../../src/02_spatiotemporal_features.py) | 图表和特征生成入口。 |
| [feat_spatiotemporal.parquet](../../data/feat_spatiotemporal.parquet) | **成员 3 正式输入**。966,677 行 × 91 列，35 个原始列加 56 个新列。实体为 80,606,601 字节。 |
| [haze_panel_clean.parquet](../../data/haze_panel_clean.parquet)、[data_dictionary.json](../../data/data_dictionary.json) | 成员 1 清洗输入及原字段字典。966,677 行 × 35 列。 |
| [日内与季风](../../figures/fig_eda_diurnal_seasonal.png)、[相关矩阵](../../figures/fig_eda_correlation_matrix.png)、[等级分布](../../figures/fig_eda_class_imbalance.png)、[风玫瑰](../../figures/fig_eda_wind_rose_pollution.png)、[空间互相关](../../figures/fig_eda_spatial_crosscorr.png) | 正式五图。 |
| [EDA 统计](../../models/metrics_stage2_eda.json)、[特征审计](../../models/metrics_stage2_features.json)、[验收记录](../../models/metrics_stage2_verification.json) | 图表统计、特征白名单、逐列缺失、生成来源及既有测试记录。 |
| [英文章节](../../reports/sec2_3_and_2_4_1_agent2.md) | Section 2.3 和 Section 2.4.1 技术草稿。课程提交正文由学生本人撰写，见协作规范第 9 节。 |
| [测试目录](../../tests/)、[验证器](../../tools/verify_agent2.py)、[打包工具](../../tools/package_agent2.py) | 四个 Agent 2 测试模块。验证器验收本地实体与图表，打包工具从固定 Git 提交制作实体 ZIP。 |

- 清洗输入 SHA256 为 `97b5db123352861b6cd41224daf1ce278c90d6d528031b9bdc2744317a5ead93`。
- 正式特征 SHA256 为 `0d508545afe359109b6b194151607b34134cefb1f289fab6dbc31c267cfff031`。
- 本轮从上述基准提交下载 Git LFS 实体并在隔离副本复验，原始数据检查 **11/11**，pytest **58 通过、0 跳过**，编译通过。文风检查返回成功，有一项句长提示。测试还报告了 14 项依赖弃用警告。仓库统计 JSON 的 `git_commit` 是各产物的生成来源，交接文档提交另查 GitHub 分支历史。

## 2. 字段/接口说明

每行对应一个格点在一个小时的预测时刻 `t`，联合主键为 `station_id` 和 `timestamp`。`station_id` 指向 `src/config.py` 的节点注册表，26 个节点分为 3 个受体、10 个走廊和 13 个源区。`timestamp` 按 `Asia/Kuala_Lumpur` 本地钟解释，文件存储为无时区 `timestamp[us]`，在已验证的 pandas 环境读取为 `datetime64[us]`。协作规范第 4 节写的是 `datetime64[ns]`，应以实际文件读入精度为准。需要 UTC 时先按吉隆坡时区本地化，再转换。

`target_pm25_3h` 为 `float64` 的 `t+3h` 浓度目标，单位 µg/m³。`target_level_3h` 为 `int8` 的同一预测步长等级，项目阈值为 15、35、55 µg/m³。`split` 沿用成员 1 的时序划分，`train` 用于拟合，`val` 用于验证和选择，`test` 留给最后评价。

| `split` | 全表行数 | 受体原始行数 | 受体离线合格行数 |
|---|---:|---:|---:|
| `train` | 676,884 | 78,102 | 77,753 |
| `val` | 145,054 | 16,737 | 16,656 |
| `test` | 144,739 | 16,728 | 16,647 |

前 50 个新增字段是 `float32`，最后六个是布尔审计字段。所有字段名如下。

| 新字段 | 含义、单位与可空性 |
|---|---|
| `pm2_5_lag_1`, `pm2_5_lag_2`, `pm2_5_lag_3`, `pm2_5_lag_6`, `pm2_5_lag_12`, `pm2_5_lag_24` | PM2.5 的精确小时滞后，µg/m³。历史缺失时为 NaN。 |
| `pm10_lag_1`, `pm10_lag_2`, `pm10_lag_3`, `pm10_lag_6`, `pm10_lag_12`, `pm10_lag_24` | PM10 的精确小时滞后，µg/m³。可空。 |
| `no2_lag_1`, `no2_lag_2`, `no2_lag_3`, `no2_lag_6`, `no2_lag_12`, `no2_lag_24` | NO₂ 的精确小时滞后，µg/m³。可空。 |
| `relative_humidity_2m_lag_1`, `relative_humidity_2m_lag_2`, `relative_humidity_2m_lag_3`, `relative_humidity_2m_lag_6`, `relative_humidity_2m_lag_12`, `relative_humidity_2m_lag_24` | 相对湿度的精确小时滞后，%。可空。 |
| `pm2_5_mean_3h`, `pm2_5_std_3h`, `pm2_5_range_3h`, `pm2_5_mean_6h`, `pm2_5_std_6h`, `pm2_5_range_6h`, `pm2_5_mean_12h`, `pm2_5_std_12h`, `pm2_5_range_12h` | 含当前 `t` 的尾随窗口均值、样本标准差（`ddof=1`）和极差，µg/m³。窗口不完整时为 NaN。 |
| `pm2_5_change_3h`, `pm2_5_slope_6h` | `x(t)−x(t−3h)`（µg/m³）和含当前小时的六小时最小二乘斜率（µg/m³/h）。可空。 |
| `hour_sin`, `hour_cos`, `dayofweek_sin`, `dayofweek_cos`, `month_sin`, `month_cos` | 吉隆坡本地周期，周期为 24、7、12，月份从 0 编码。无量纲，当前数据无缺失。 |
| `u_wind`, `v_wind` | 东向与北向风分量，m/s。已知静风记为零，来源无效时为 NaN。 |
| `upwind_pm25_advection_1h`, `upwind_pm25_advection_2h` | 同一分区内源节点在 `t−1h`、`t−2h` 的方向加权浓度，µg/m³。无有效邻居时为 NaN。 |
| `spatial_pm25_gradient` | 当前目标 PM2.5 减去同刻方向加权邻域 PM2.5，µg/m³。可空。 |
| `upwind_weight_sum_1h`, `upwind_weight_sum_2h`, `upwind_neighbor_count_1h`, `upwind_neighbor_count_2h` | 两档有效权重和（m/s）与邻居计数。可用历史中无有效邻居时为零，历史不足时为 NaN。 |
| `agent2_current_source_clean`, `agent2_history_complete` | 当前行通过源质量检查，或所需时间历史全部可算。布尔值。 |
| `agent2_target_within_split_3h`, `agent2_target_aligned_3h`, `agent2_target_safe_3h` | 是否找到同格点同分区未来行，是否能核对浓度与等级，以及是否通过分区、对齐和未来来源质量检查。布尔值。 |
| `agent2_model_eligible` | 当前源、时间历史和未来目标审核均通过。仅供离线选样，布尔值。 |

模型列以 [特征审计](../../models/metrics_stage2_features.json) 的 `model_feature_columns` 为唯一允许清单。62 列包含 50 个新增预测列和 12 个原列，即 `pm2_5`, `pm10`, `no2`, `so2`, `o3`, `co`, `temperature_2m`, `relative_humidity_2m`, `wind_speed_ms`, `precipitation`, `lat`, `lon`。`target_*` 只作标签，`agent2_*` 只作审计，ID、时间、角色和分区只用于索引或筛选。原始风向角、km/h 风速、上游 `b_proto_*` 和 `*_anomaly` 也在当前允许清单之外。线上发报依据预测时刻可验证的数据可用性，禁止用含未来标签核对的标记做门控。

逐列空值数量和比例见同一 JSON 的 `nan_counts` 和 `nan_fractions`。全网三项空间浓度或梯度各缺约 73.39%。在受体离线合格样本中，候选列只有这三项空间量有 NaN，`train` 约 16.20%，`val` 约 16.15%，`test` 约 9.70%。全网历史滞后每列缺 534 至 2,387 行，滚动列每列缺 888 至 2,606 行，风分量各缺 443 行。真实缺口、分区暖机和零方向支持有不同含义。填补统计须在训练折拟合，空间浓度缺失不能视作零浓度。

## 3. 关键决策和理由

- 时间特征以 `station_id × split` 为边界在小时网格计算。缺小时保留空位，窗口只读 `t` 及之前的有效记录。不同顶层分区各自开始累积历史，原表行顺序、值、类型、标签和分区保留。被标记插值或缺失的源行从新历史和空间计算中屏蔽，因为上游插值可能使用未来端点。
- 风向使用气象来向。计算 `u = −s sin(θ)` 和 `v = −s cos(θ)`，其中 `s` 用 m/s，`θ` 转弧度。空间聚合用源节点在相同小时的风和浓度，将向目标的正向投影乘以 `exp(−D²/(2σ²))`。`D` 为大圆距离，`σ = 15 km` 是固定的可复现基线。训练诊断记录支持率和权重集中度。
- 图 1、2、4 的有关统计使用训练数据，图 3、5 的全时段结果用于描述。空间互相关按坐标和角色预选比较对象，再比较原始与去训练月小时气候态曲线。预测特征选择依照独立时序验证，测试集仅用于最终评价。

## 4. 已知缺陷和坑

- 当前 26 个坐标对应 CAMS 模式采样序列，浓度和风场均为模式估计。325 对训练 PM2.5 序列中无完全相同对，最大绝对 Pearson 相关约 0.9086。样本坐标和实际网格中心仍须区分，模式误差的独立性、地面有效性和归档数据在 `t` 的真实发布时间都待验证。
- 动态上风向是风场启发的空间统计。跨境来源、旅行时间和污染物因果传输需要额外的观测与物理证据。15 km 基线在全网训练时刻有约 73.31% 缺少有效方向支持，区域贡献容易集中于少数来源。
- 上游质量标记仍有建模意义。`qc_pm25_gt_pm10`、`qc_censored` 等保留供敏感性分析。要复用上游气候态或派生异常量，须在每个训练折重新拟合并独立记录，否则保留当前白名单。
- 已复现两项外部输入防护缺口。同坐标源向第三目标供值时可能重复计权，正负无穷值也可能进入滞后列。当前 26 节点坐标无重复，正式预测列无无穷值。更换输入时先检查源坐标唯一性及数值的有限性。
- GitHub 自动源码 ZIP 可能带 133 字节 LFS 指针。成员 3 应取得实体或按第 5 段重建后再读取。已有验证记录采用 Python 3.13.9 和 `requirements-agent2.lock.txt`，全项目依赖的其他组合仍需单独验证。

## 5. 下一棒怎么用

先取得已有 02 分支。以下命令以仓库根目录为工作目录，Python 版本为 3.13.9。克隆后在已激活的隔离环境安装依赖。仅使用源码 ZIP 时，从已解压的根目录开始安装，直接执行重建命令。

```sh
git clone --branch feat/02-eda --single-branch https://github.com/baiyizhuoait/HazeGuard-KL.git
cd HazeGuard-KL
git lfs pull
python -m venv .venv
```

PowerShell 可运行 `.venv\Scripts\Activate.ps1`。Bash 可运行 `source .venv/bin/activate`。

```sh
python -m pip install -r requirements-agent2.lock.txt
python -m pip check
python src/01_verify_data.py --check
```

需要重建特征和图表时，用随项目提供的 `data/haze_panel_clean.parquet` 运行以下入口。此前在无 Git 的独立源码包中实际重建过，特征实体和五图与原审计哈希一致。`git lfs pull` 和 `tools/package_agent2.py` 适用于 Git 检出目录。

```sh
python src/02_spatiotemporal_features.py
python src/02_eda_analysis.py
```

拿到实体和五图后，可按下面命令验收。`verify_agent2.py` 会写本次 `models/metrics_stage2_verification.json`，建议在个人工作副本运行。`pytest` 单独运行时可能因缺实体而跳过集成测试，应以验证器的完整结果为准。

```sh
python -m pytest tests/ -q -rs
python tools/check_prose.py reports/
python tools/verify_agent2.py
```

```python
import json
from pathlib import Path

import pandas as pd

feature_path = Path("data/feat_spatiotemporal.parquet")
with feature_path.open("rb") as stream:
    if stream.read(4) != b"PAR1":
        raise ValueError("请先取得完整 LFS 实体或运行特征生成器")

audit_path = Path("models/metrics_stage2_features.json")
audit = json.loads(audit_path.read_text(encoding="utf-8"))
panel = pd.read_parquet(feature_path)
feature_columns = audit["model_feature_columns"]
eligible = panel.loc[panel["role"].eq("receptor") & panel["agent2_model_eligible"]]
train = eligible.loc[eligible["split"].eq("train")].sort_values(["timestamp", "station_id"])
X = train[feature_columns]
y_level = train["target_level_3h"]
y_pm25 = train["target_pm25_3h"]
print(panel.shape, X.shape)
```

以上读入的预期形状为 `(966677, 91)` 和 `(77753, 62)`。实现 FIG 时先在按格点与分区排序的完整小时网格构造截至 `t` 的尾随窗口，再使用离线合格标记筛样本。复用已交付字段及原名。主训练和评价只用 `role == 'receptor'`，其他格点留作已定义的空间输入和泛化对照。

内部验证以全局唯一 `timestamp` 排序并划分时间折。同一时间戳的受体放在同一折，删除训练折中满足 `timestamp + 3h >= 首个验证时刻` 的行。按实际小时处理标签间隔。每折仅用该折训练行拟合 FIG/FCM 原型、填补、标准化和特征选择。候选方案用按时间向前的验证折和保留 `val` 选取，最后才使用 `test`。若做留一格点，拟合阶段排除被留出的格点。

成员 3 的目标文件按协作规范为 `src/03_granulation_features.py`、`src/03_feature_selection.py`，产物为 `data/features_final.parquet`、`models/selected_features.json` 及两张指定图。评价时固定同一受体、样本、分区、预报步长和模型设置，比较原始候选、时序/空间特征、加入 FIG、以及特征筛选前后的验证结果。沿用 `src/utils_metrics.py` 的指标口径，在 `fig_feature_ablation.png` 记录增量，并在 `fig_granulation_bands.png` 说明粒化行为。对于高等级预警，同时看高危召回、误报与时间段稳定性。

## 6. 我没做的事

- 成员 3 的 FIG 原型学习、特征筛选和 `features_final.parquet` 属于下一棒。当前交接验收覆盖输入契约和读入，模型增益待其时序消融验证。
- 区域尺度调参、实时延迟、独立地面数据验证及火点归因仍待后续研究。当前成果支持回顾性实验。
- 正式课程报告正文由学生本人完成。现有英文章节提供技术结果和出处供核对。
