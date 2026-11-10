# HazeGuard KL 26 格点技术契约

2026-11-10 更新。当前范围采用 26 格点。第二阶段的空间设计、命名和交付依据 QN-001 的修订，沿用既有时间特征和已登记的主读表路径。 [质量通告](quality_notice_001.md) · [成员 2 交接](handover/02_eda_and_spatiotemporal.md)

## 1 数据与空间单元

- 只读输入 `data/haze_panel_clean.parquet` 为 966,677 行 × 35 列。SHA256 为 `97b5db123352861b6cd41224daf1ce278c90d6d528031b9bdc2744317a5ead93`。
- 有效记录从 2022-08-04 08:00 至 2026-10-31 20:00。时钟采用 `Asia/Kuala_Lumpur`，输入落盘精度为 `timestamp[us]`。
- 唯一键为 `station_id, timestamp`。节点注册表 `src/config.py::NODES` 包含 26 个模式采样坐标和角色，分为 3 个受体、10 个走廊和 13 个源区。样本坐标与实际模型网格中心需要区分。
- 主训练和评价域为 `role == 'receptor'`，包含 KL_City、Port_Klang、Sepang。其余节点提供空间输入和泛化对照。序列差异及相关性在训练时刻计算，模式误差的独立性和地面代表性仍须验证。

等级阈值为 15、35、55 µg/m³，定义 L0 <15、L1 [15,35)、L2 [35,55)、L3 ≥55。它们是项目实验等级。高危召回使用 L2∪L3 的 union 口径，并保留 M0 持续性对照。

## 2 文件与输出契约

| 内容 | 有效路径 |
|---|---|
| 输入与字段定义 | `data/haze_panel_clean.parquet`、`data/data_dictionary.json`、`data/manifest.json` |
| EDA、特征、存储 | `src/02_eda_analysis.py`、`src/02_spatiotemporal_features.py`、`src/stage2_storage.py` |
| 正式输出 | `data/feat_spatiotemporal.parquet/part-*.parquet` |
| 五张图 | `figures/fig_eda_diurnal_seasonal.png`、`figures/fig_eda_correlation_matrix.png`、`figures/fig_eda_class_imbalance.png`、`figures/fig_eda_wind_rose_pollution.png`、`figures/fig_eda_spatial_crosscorr.png` |
| 英文章节 | `reports/sec2_3_and_2_4_1.md` |
| 统计和验收 | `models/metrics_stage2_eda.json`、`models/metrics_stage2_features.json`、`models/metrics_stage2_verification.json` |
| 测试和交接 | `tests/test_stage2_*.py`、`docs/handover/02_eda_and_spatiotemporal.md` |

输出保留原字段、值、类型、行顺序、标签和分区。正式目录由四个有序 Zstandard 分片组成，单片不超过 50,000,000 字节。`pd.read_parquet("data/feat_spatiotemporal.parquet")` 读取完整逻辑表。契约为 966,677 行 × 98 列，35 个原始列加 63 个新增列。

| 新增族 | 数量 | 定义 |
|---|---:|---|
| 精确滞后 | 24 | 四个变量各有 1、2、3、6、12、24 小时滞后。 |
| 滚动统计 | 9 | PM2.5 的 3、6、12 小时均值、样本标准差（`ddof=1`）与极差。 |
| 变化与趋势 | 2 | 三小时变化和六小时 OLS 斜率。 |
| 周期 | 6 | 小时、星期、月份各一对正弦和余弦。 |
| 风矢量 | 2 | `u_wind`、`v_wind`，m/s。 |
| 近场浓度和梯度 | 4 | `upwind_pm25_advection_{1,2,3}h` 与 `spatial_pm25_gradient`。 |
| 近场支持诊断 | 6 | 三档 `upwind_weight_sum_*` 和 `upwind_neighbor_count_*`。 |
| 远场背景 | 4 | `far_source_pm25_mean_{12,24,48}h` 与 `far_source_pm25_lag_24h`。 |
| 离线审计 | 6 | `audit_source_clean`、`audit_history_complete`、`audit_target_within_split_3h`、`audit_target_aligned_3h`、`audit_target_safe_3h`、`audit_model_eligible`。 |

新增连续值为 `float32`，审计列为 `bool`。按 QN-001 Q2-2 同步迁移六列及下游引用。预测输入仅从特征审计的 `model_feature_columns` 取列，共 63 列，含 12 个原始候选与 51 个新增预测量。六个空间支持诊断及六个审计标记保留在逻辑表中，供下游按主键核对。标签、索引、分区、角色、地区、上游气候态和异常量也在当前预测白名单之外。

## 3 时间与防泄漏

原时序切分为 train 676,884、val 145,054、test 144,739。节点内的每个分区单独重建完整小时网格。滞后按真实小时，PM2.5 滚动和趋势截至 `t`，结果只写回原行。插值及缺失标记的源行从新历史和空间聚合中屏蔽。缺历史留空，保持分区独立暖机。

目标用同格点精确 `t+3h` 核对。`audit_model_eligible` 用于离线训练和评价筛样本，包含未来标签核对，线上可用性应独立按 `t` 时点核实。后续 CV 同一时刻的受体同折，并从训练折清除标签跨验证边界的行。填补、变换、FIG/FCM 和选择仅用本折训练数据拟合，最终测试集留给报告。

## 4 近场、远场和支持诊断

以气象来向计算 `u=−s·sinθ`、`v=−s·cosθ`，其中 `s` 用 m/s，`θ` 转弧度。优先读取 `wind_speed_ms` 并核对可用的 km/h 列。风速已知时，低于 0.5 m/s 或已有静风标记的矢量归零。缺风速时，静风标记无法补齐风矢量。

近场源距目标 ≤150 km，KL 与 Peninsula 合为同一组，Sumatra 和 Borneo 各自成组，只用组内候选。排除自身和目标的同坐标候选。对源 `j` 和目标 `i` 以 Haversine 计距 `D`，权重为正向源风投影乘以 `exp(−D²/(2σ²))`，剔除 `≤1e−12 m/s` 的权重。正式 `σ=60 km`。三档浓度使用同分区的 `t−1h`、`t−2h`、`t−3h` 源风与浓度，梯度为 `t` 时刻目标浓度减去同刻近场加权浓度。

远场把 Sumatra、Borneo 13 个源区格点质量合格的 PM2.5 每小时等权平均。每个分区分别计算严格过去 12/24/48 小时的完整窗口均值以及 `t−24h` 值。无源可用小时保留为空。该背景量不使用风权重。

训练集报告近场主尺度、50/80 km 灵敏度及原 15 km 对照，并用相同分母记录所有节点的零权重。区分几何无邻居、候选源输入不可用、静风、无正投影和低于权重下限。分区、节点、目标风速组、有效邻居数及权重集中度见特征审计。无序坐标对在 50/60/80/150 km 内为 5/5/8/25 对，未加区域筛选。故 60 km 的 5 对低于通告所列 10 对。全网和受体缺失按各自分母据实报告。

## 5 EDA 约定

五张图采用英文、300 dpi 及已登记文件名。图 1 为训练受体日内与季风分布，图 2 为训练安全标签的 Pearson/Spearman，图 3 为等级分布及分区和年份差异，图 4 为风向、风速、季风及受体差异，图 5 为按几何和角色预选配对的全时段 ±12 h 互相关。图 3、5 的全时段展示用于描述。

正滞后 `k` 定义为 `corr(source(t−k), destination(t))`。另用同一预选配对在清洁训练样本上计算 ±48 h，分别核对近场 +2 至 +4 h 和远场 +18 至 +36 h。图表和报告以实测峰、样本数及去训练气候态后的结果为准。互相关峰说明时间关联。风权重、来源归属、旅行时间和因果关系各有独立的证据要求。

`models/metrics_stage2_eda.json` 保留报告数字、图表哈希、配对元数据和关键峰值，文件上限 100,000 字节。曲线数组使用统一滞后轴，扩展诊断中的配对编号从零起，对应预选表。

## 6 运行、来源和交付

02 模块在已有 `feat/02-eda` 上交接。共享 README 和协作规范由组长维护。成员 3 使用 [交接入口](handover/02_eda_and_spatiotemporal.md)，使用 Python 3.13.9 和 `requirements-stage2.lock.txt`。

```sh
python -m pip install -r requirements-stage2.lock.txt
python -m pip check
python src/02_spatiotemporal_features.py
python src/02_eda_analysis.py
python tools/verify_stage2.py
```

`requirements-stage2.txt` 记录直接依赖，锁文件记录验收环境。生成器默认读取本地清洗表。替代输入用 `--input` 明确指定，风速单位不明确时使用 `--wind-speed-unit`。源代码 ZIP 可使用清洗输入重建并验收全部分片。

特征审计记录输入 SHA256、脚本 SHA256 和 `output_parts`。目录摘要 `output_sha256` 对有序分片清单计算，各片的 SHA256、字节数和行数均可单独核对。生成统计的 `git_commit` 记录生成时的检出来源。`tools/package_stage2.py` 从 Git 检出目录导出固定提交，写 `SOURCE_COMMIT.txt` 和 `DELIVERY_MANIFEST.json`，解压后验证、重生成并核对每个分片及每张图。

可取得的 AIT201 作业说明副本未提供可核实的 C.2 原文。阶段命名依据 QN-001，正式课程材料按学生本人撰写和课程 §D 提交流程处理。02 工程包用于复现和下游交接。 [仓库审计](repository_audit_stage2.md)
