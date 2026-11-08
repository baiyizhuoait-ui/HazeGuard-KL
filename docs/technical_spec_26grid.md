# HazeGuard KL 当前 26 格点技术契约

整理日期：2026-11-08。以项目组本次明确确认的 26 格点决定为当前验收依据。

本文件整合已存在于 README、`src/config.py`、数据字典、manifest 和交接记录中的有效约定。审计没有在任何远程分支或可达历史中找到独立的新版规范文件，因此不声称找到了另一份正式 v2.1 文档。原 v2.0 中不冲突的时间特征、风矢量、15 km 基线及文件名继续有效。旧八站内容仅作历史来源，不作为本轮待完成项。

## 1 数据与空间单元

- 只读输入：`data/haze_panel_clean.parquet`，966,677 行 × 35 列。
- 输入 SHA256：`97b5db123352861b6cd41224daf1ce278c90d6d528031b9bdc2744317a5ead93`。
- 有效记录范围：2022-08-04 08:00 至 2026-10-31 20:00，`Asia/Kuala_Lumpur`。采集配置日期和最终清洗后起止时间有区别。
- 唯一键：`station_id, timestamp`；保留键名用于兼容，但 station_id 表示模式格点采样序列，不是地面监测站。
- 单一注册表：`src/config.py::NODES`，26 ID，3 receptor、10 corridor、13 source。实际 ID、角色和坐标必须与注册表核对。
- 主训练/评价域：`role == 'receptor'`，即 KL_City、Port_Klang、Sepang。其他节点供空间输入和后续泛化实验使用，不混入主训练域。
- 经纬度为既有有效采样坐标。未保存全部接口返回的实际网格中心，不能把采样坐标未经验证地称为真实 CAMS 网格中心。
- 实际序列需按相同训练 timestamp 检查重复和相关性。不同序列只能证明所检样本不相同，不能证明模式误差独立或地面真实性。

阈值沿用 `config.LEVEL_THRESHOLDS = (15,35,55)` µg/m³：L0 <15，L1 [15,35)，L2 [35,55)，L3 ≥55。它们是项目实验等级，不是官方小时级 AQI 健康标准。

## 2 文件与输出契约

| 内容 | 有效路径 |
|---|---|
| 输入与字段定义 | data/haze_panel_clean.parquet、data/data_dictionary.json、data/manifest.json |
| EDA | src/02_eda_analysis.py |
| 特征工程 | src/02_spatiotemporal_features.py |
| 正式输出 | data/feat_spatiotemporal.parquet |
| 五张图 | figures/fig_eda_diurnal_seasonal.png、fig_eda_correlation_matrix.png、fig_eda_class_imbalance.png、fig_eda_wind_rose_pollution.png、fig_eda_spatial_crosscorr.png |
| 英文章节 | reports/sec2_3_and_2_4_1_agent2.md |
| 统计和验收 | models/metrics_stage2_eda.json、metrics_stage2_features.json、metrics_stage2_verification.json |
| 测试/交接 | tests/test_agent2*.py、docs/handover/02_eda_and_spatiotemporal.md |

输出保留全部原字段、值、类型、行顺序和标签，不覆盖输入。当前 91 列 = 35 原列 + 56 新增列。新增连续值 float32，审计标记 bool；不插值、不随机补数据、不改 split、不拟合模型。

| 新增族 | 数量 | 定义 |
|---|---:|---|
| Lag | 24 | pm2_5/pm10/no2/relative_humidity_2m，分别滞后 1/2/3/6/12/24 小时 |
| Rolling | 9 | PM2.5 在3/6/12小时的mean、sample std(ddof=1)、range |
| Change/trend | 2 | x(t)−x(t−3)、6小时OLS斜率 |
| Cyclical | 6 | hour/dayofweek/month 的 sin/cos，周期24/7/12，month从0编码 |
| Wind | 2 | u_wind、v_wind，m/s |
| Spatial/availability | 7 | 两档上风向均值、当前空间梯度、两档权重和及有效邻居数 |
| Offline audit | 6 | agent2_*源质量、历史完整性、未来标签核对与评价样本标记 |

预测输入以 feature JSON 的 `model_feature_columns` 为唯一允许清单，共62列（12原始候选+50新增预测列）。禁止输入 `target_*`、`agent2_*`、ID、split、role、region、timestamp，以及未经折内重拟合的 b_proto/异常值字段。

## 3 时间与防泄漏

保留既有逐节点70/15/15时序切分：train 676,884、val 145,054、test 144,739。每个 station_id×split 在内部重建完整小时索引后 shift/rolling，最终只写回原行。窗口包含当前t；仅允许t及更早源值。缺小时不会变成“上一条记录”，也不会静默用零替代。

本轮继续采用严格 split 重启，不跨边界取历史。允许在线使用前段已知历史在因果上可以合理，但会改变目前评估契约，应另行明确实验；本轮不变更。原数据已插值标记的源行不进入新增历史/风/空间计算，保留原值供审计。

target 使用精确t+3小时核对，越界、不对齐或源质量不足只加离线标记，不删除原标签。`agent2_model_eligible` 依赖未来核对，只能用于离线训练/评价筛选；不能作为预测输入或线上发出预测的条件。后续CV还须按全局timestamp分折、同刻格点共同分配、purge标签跨折样本，并折内拟合所有学习型变换。

## 4 风场与空间基线

风速优先 `wind_speed_ms`，有 `wind_speed_kmh` 时检查两者一致。气象来向角θ转弧度后：u=−s·sinθ、v=−s·cosθ；向东/向北为正。静风小于0.5m/s或质量标记静风时矢量归零；缺失有向风不当作可靠方向。

距离使用 Haversine km，方向为候选源j处指向目标i的局部东/北单位向量。排除自身、同坐标及不可用源；每个有效小时使用源风和源浓度共同对齐。

`w(j→i,t) = max(0, wind(j,t) · direction(j→i)) exp(−D²/(2σ²))`，保留 σ=15 km 作为可复现基线。投影浮点零噪声及权重≤1e−12 m/s不归一化。t时刻的1/2小时空间均值使用t−1/t−2的源风和浓度；空间梯度为当前目标浓度减去同刻加权均值。无有效邻居时浓度特征为NaN，权重和/计数为0；没有所需历史时也可能为NaN。

必须报告训练集邻距、有效邻居数、零支持频率、最大归一化权重和有效邻居数1/Σp²，以判断15km核在区域网络中是否过窄。本轮不按测试成绩选择新尺度。权重诊断不等于证明跨海输送轨迹、旅行时间或排放来源。

## 5 EDA 约定

五图均300dpi、英文、保持有效文件名。图1采用训练受体日内均值与七日块bootstrap 95%区间及SW(6–9月)/NE(11–3月)分布；图2只用训练且未来标签安全的受体行，展示Pearson/Spearman；图3展示26格点与受体的等级数量、比例以及split/年份变化；图4展示风速/来向、训练SW季风及受体差异；图5按坐标、role、region预先选邻近/区域候选组合，禁止按相关峰值挑对，比较−12到+12小时的raw与去训练月/小时气候态曲线。

正滞后k定义为corr(source(t−k),destination(t))。标出峰值k及r，但不能据此认定物理传输。图3/5可覆盖全时段作纯描述，不能据此挑选预测特征/带宽/阈值。所有数值来自生成JSON。

## 6 运行、下游和交付

Python3.13.9的独立环境使用 `requirements-agent2.lock.txt`；`requirements-agent2.txt`是可读的直接依赖，原 `requirements.txt`保留给全项目。正式Agent2默认只读当前输入路径；其他版本必须显式传 `--input` 并明确未注明风速的单位，不能默默回退旧八站数据。

```sh
python -m pip install -r requirements-agent2.lock.txt
python src/02_spatiotemporal_features.py
python src/02_eda_analysis.py
python tools/verify_agent2.py
```

成员3当前尚无已提交实现，只验收输入契约，不生成假FIG成果。未来文件名按现行CONTRIBUTING表：`src/03_granulation_features.py`、`src/03_feature_selection.py`，只在训练折拟合、保留正确时间排序与标签purge。

大型特征parquet使用Git LFS；克隆可`git lfs pull`，或用真实清洗输入重建。完整交付ZIP必须带实体数据和逐文件SHA256，不能把LFS指针视为完整数据。`tools/package_agent2.py`从固定提交导出、解压并实际重跑，旁附合并后验收记录。无Git目录时来源由SOURCE_COMMIT.txt明确记录。

历史归档、重复候选证据和其他成员预留分支保留；不删除有效输入/模型、不重写共享历史。清理决策及执行记录见 repository_audit_agent2_26grid.md。
