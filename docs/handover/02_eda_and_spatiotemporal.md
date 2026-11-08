# 02 模块交接 EDA 与时空特征

基于已合并main `3cdf1ba`继续完善；保留01的全部数据、代码、模型和manifest。项目组已正式确认26格点，当前验收以 [26格点技术契约](../technical_spec_26grid.md) 为准，不再将旧八站视为待补工作。

## 1. 我产出了什么

- `src/02_eda_analysis.py` 和 `src/02_spatiotemporal_features.py`。
- `figures/fig_eda_*.png`：规定文件名的五张图，全部 300 dpi，英文图内文字。
- `data/feat_spatiotemporal.parquet`：966,677 行 × 91 列，原 35 列不变，新增 56 列。
- `models/metrics_stage2_eda.json`：全部绘图统计、相关系数、逐格点类别分布、互相关的有效配对数。
- `models/metrics_stage2_features.json`：输入输出 SHA256、列清单、缺失计数、参数、空间覆盖率、候选特征清单和 Gate 0 披露。
- `models/metrics_stage2_verification.json`：检查结果、实际运行环境和已知限制。
- `reports/sec2_3_and_2_4_1_agent2.md`：有数据证据的英文章节技术草稿，需作者核实并遵守课程的写作要求。
- `tests/test_agent2*.py`：原测试及26格点EDA、空间权重诊断的独立测试。
- `docs/repository_audit_agent2_26grid.md`：42个原文件分类、先审查后清理的计划和执行记录。
- `tools/package_agent2.py`：从固定提交打包、展开LFS并实际解压重跑。

完整 parquet 约 81 MB，通过 Git LFS 存放；普通 Git 中只保存指针。克隆后先执行 `git lfs pull`，不要把小指针当成数据读取。精确字节数和完整 SHA256 以 feature audit 为准。

## 2. 字段和接口说明

| 字段族 | 语义与单位 |
|---|---|
| `{pm2_5,pm10,no2,relative_humidity_2m}_lag_{1,2,3,6,12,24}` | 精确小时滞后，保留对应原变量单位 |
| `pm2_5_{mean,std,range}_{3,6,12}h` | 含当前小时的 PM2.5 滚动统计，µg/m³；std 使用 ddof=1 |
| `pm2_5_change_3h`, `pm2_5_slope_6h` | 端点差 µg/m³；六小时最小二乘斜率 µg/m³/h |
| `{hour,dayofweek,month}_{sin,cos}` | 本地日历周期，无量纲 |
| `u_wind`, `v_wind` | 向东、向北的风分量，m/s，静风置零 |
| `upwind_pm25_advection_{1,2}h` | 同 split 内邻居在 t−k 的风与浓度加权均值，µg/m³ |
| `spatial_pm25_gradient` | 本格点 t 时刻 PM2.5 减去同一时刻的上风向均值，µg/m³ |
| `upwind_weight_sum_{1,2}h` | 有效投影权重之和，m/s |
| `upwind_neighbor_count_{1,2}h` | 实际贡献邻居数，计数 |
| `agent2_current_source_clean` | 当前行未被上游标记为插值或仍缺失 |
| `agent2_history_complete` | 本模块要求的滞后和 PM2.5 滚动/趋势全部可计算 |
| `agent2_target_within_split_3h` | 原表中能找到同格点、同 split 的 t+3h 行 |
| `agent2_target_aligned_3h` | 原未来值及分级标签能由精确 t+3h 行核对 |
| `agent2_target_safe_3h` | 上述目标核对通过，未来来源行也未被插值/缺失标记 |
| `agent2_model_eligible` | 当前源行、历史完整性和未来目标检查同时通过 |

新增连续值是 float32，布尔标记是 bool。历史不够、真实缺口、分区边界和无有效邻居均保留 NaN。NaN 不是 0，不能统一填 0。空间权重的数值下限为 1e−12 m/s；自身和同坐标链接排除。无邻居时权重和/计数为零；没有历史时它们也可以为 NaN。

全26格点的上风向浓度/梯度缺失率约73.39%，受体一小时上风向缺失率分别约13.85%（KL_City）、27.13%（Port_Klang）、4.59%（Sepang）；切勿将区域整体缺失率误当每个受体的比例。逐字段精确缺失率在feature JSON。正式表还保留原始NO₂/O₃/CO的少量缺失，成员3必须在训练折内处理。

## 3. 关键决策和理由

所有位移按真实 timestamp 对齐，并在 station_id × split 内计算；不以缺行后的行号代替小时数。不把 train 的历史接到 val，严格执行 v2.0 的边界规则。滚动统计使用 PM2.5，另三项变量实现完整六档滞后。

源浓度和源风矢量始终取相同的 t−k，且源行 split 必须匹配目标行。没有用目标格点的风代替源格点的风。上风向来自气象“来向”的反向矢量，距离为大圆距离 km。保留 sigma=15 km，不用测试集选新尺度，也不声称一两小时足以跨海输送。

日内/目标相关及风玫瑰使用训练数据；类别分布和互相关使用全时段纯描述。图3增加split和年份变化；图4增加SW季风与受体分图，JSON记录全部26格点。日内置信带用连续七天块bootstrap。互相关按坐标、role、region预先选局部及区域候选对，再比较raw和去训练month/hour气候态残差；正k定义为corr(source(t−k),destination(t))，禁止按污染相关性择对。

训练空间支持诊断在 `spatial_training_diagnostics`，包含有效采样坐标距离、零权重、邻居数量、最大归一化权重和有效邻居数1/Σp²。15km为继承基线，不因本次诊断静默换尺度。无有效邻居不等于污染浓度为零。

## 4. 已知缺陷和坑

1. 空气质量是 CAMS 格点估计，不是监测站实测。小时表不等于原生逐小时独立信息。现有抓取脚本未固定气象 models，缓存又未包含响应级元数据，因此不能把天气一概写成“9 km ERA5”。历史存档也未证明在当时已发布，不能直接称为已验证的实时预警。
2. 上游先插值、后划分 split；其线性插值可能使用未来端点。本模块屏蔽所有 qc_interpolated/is_missing 源行，保持原表供审计。原 anomaly/b_proto 仍是上游全 train 拟合产物，默认候选清单将它们排除，不能仅筛 eligibility 后把所有数值列喂给模型。
3. `agent2_target_*` 和 `agent2_model_eligible` 是离线标签/评估筛选标记，依赖未来可核对性，禁止作为输入特征或线上发预测的条件。
4. sigma=15 km 对扩展后的区域网络偏窄，远距离 Sumatra/Borneo 信息几乎不起作用。空间覆盖率小于 100%，必须看 weight_sum 和 neighbor_count。对更大 sigma 的研究应在 train/val 内另做消融并披露，本次未调参。
5. 原始 qc_pm25_gt_pm10、qc_censored 等异常值保留，后续要做敏感性分析。本模块未独立证实 README 中“真实跨境烟霾事件”等归因。
6. Agent 2 使用独立的 `requirements-agent2.txt` 锁定实际验证版本；全新隔离环境的验收记录随最终交付包提供。未改成员 1 的全项目 `requirements.txt`，未声称其固定版本组合已复验。
7. 受体原 train 行数为 78,102（按数据重新统计），可用行 77,753；与旧交接中的 78,100 文字不完全一致，以表和审计 JSON 为准。过滤后 M0/M1 与后续模型须在相同样本上比较。

## 5. 下一棒怎么用

```bash
git lfs pull
python src/01_verify_data.py --check
python src/02_eda_analysis.py
python src/02_spatiotemporal_features.py
python -m pytest tests/ -q
python tools/check_prose.py reports/
```

默认只读 `haze_panel_clean.parquet`，缺少时直接报错，不隐式回退旧方案。兼容旧字段的其他文件必须显式传 `--input`；仅有未注明单位的 `wind_speed_10m` 时必须用 `--wind-speed-unit km/h` 或 `m/s`，不能靠量级猜。正式验证器要求当前26个ID、角色与坐标匹配注册表。

```python
import json
import pandas as pd

df = pd.read_parquet("data/feat_spatiotemporal.parquet")
audit = json.load(open("models/metrics_stage2_features.json", encoding="utf-8"))
feature_columns = audit["model_feature_columns"]
train = df.loc[(df["role"] == "receptor") & (df["split"] == "train")
               & df["agent2_model_eligible"]].sort_values("timestamp")
X = train[feature_columns]
y = train["target_level_3h"]
```

先处理模型不支持的 NaN，填补器只能在训练折拟合。进入交叉验证时按全局 timestamp 划分，并保证同一时刻的各格点不分散到相邻折；还应清除 target 时间触及验证折的训练样本。本模块的顶层 split 标记不能自动替代折内 purge。若使用上游气候态、FCM、标准化或特征选择，必须折内重拟合。LOSO 的拟合也要排除被留出的格点。

## 6. 我没做的事

没有修改01的代码、数据、manifest、配置、模型或原依赖文件；本轮按授权更新README、协作示例与忽略规则。没有重拉API、训练RF、替成员3构造FIG结果、调sigma、验证真实地面浓度、验证火点归因或评估上线延迟。成员3程序尚未提交，只完成其读取契约验证。没有删除共享分支。GitHub合并结果、最终commit和ZIP复验结果以随包交付记录为准。报告为可核查的技术草稿，需作者核实。
