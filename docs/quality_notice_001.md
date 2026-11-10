# 质量通告 001

| | |
|---|---|
| 编号 | QN-001 |
| 发布人 | 1号（组长） |
| 日期 | 2026-11-10 |
| 收件人 | 2号、4号；抄送全体 |
| 触发原因 | 外部评审 5 条意见 + 组内复查，全部经数据复算证实 |
| 处理要求 | 逐项完成，PR 合并前自查（见 §5） |

> 结论先行：这次复查确认了 5 个真实问题。2 号的工作底子很好（验证面板、58 个测试、
> 交接模板都用对了），4 号的代码防御性也不错（finite 校验、排序都做了）。
> 问题集中在**命名合规**和**两个已经发现却没有执行的设计决策**上，不是能力问题。
> 下面每一条都给了现状证据、具体做法和验收标准。

---

## 一、对 2号 的要求（5 项）

### Q2-1　上风向特征按两层重算【最重要，不做得重跑一遍也没意义】

**现状证据**（来自你们自己交付的 `models/metrics_stage2_features.json`）：

```
parameters.sigma_km = 15.0
coordinate_pairs_within_sigma = 0          ← σ 内一对邻居都没有
overall.zero_weight_rows  = 496,245
overall.zero_weight_fraction = 73.31%      ← 七成行的上风向权重是零
```

`handover/01_data_and_baseline.md` §4.9 已经写了原因和改法，但没有落实。
**现在的 26 格点网络，在这组特征里基本没用上**——除巴生谷内部 3 个格点外，
其余 23 个节点的权重全是 0（Malacca 118km → 4.7e-14，Jambi 565km → 1e-308）。

**要求：拆成两层。**

| 层 | 节点范围 | 参数 | 滞后 | 产出列 |
|---|---|---|---|---|
| 近场短临平流 | 距目标 ≤150 km 的邻居（Port_Klang、Sepang、Seremban、Malacca、Temerloh 及同谷其余受体） | σ = 60 km（另跑 50 / 80 各一次作敏感性） | t−1h、t−2h、t−3h | `upwind_pm25_advection_1h / _2h / _3h` |
| 远场背景负荷 | 苏门答腊 8 + 加里曼丹 5 | 不做平流权重 | 12h / 24h / 48h 滚动均值 + 24h 滞后 | `far_source_pm25_mean_12h`、`far_source_pm25_mean_24h`、`far_source_pm25_mean_48h`、`far_source_pm25_lag_24h` |

原有 `spatial_pm25_gradient` 保留，但只在近场定义。

**验收标准：**

1. 重算后 `spatial_training_diagnostics.overall.zero_weight_fraction` ≤ 10%（现在是 73.31%）；
2. `coordinate_pairs_within_sigma` ≥ 10；
3. 受体行中空间列的 NaN 率 ≤ 5%（现在是 16.20%）；
4. 旧的两列 `upwind_pm25_advection_1h/_2h` 定义被替换（不是并存），交付清单同步更新；
5. 把"近邻滞后 2–4h 达峰、源区滞后 18–36h 达峰"的证据图补进
   `figures/fig_eda_spatial_crosscorr.png` 的图注或报告正文——峰值位置就是传输时间的证据。

### Q2-2　清除全部 agent 命名【合规红线，课程 C.2 关联】

仓库已约定交付物不得出现 `agent` / `智能体` 字样（`CONTRIBUTING.md` §3 第 3 条、
`verify_data.py` 第 13 项门禁）。当前 `feat/02-eda` 有 10 个文件名、6 个特征列名、
若干代码字符串和 PR 名命中。

**逐项清单：**

| 类型 | 现名 | 改为 |
|---|---|---|
| 报告文件 | `reports/sec2_3_and_2_4_1_agent2.md` | `reports/sec2_3_and_2_4_1.md` |
| 依赖文件 | `requirements-agent2.txt` / `.lock.txt` | `requirements-stage2.txt` / `.lock.txt` |
| 测试 | `tests/test_agent2.py` 等 4 个 | `tests/test_stage2_*.py` |
| 工具 | `tools/verify_agent2.py`、`tools/package_agent2.py` | `tools/verify_stage2.py`、`tools/package_stage2.py` |
| 文档 | `docs/repository_audit_agent2_26grid.md` | `docs/repository_audit_stage2.md` |
| 特征列（6 个） | `agent2_current_source_clean`、`agent2_history_complete`、`agent2_target_within_split_3h`、`agent2_target_aligned_3h`、`agent2_target_safe_3h`、`agent2_model_eligible` | 前缀改为 `audit_`：`audit_source_clean`、`audit_history_complete`、`audit_target_within_split_3h`、`audit_target_aligned_3h`、`audit_target_safe_3h`、`audit_model_eligible` |
| README 小节标题 | 「Agent 2 独立运行与成员 3 入口」 | 「02 模块独立运行与 3 号入口」 |
| 分支 / PR 名 | `fix/02-agent2-verified` | 后续 PR 直接从 `feat/02-eda` 发起，不再复用旧名 |

**验收标准**：`python src/01_verify_data.py --check` 的禁词项 PASS（16/16）；
`grep -rniE "agent|智能体"` 在交付文件中零命中。

### Q2-3　`models/metrics_stage2_eda.json` 瘦身

现在 469 KB，契约约定该类文件存"指标 + 超参"，应当轻量。
**要求**：保留汇总统计（逐图的关键数值、分布摘要），把逐行明细数组
（如果有逐小时序列、逐节点大表）移出 JSON——要么删掉（PNG 本身才是交付物），
要么落成 `data/eda_stats_stage2.parquet` 并在交接文档注明。
**验收**：`metrics_stage2_eda.json` ≤ 100 KB，且报告引用的每一个数都还能在里面查到。

### Q2-4　特征表体积与 LFS

`feat_spatiotemporal.parquet` 实体 80.6 MB，超过契约的 50 MB 上限，当前以 Git LFS 指针入库。
**两选一，在交接文档写明选了哪个：**

- **方案 a（推荐）**：改用 zstd 压缩重写 parquet（`to_parquet(compression="zstd")`），
  通常能降 30–40%；同时把 6 个 `audit_*` 布尔列和纯诊断列
  （`upwind_weight_sum_*`、`upwind_neighbor_count_*`）移到一个单独的小文件
  `data/audit_stage2.parquet`——建模用不到它们。若压到 ≤ 50 MB 就真实入库。
- **方案 b**：保留 LFS，但必须在 `docs/handover/02_eda_and_spatiotemporal.md` 写明
  LFS 配额风险（免费额度 1 GB 带宽/月，6 个人 clone 两次就超），
  并同步放一份到 OneDrive，链接写进 handover。

### Q2-5　越权改动的处理结果（不用你再做什么，说明一下）

你改的 `CONTRIBUTING.md` / `README.md` / `.gitignore` 属于 1 号专属文件。
内容我已逐条审阅：**三处修正全部吸收**（行数 78,102、datetime64[us]、
commit 示例里的数字以 JSON 为准），`.gitignore` 新增项也照收。
今后对这三个文件的改动，请在群里提出来，由我落笔——不是不信你的内容，
是文件所有权必须只有一个人，否则出了问题找不到责任人。

---

## 二、对 4号 的要求（5 项）

### Q4-1　分支改名

```bash
git checkout branch4
git branch -m branch4 feat/04-nominal-rf
git push origin -u feat/04-nominal-rf
git push origin --delete branch4
```

### Q4-2　加 receptor 过滤【目前最大的正确性问题】

`src/04_standard_and_reg_rf.py` 现在读入后**没有过滤节点**。
一旦 3 号交付全 panel 特征表，你会直接在 26 个节点上训练——
违反 `CONTRIBUTING.md` §3 第 7 条，而且 M1 就失去可比性。

在 `load_and_validate()` 读入并校验列之后、排序之前，加：

```python
if "role" in frame.columns:
    frame = frame[frame["role"].eq("receptor")].copy()
    if frame.empty:
        raise ValueError("no receptor rows found; check data/…parquet")

# 与契约登记值对账（数字来自 src/config.py 的 PANEL_CONTRACT）
n_train = int((frame["split"] == "train").sum())
expected = C.PANEL_CONTRACT["receptor_train_rows"]
if abs(n_train - expected) > 200:          # 容忍 03 离线筛选的正常损耗
    raise ValueError(
        f"receptor train rows = {n_train}, contract expects ~{expected}; "
        "if this is intentional, ask 1号 to register the change first")
```

同时在文件头部的 docstring 写明：**训练域 = 3 个受体节点**。

### Q4-3　补 M4 Boosting

现在只有 M2/M3（RF 两种权重）和回归，**没有提升法**。W7 明确讲 boosting，
这是课程覆盖的缺口。规格：

| 项 | 要求 |
|---|---|
| 模型 | `sklearn.ensemble.HistGradientBoostingClassifier`（优先，快、原生支持缺失值）；如不可用退回 `GradientBoostingClassifier` |
| 固定项 | `random_state=42`，`early_stopping=False`（保证可复现，不要让验证集参与训练） |
| 搜索空间 | `learning_rate ∈ {0.03, 0.06, 0.1}`，`max_iter ∈ {200, 400}`，`max_leaf_nodes ∈ {15, 31}`，`min_samples_leaf ∈ {20, 50}` |
| 评估 | 与 M2/M3 完全相同的 TimeSeriesSplit(5)、同一份特征、同一份指标函数 |
| 交付 | `models/boosting.joblib`、`models/metrics_stage4.json`（含 M2/M3/M4 三者）、`figures/fig_boosting_vs_rf.png` |

**重要**：如果 M4 赢了 M2/M3，照实报，不要因为"我们主线是 RF"就淡化；
如果输了，也照实报——boosting 对平滑噪声更敏感，这本身就是结论。

### Q4-4　03 没交付之前怎么干活

`features_final.parquet` 还不存在（3 号未提交），所以你的脚本现在跑不起来，这是正常的。
要求：

1. **冒烟测试**：临时用 `--data data/haze_panel_clean.parquet` 加 18 个基础特征
   （参考 `src/01_baseline_dt.py` 的 `BASE_FEATURES` + `wd_sin/wd_cos`），
   把整条流程（加载→排序→CV→训练→落盘）跑通一遍，确认没有 bug；
2. 冒烟结果**只用于验证代码能跑**，写进 commit message 注明 "smoke test on stage-1 features"，
   **不得写进 `metrics_stage4.json` 或报告**；
3. 3 号交付后，只改 `--data` 路径与特征清单重跑，正式数字以那时为准。

### Q4-5　其他

- 代码报错信息里的 "Agent 4 …"、"Agent 3 …" 改为 "4号"、"3号"；
- 模型文件名：`models/rf_nominal.joblib`、`models/rf_balanced.joblib`
  （契约 §4 已登记，不再用 `rf_nominal_balanced` 这种两个模型混在一起的名字）；
- 三个模型都要跑 3 个种子报均值 ± 标准差（§3 第 6 条）；
- 交 PR 前跑 `python src/01_verify_data.py --check`（16/16）和
  `python tools/check_prose.py reports/`。

---

## 三、全组通用（3/5/6/7 号开工前必读）

1. **训练域**：主实验只在 3 个受体节点上训练。走廊/源区节点只作特征输入。
2. **排序**：`TimeSeriesSplit` 之前必须按 `timestamp` 排序，面板落盘顺序不是时间序。
3. **口径**：`High_Risk_Recall` 是 union 口径 P(ŷ∈{2,3} | y∈{2,3})；
   逐类严格口径 0.5707 只是诊断项。做表必须写明用哪个。
4. **对手**：M0 纯持续性是必比的基线（M1 在高危召回上输给了它，0.7693 vs 0.7752）。
   你们每个模型的表里都要带上 M0。
5. **禁词**：交付物不得出现 agent / 智能体（verify 第 13 项门禁会拦）。
6. **数字溯源**：报告里每个数都要能在 `models/*.json` 里查到。

---

## 四、时间建议

| 事项 | 建议完成 | 依赖 |
|---|---|---|
| Q2-1 / Q2-2（2号） | 11-12 前 | 无，越早越好——3 号卡在你的特征表上 |
| Q2-3 / Q2-4（2号） | 11-13 前 | 与上项一起交付 |
| Q4-1 / Q4-2 / Q4-4（4号） | 11-12 前 | 无（冒烟测试用 01 面板） |
| Q4-3 正式跑（4号） | 3 号交付后 2 天内 | 03 |
| 3 号 | 2 号合并后 3 天内交付 features_final | 02 |

现在离 12-17 截止还有 5 周。**卡点在 02 → 03 这一段**，2 号的重跑请优先。

---

## 附：证据索引

| 发现 | 证据位置 |
|---|---|
| σ=15 下 73.31% 零权重、σ 内 0 对 | `feat/02-eda: models/metrics_stage2_features.json → spatial_training_diagnostics` |
| 高危召回两种口径 | `src/utils_metrics.py` + `models/metrics_stage1.json`（0.7693 / 0.5707） |
| M1 高危召回输给 M0 | `models/metrics_stage1.json → metrics_persistence_M0_test` |
| 报告数字差 89 行 | `models/cleaning_report.json` vs `reports/sec1_data_and_baseline.md`（已修） |
| agent 命名分布 | `feat/02-eda` 文件树 + 特征表列名 |
| 课程三条提交要求 | 课程 PDF §D（README.txt / 20MB / Rubric 随报告提交） |
