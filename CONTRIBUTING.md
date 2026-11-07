# 协作规范

七个人（1 号带头，2~7 号各管一个模块）做同一个仓库，最容易出的问题不是算法写错，而是**别人不知道你在改什么**、
**两个人同时改同一个文件**、**下游读到的列名和上游写的对不上**。
这份文档就是为了把这些事提前定死。

开工之前请读完，尤其第 3 节和第 5 节。

---

## 1. 分工与文件归属

**一个文件只有一个负责人。** 别人只能读，不能改。发现问题就在群里说，或者提 Issue。

下面这张表**锁死**了，不再改。1 号到 7 号与模块号一一对应，产物文件名与课程要求逐字一致——
下游是按文件名读的，写错一个字母就读不到，而且不会报错。

| 号 | 职责 | 代码文件 | 交付产物 |
|---|---|---|---|
| **1号**（组长） | 数据工程、气候基线解耦与基线模型 | `src/config.py`、`src/01_data_pipeline.py`、`src/01_baseline_dt.py`、`src/01_verify_data.py`、`src/utils_metrics.py`、`tools/discover_nodes.py` | `data/haze_panel_clean.parquet`、`data/manifest.json`、`data/data_dictionary.json`、`models/baseline_dt.joblib`、`models/metrics_stage1.json`、`models/gate0_report.json`、`models/cleaning_report.json`、`figures/fig_baseline_dt_*.png` |
| **2号** | EDA 与时空季风传导特征 | `src/02_eda_analysis.py`、`src/02_spatiotemporal_features.py` | `data/feat_spatiotemporal.parquet`、`figures/fig_eda_diurnal_seasonal.png`、`figures/fig_eda_correlation_matrix.png`、`figures/fig_eda_class_imbalance.png`、`figures/fig_eda_wind_rose_pollution.png`、`figures/fig_eda_spatial_crosscorr.png` |
| **3号** | 模糊信息粒化与特征选择 | `src/03_fuzzy_granulation.py`、`src/03_feature_selection.py` | `data/features_final.parquet`、`models/selected_features.json`、`figures/fig_feature_ablation.png`、`figures/fig_fcm_prototypes.png` |
| **4号** | 多分类 RF、回归转分级与复杂度控制 | `src/04_standard_and_reg_rf.py`、`src/04_complexity_experiments.py` | `models/rf_nominal_balanced.joblib`、`models/rf_regressor.joblib`、`models/metrics_stage4.json`、`figures/fig_rf_complexity_curve.png`、`figures/fig_window_sensitivity.png` |
| **5号** | 有序 RF 与分位数区间预测 | `src/05_ordinal_rf.py`、`src/05_quantile_rf.py` | `models/ordinal_rf.joblib`、`models/quantile_rf.joblib`、`models/ordinal_probs_val_test.npz`、`models/metrics_stage5.json`、`figures/fig_ordinal_error_distribution.png`、`figures/fig_quantile_fan_chart.png` |
| **6号** | 代价敏感优化、SHAP 与泛化评估 | `src/06_cost_sensitive_gpso.py`、`src/06_shap_explainability.py`、`src/06_generalization_study.py` | `models/gpso_optimal_thresholds.json`、`models/all_models_benchmark.csv`、`models/loso_results.csv`、`models/seasonal_metrics.csv`、`figures/fig_shap_beeswarm.png`、`figures/fig_shap_dependence.png`、`figures/fig_shap_waterfall_haze_event.png`、`figures/fig_confusion_matrices_comparison.png` |
| **7号** | 交互系统与示范报告总成 | `app.py`、`src/07_compile_report.py` | `reports/HazeGuard_KL_Demonstration_Report.md`、`requirements.txt`（定稿）、`README.md`（定稿） |

每个人还要交两样东西，不写就算没做完：

- `reports/sec<章节号>_<主题>.md` —— 自己那几节的英文学术正文
- `docs/handover/NN_<主题>.md` —— 交接文档（见第 7 节）

### 几个特例，说清楚免得推诿

- **`utils_metrics.py` 只有 1 号能改。** 这是全组唯一的指标实现。谁发现自己需要它没有的功能，
  找 1 号加，**不要自己另写一个 `evaluate_predictions`**。两份口径一旦分家，
  最后合报告时数字一定打架，而且查不出是哪份错了。
- **`README.md` 和 `requirements.txt` 分两段管。** 前期由 1 号维护（项目说明、依赖）；
  最后一棒 7 号负责定稿（补运行说明、截图、最终依赖）。7 号动手前先在群里说一声。
- **`CONTRIBUTING.md`、`.gitignore`、`src/config.py` 永远只有 1 号能改。**
  改契约（列名、粒度、路径）必须先在这里登记，再改代码。

---

## 2. 分支怎么用

只有一条长期分支 `main`，加上每人一条短期分支。**分支已经建好了，直接切就行，不用自己想名字。**

| 分支 | 谁 | 模块 |
|---|---|---|
| `main` | 全体 | 集成分支，任何时刻拉下来都能跑通 |
| `feat/02-eda` | 2号 | EDA 与时空传导特征 |
| `feat/03-fig` | 3号 | 模糊信息粒化与特征选择 |
| `feat/04-nominal-rf` | 4号 | 名义 RF 与回归分级 |
| `feat/05-ordinal-rf` | 5号 | 有序 RF 与分位数 RF |
| `feat/06-cost-shap` | 6号 | 代价敏感优化与可解释性 |
| `feat/07-streamlit` | 7号 | 交互系统与报告 |

自己的活干完了就删掉自己的分支，别留着。

### 开工

```bash
git checkout main
git pull
git checkout feat/02-eda        # 换成自己那条
```

### 收工

```bash
git add -A
git commit -m "feat(02): EDA 五张图 + 时空传导特征

- 日内双峰与季风期对比：西南季风期 PM2.5 中位数比东北季风高 1.8 倍
- 上风向权重用 sigma=15km，静风时段按 qc_calm_wind 置零
- 新增 42 列，未改名未删列"

git push
```

然后到 GitHub 上开 Pull Request 合并到 `main`，按第 5 节清单自查一遍。

### 什么时候该合并

**做完就合，不要攒。** 一条分支留超过一周，`main` 已经往前走了，
合的时候冲突会成倍增加，而且你会发现自己基于一个过时的上游在干活。

只有 04 和 05 是真并行（两条都只读 `features_final.parquet`，写不同的文件），
它们可以同时开着；其余都是串行，前一条不合，后一条没法真正开始。

### 实际先后顺序

课程要求里把 02 和 03 画成并行，但 **03 要读 02 的 `feat_spatiotemporal.parquet`**，
所以真实顺序是：

```
01（已完成）→ 02 → 03 → 04 / 05（这两条可以真并行）→ 06 → 07
```

03 想提前开工是可以的：先写 `src/03_fuzzy_granulation.py`，用 01 的清洗表单独跑 FIG 特征，
等 02 合并后再做合并和消融。但不要写着"并行"结果去读一个还不存在的文件。

### 保护 main

仓库设置里勾上 `Settings → Branches → Add branch protection rule`，
对 `main` 打开：

- Require a pull request before merging
- Do not allow bypassing the above settings

这样没人能直接把代码推到 `main` 上，§5 的检查清单就不会被绕过。
（管理员可以临时关掉，但别养成习惯。）

---

## 3. 改别人东西的红线

1. **不许改上游产物的列名。** `pm2_5_anomaly` 就是 `pm2_5_anomaly`，
   不是 `pm25_anomaly`。下游按名字取数，改了就是静默丢列——程序不报错，指标莫名其妙变差。
2. **只能加列，不能删列、不能改名、不能改粒度。**
   `haze_panel_clean.parquet` 是"节点 × 小时"，下游不能重采样后覆盖它，要派生就新建文件。
3. **不许改别人的代码"顺手修一下"。** 哪怕只改一个变量名。发现了就告诉负责人。
4. **不许在报告里写没有出处数字。** 每个数都要能在 `models/*.json` 里找到。
5. **不许用 test 段做任何选择。** 调超参、挑特征、定阈值，统统在 val 上做。
   test 只在最后报一次。用了就是数据泄露，结论作废。
6. **随机种子固定 42。** 要比较两个模型，至少跑 3 个种子，报均值 ± 标准差。

---

## 4. 命名约定

### 文件

```
src/NN_<english_slug>.py          例：03_fuzzy_granulation.py
models/metrics_stage<N>.json      例：metrics_stage4.json
models/<model_key>.joblib         model_key: rf_nominal / rf_regressor / ordinal_rf / quantile_rf
models/<name>.npz                 概率矩阵，键名统一 val_probs / test_probs / val_y / test_y
figures/fig_<topic>.png           topic 用小写下划线
reports/sec<章节号>_<主题>.md
docs/handover/NN_<主题>.md
```

### 指标键名（**必须一字不差**）

```
Macro_F1
Balanced_Accuracy
High_Risk_Recall
MAOE
Severe_Jumping_Error_Rate
Avg_Asymmetric_Cost
n_samples
n_high_risk
```

不要写 `macro_f1`、`MacroF1`、`f1_macro`、`highrisk_recall` 这些变体。
最后合成报告时是按这套键名批量取数的，拼错不会报错，只会少一列。

### 其他口径

| 项 | 约定 |
|---|---|
| 等级编码 | 0 / 1 / 2 / 3（int8），阈值 (15, 35, 55) µg/m³ |
| 高危 | `y ∈ {2, 3}` |
| 风速 | 用 `wind_speed_ms`（m/s），不要用 `wind_speed_kmh` |
| 时间 | `Asia/Kuala_Lumpur`，`datetime64[ns]` |
| 空间单元 | `station_id` 指 CAMS 格点，**不是监测站**，别在图表里写 "station/measured" |
| 图内文字 | 英文（中文字体在服务器上容易变方块） |
| 报告正文 | 学术英文；`docs/` 和 `CONTRIBUTING.md` 用中文 |

---

## 5. 合并前检查清单

合并到 `main` 之前，逐条确认。少一条就先别合。

- [ ] `python src/01_verify_data.py --check` 仍然 **11/11 通过**
- [ ] 产物路径和 README 第"目录结构"一节写的**完全一致**（不是差不多）
- [ ] `models/metrics_*.json` 是脚本生成的，带 `generated_at` 和 `git_commit`
- [ ] `docs/handover/` 里自己的交接文档写完了，**"已知缺陷"和"我没做的事"两段不能空**
- [ ] `reports/sec*.md` 是英文学术写法，图表编号连续，每个数字能在 json 里查到
- [ ] `git diff main --stat` 里没有 `data/raw/`、没有绝对路径（`D:\`、`C:\Users`）、没有密钥
- [ ] 没有单个文件超过 50 MB
- [ ] 没有改别人负责的、已经存在的文件

### 什么能提交，什么不能

| 提交 ✅ | 不提交 ❌ |
|---|---|
| `src/*.py`、`requirements.txt` | `data/raw/`（93 MB 缓存，脚本能重新生成） |
| `reports/*.md`、`docs/*.md/json` | `__pycache__/`、`.ipynb_checkpoints/` |
| `figures/*.png`（单张 < 5 MB） | API key、token、`.env` |
| `models/*.json`、`*.joblib`（< 20 MB） | 任何绝对路径（换台电脑就废了） |
| `data/*.parquet`（单个 < 50 MB） | 超过 2 MB 的 notebook 输出 |

仓库总量控制 **200 MB** 以内，当前约 26 MB。

---

## 6. 提交信息怎么写

格式：

```
<type>(<模块>): <一句话>

<正文：做了什么、为什么这么做、关键数字>

<影响：动了哪个契约 / 已知缺陷>
```

`type` 用 `feat` / `fix` / `docs` / `data` / `refactor` / `chore`。

**每一次 commit 都要顺便把进展说清楚**，因为其他人是看 commit 和群消息知道你在干嘛的。
不要写"update"、"fix bug"、"改了一下"。

好的例子：

```
feat(03): FIG 粒化与 Top-25 特征选择

- FCM c=8, m=2.0，原型只在 train 上拟合，未触碰 val/test
- 解粒化重构误差 fig_recon_error_6h/12h 纳入候选特征
- 消融：Raw → +Lag → +Spatial → +FIG，val Macro-F1 0.662 → 0.701 → 0.734 → 0.762

影响：features_final.parquet 新增 12 列，未改名未删列
已知缺陷：c=8 时原型 6 只占 1.8% 样本，见 handover/03_fig.md 第 4 节
```

---

## 7. 交接文档

每一棒做完，负责人在 `docs/handover/NN_<主题>.md` 写一份，固定六段：

1. **我产出了什么** —— 文件、形状、SHA256（如果是数据表）
2. **字段/接口说明** —— 新增了哪些列，单位、语义、能不能空
3. **关键决策和理由** —— 为什么这么做，放弃了什么
4. **已知缺陷和坑** —— 下一棒一定会踩到的东西
5. **下一棒怎么用** —— 最小可运行代码片段
6. **我没做的事** —— 划清边界，别让人以为已经有了

**第 4 段和第 6 段必须写。** 只报成绩的交接文档等于没写。

---

## 8. 出问题了怎么办

| 情况 | 做法 |
|---|---|
| 自己的分支搞砸了 | `git reset --hard origin/main` 重来，别硬撑 |
| 上游数据有问题 | 告诉负责人，不要自己改 `data/` 下面的文件 |
| 两个人都改了同一个文件 | 后改的人负责解冲突，并且以后不要碰这个文件 |
| 已经合进 `main` 的代码有问题 | 开 `fix/xxx` 修，不要 revert 别人的东西 |
| 要改契约（列名、粒度、路径） | 先在群里说，组长确认后改 README 和所有引用 |

**永远不要 `git push --force` 到 `main`。**
