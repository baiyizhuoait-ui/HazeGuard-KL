# 协作规范

七个人做同一个仓库，最容易出的问题不是算法写错，而是**别人不知道你在改什么**、
**两个人同时改同一个文件**、**下游读到的列名和上游写的对不上**。
这份文档就是为了把这些事提前定死。

开工之前请读完，尤其第 3 节和第 5 节。

---

## 1. 分工与文件归属

**一个文件只有一个负责人。** 别人只能读，不能改。发现问题就在群里说，或者提 Issue。

| 模块 | 负责人 | 负责的代码 | 负责的产物 |
|---|---|---|---|
| 01 数据工程与基线 | **组长** | `src/config.py`、`src/01_*.py`、`src/utils_metrics.py` | `data/haze_panel_clean.parquet`、`models/baseline_dt.joblib`、`models/metrics_baseline.json` |
| 02 EDA 与时空特征 | 成员 A | `src/02_*.py` | `data/feat_spatiotemporal.parquet`、`figures/fig_eda_*.png` |
| 03 粒化与特征选择 | 成员 E | `src/03_*.py` | `data/features_final.parquet`、`models/selected_features.json` |
| 04 名义 RF 与回归分级 | 成员 B | `src/04_*.py` | `models/rf_*.joblib`、`models/metrics_stage4.json` |
| 05 有序 RF 与分位数 RF | 成员 C | `src/05_*.py` | `models/ordinal_rf.joblib`、`models/ordinal_probs.npz` |
| 06 代价优化与可解释性 | 成员 D | `src/06_*.py` | `models/benchmark.csv`、`figures/fig_shap_*.png` |
| 07 交互系统与报告 | 成员 F | `app.py`、`src/07_compile_report.py` | `reports/final_report.md`、`README` 的截图 |

公共文件（`README.md`、`CONTRIBUTING.md`、`requirements.txt`、`.gitignore`）只有组长能改。

### 关于 `utils_metrics.py`

这是全组**唯一**的指标实现。任何人发现自己需要它没有的功能，**找组长加**，
不要在自己脚本里另写一个 `evaluate_predictions`。两份口径一旦分家，
最后合成报告时数字一定打架，而且很难查是哪一份错了。

---

## 2. 分支怎么用

只有一条长期分支 `main`，加上每人一条短期分支。

| 分支 | 谁用 | 规则 |
|---|---|---|
| `main` | 全体 | 任何时刻 checkout 下来都能跑通；只有过了第 5 节检查才合并 |
| `feat/02-eda` 这类 | 各自 | 从最新 `main` 切出，做完合并后删掉 |

命名：`feat/<模块号>-<简短英文>`，例如 `feat/03-fig`、`feat/06-gpso`。
修 bug 用 `fix/<简短描述>`。

```bash
git checkout main
git pull
git checkout -b feat/02-eda
# ...干活...
git push -u origin feat/02-eda
```

**不要在 `main` 上直接 commit。** 也不要把一条分支留超过一周不合并——
拖越久，合的时候冲突越多。

### 实际先后顺序

课程要求里把 02 和 03 画成并行，但 **03 要读 02 的 `feat_spatiotemporal.parquet`**，
所以真实顺序是：

```
01（已完成）→ 02 → 03 → 04 / 05（这两条可以真并行）→ 06 → 07
```

03 想提前开工是可以的：先写 `src/03_fuzzy_granulation.py`，用 01 的清洗表单独跑 FIG 特征，
等 02 合并后再做合并和消融。但不要写着"并行"结果去读一个还不存在的文件。

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
