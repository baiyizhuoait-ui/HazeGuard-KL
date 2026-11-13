# 04 模块交接：名义随机森林、平衡随机森林与提升法

本模块实现 M2、M3、M4，并用同一批受体样本、同一组 Top-25 特征和同一套指标比较三者。数值证据以 `models/metrics_stage4.json` 为准，英文报告章节见 `reports/sec2_6_models.md`。

## 1. 我产出了什么

训练域只包含 `role == receptor` 的 KL_City、Port_Klang、Sepang。进入本阶段的有效行数为 train 77,753、val 16,656、test 16,647，每行有 25 个有限数值特征和一个四级标签。代码按 `timestamp, station_id` 排序，并按唯一时间戳建立 5 折时间序列划分。

| 类别 | 文件 | 内容或形状 |
|---|---|---|
| 训练代码 | `src/04_rf_and_boosting.py` | M0、M2、M3、M4 的选参、三种子验证、一次测试和产物保存 |
| 实验代码 | `src/04_complexity_experiments.py` | 深度 2–25 的复杂度曲线，以及 3/6/12/24 小时时间窗敏感性 |
| M2 | `models/rf_nominal.joblib` | `RandomForestClassifier`，25 个输入特征，4 个输出类别，4,152,596 bytes |
| M3 | `models/rf_balanced.joblib` | `RandomForestClassifier`，25 个输入特征，4 个输出类别，10,159,600 bytes |
| M4 | `models/boosting.joblib` | `HistGradientBoostingClassifier`，25 个输入特征，4 个输出类别，1,866,896 bytes |
| 数值证据 | `models/metrics_stage4.json` | 36,034 bytes；含 12 组 RF、24 组 Boosting 搜索记录，三种子验证统计和 `test_evaluation_count=1` |
| 图 | `figures/fig_rf_complexity_curve.png` | 3511 × 1318 px；训练/验证 Macro-F1 与 log-loss 随深度变化 |
| 图 | `figures/fig_window_sensitivity.png` | 2309 × 1409 px；时间窗、验证 Macro-F1 与单样本延迟 |
| 图 | `figures/fig_boosting_vs_rf.png` | 3904 × 1276 px；M0/M2/M3/M4 固定测试集对比 |

三个模型的 SHA256 分别为：

- `rf_nominal.joblib`: `baa4cf77392bc9e1d98122b7f70cfc541652506a0509e416246d9516f0436874`
- `rf_balanced.joblib`: `f8f58f1227c128b33ab0d4e393ddcad95c968d35ff5f77c4e29c19cb6b4d147e`
- `boosting.joblib`: `3a8102172463125a16e89527eaa3e9a59dee3fbee2544bc42c1f53c9c8717f48`

## 2. 字段/接口说明

三个 joblib 文件直接保存已拟合的 sklearn 分类器，没有另外包装预处理器、特征名或标签映射。加载后 `classes_ == [0, 1, 2, 3]`，`n_features_in_ == 25`。`predict(X)` 返回形状 `(n,)` 的等级整数，`predict_proba(X)` 返回形状 `(n, 4)` 的概率矩阵，列顺序与 `classes_` 一致。高危概率可按 `proba[:, 2] + proba[:, 3]` 计算。

输入 `X` 必须是形状 `(n, 25)` 的有限数值矩阵。特征顺序不能由调用方猜测，必须读取 `metrics_stage4.json -> feature_names` 后按该顺序取列。训练入口会拒绝任一 Top-25 特征含 NaN 或无穷值；虽然 M4 算法本身能处理 NaN，本次保存模型的接口仍以“25 列全部有限”为契约。

M2 和 M3 的结构参数相同：100 棵树、`max_depth=14`、`min_samples_leaf=20`、`max_features=0.5`。两者唯一的建模差异是 M2 使用 `class_weight=None`，M3 使用 `class_weight='balanced_subsample'`。M4 使用 `learning_rate=0.06`、`max_iter=400`、`max_leaf_nodes=31`、`min_samples_leaf=20`，并固定 `random_state=42`、`early_stopping=False`。

产物使用 Python 3.11–3.13 项目的锁定依赖生成，其中 scikit-learn 为 1.9.0、joblib 为 1.5.3。joblib 反序列化能够执行对象构造逻辑，只应加载仓库中已核对摘要的可信文件。

## 3. 关键决策和理由

### 为什么 RF 固定 `max_depth=14`

12 组 RF 搜索中，不限深度候选的 M2 验证 Macro-F1 达到 0.7190，但用同一组结构参数训练 M3 后，压缩模型仍超过仓库规定的单文件 20 MiB 上限。不能只裁剪 M3，因为 M2/M3 对照要求除类别权重外保持相同。最终选择规则因此先排除不限深度候选，再在可交付候选中按验证 Macro-F1 选择。深度 14、100 棵树、叶节点最少 20 个样本的候选在该集合中最好，M2 验证 Macro-F1 为 0.7142，两个 RF 文件也都满足大小约束。

独立深度扫描显示 M3 的验证 Macro-F1 从深度 13 起进入平台区，深度 14 为 0.7097，深度 21 的最高值为 0.7136。继续加深只带来约 0.004 的验证差异，同时训练分数继续上升，序列化体积也更难控制。因此深度 14 是验证表现、对照公平性和仓库约束共同作用的结果，不应写成无约束搜索的绝对最优。

### 为什么 M4 用 HistGradientBoosting，而不是 AdaBoost

`docs/quality_notice_001.md` 的 M4 契约优先指定 `HistGradientBoostingClassifier`，原因是它在表格数据上训练较快，并原生支持缺失值。它可以在可接受时间内完成 24 组全参数网格和 5 折时间验证；`early_stopping=False` 也避免分类器再从训练数据内部切出一块隐式验证集。AdaBoost 会引入弱学习器及其学习率、迭代轮数的另一套选择问题，本阶段没有做这项额外对照。因此这里只能说明 HistGradientBoosting 符合当前规范，不能据此声称它普遍优于 AdaBoost。

### 模型选择纪律

超参数只由 train 内 5 折和固定 val 决定。固定方案后，种子 42/43/44 只用于 val 稳定性统计；所有选择落盘后才用种子 42 对 test 评估一批，JSON 明确记录 `test_evaluation_count=1`。M4 的验证 Macro-F1 为 0.7236，三模型最高，但 test 只有 0.7005。M3 在 test 的 Macro-F1 为 0.7018，高危召回为 0.8873，平均非对称代价为 0.6512，所以本阶段把 M3 作为风险目标下的首选模型，同时保留 M4 的验证集优势与较低 MAOE 结论。

## 4. 已知缺陷和坑

1. **joblib 不保存本项目的特征顺序。** 三个分类器的 `feature_names_in_` 均为空；直接把 DataFrame 转成数组很容易列错位。必须从 JSON 取 `feature_names`。
2. **不能直接传 API 原始字段。** 模型前面没有缺失值填补、滞后构造或 Top-25 选择器。调用方要先复现 02/03 阶段处理，并保证 25 列全部有限。
3. **内部时间 CV 不是完整嵌套验证。** Top-25 是 03 阶段用全部 train 标签选出的，本阶段没有在每个时间折内重做特征选择。CV 分数可能偏乐观，val/test 没有参与特征选择。
4. **M4 的三种子标准差为 0。** 当前 HistGradientBoosting 配置没有随机子采样，三个 `random_state` 运行得到相同结果。这不等于在不同时间段或不同节点上稳定。
5. **val 与 test 的类别构成不同。** 高危样本由 val 的 4,842 增至 test 的 6,848，模型排序发生变化。不能用 test 排名返回去改超参数或阈值。
6. **M3 的收益有代价。** 相对 M2，test 高危召回从 0.8094 提高到 0.8873，平均代价从 0.7700 降到 0.6512，但 MAOE 从 0.2813 升到 0.3003。类权重把决策边界推向更高风险等级，误报也随之增加。
7. **版本和安全性有约束。** 不匹配的 sklearn 版本可能无法稳定加载模型；来源不明的 joblib 不能加载。先核对上面的 SHA256。
8. **训练数据是模式产品上的教学实验。** 等级阈值是项目实验定义，不能解释成官方小时 AQI，也不能声称结果代表地面监测或个人暴露。

需要组长在总报告中保持三处口径一致：`max_depth=14` 是带文件大小约束的选择；“M3 是赢家”限定在本阶段的高危召回和代价目标；M4 三种子零标准差源于算法在当前配置下确定，不是额外的泛化证据。

## 5. 下一棒怎么用

在仓库根目录、锁定依赖已安装且 `rows` 已包含 03 阶段 Top-25 字段时，可运行：

```python
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

root = Path.cwd()
metrics = json.loads((root / "models/metrics_stage4.json").read_text(encoding="utf-8"))
features = metrics["feature_names"]

model_path = root / "models/rf_balanced.joblib"
expected_sha256 = "f8f58f1227c128b33ab0d4e393ddcad95c968d35ff5f77c4e29c19cb6b4d147e"
assert hashlib.sha256(model_path.read_bytes()).hexdigest() == expected_sha256

model = joblib.load(model_path)
X = rows.loc[:, features].to_numpy(dtype=float)
assert X.shape[1] == 25 and np.isfinite(X).all()

level = model.predict(X)
proba = model.predict_proba(X)
high_risk_probability = proba[:, 2] + proba[:, 3]
```

若要复查既有结果，优先读取 `models/metrics_stage4.json`，不要重新预测 test。若要独立重训，应先准备 03 阶段的 `data/features_final.parquet` 和 `models/selected_features.json`，再运行：

```sh
python src/04_rf_and_boosting.py
python src/04_complexity_experiments.py
```

两个入口默认使用仓库契约路径。独立实验应在干净副本中进行，避免覆盖已冻结的正式指标。

## 6. 我没做的事

- 没有用 test 调参、选阈值或选模型，也没有在补文档时重新预测 test。
- 没有做概率校准、ROC 阈值寻优、代价敏感训练、LOSO 泛化或特征解释；这些属于后续模块。
- 没有比较 AdaBoost、XGBoost、LightGBM 或其他新增算法，也没有增加项目依赖。
- 没有重做 03 阶段的 Top-25 特征选择，没有改 01–03 的代码、数据或指标。
- 没有把 M3 的高危召回优势写成所有指标都占优；M4 的验证集表现和 M2/M4 的较低 MAOE 均保留报告。
- 没有把四级标签、模型概率或树集成分散度解释成临床风险、官方 AQI 或经过校准的不确定性。
