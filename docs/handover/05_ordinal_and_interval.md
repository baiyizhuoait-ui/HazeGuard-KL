# 05 模块交接：回归分级、有序随机森林与树间区间

本模块按 `docs/spec_v3.md` 第 4 节实施。数值来源为
`models/metrics_stage5.json`；学习说明及结果表见 `reports/sec2_6_and_2_7.md`。

## 1. 上游与运行

- 本地基于 main `7dfe3e825d7aef851db60680eb9eb0a636b03e46`。
- 输入取自 03 分支 `50b6e63dfc936945e36999c797e85bf7f5ba4696`。
- 特征表 SHA256：`64285f40e81c72ab4872d282d52be68401078e8c33dec50134d93e7877d2f55b`。
- 仅使用受体节点 KL_City、Port_Klang、Sepang；25 列与上游顺序相同。
- 行数：train 77,753 / val 16,656 / test 16,647。
- 未修改其他成员代码、配置、指标或数据；本地从 03/04 分支取出的文件只是依赖与对照。
- 公共依赖沿用 `requirements.txt`，不增加 quantile-forest、SHAP 或其他算法库。

在已经整合 01～03 产物的仓库根目录运行：

```sh
python src/05_ordinal_rf.py --validate-only
python -m unittest discover -s tests -p test_stage5.py -v
python src/05_ordinal_rf.py --n-jobs 8
python tools/verify_stage5.py
```

`python src/05_reg_to_cls.py` 是同一完整流程的另一入口，**不必接着再运行一次**。
已有 `metrics_stage5.json` 或 `stage5_selection.json` 时拒绝覆盖重跑，避免反复查看 test。
核对结果用 `verify_stage5.py`，它读取已有预测，不重新预测 test。
代码和 JSON 的摘要检查兼容 Git 检出时的 LF/CRLF 换行转换，二进制产物仍要求逐字节一致。
若确需独立复现实验，应使用另一份干净副本，并将其作为重复实验记录，而不是试着挑更好 test。

## 2. 实现

### M5 回归转分级

`RandomForestRegressor` 学习 `target_pm25_3h`。
固定分档为 15 / 35 / 55 µg/m³，边界值归到较高档。
另提供验证集阈值校准对照：每个切点相对固定值偏移 {-4,-2,0,2,4}，共 125 组，
以 val Macro-F1 最大为主规则；相同分数优先 MAOE 小、偏移小的组合。
**不改真实标签的定义**，校准只改预测的切点。
切点由种子 42 的 val 决定后，在种子 43/44 和 test 上固定使用。

### M6 有序 RF

`src/ordinal_forest.py` 定义可序列化的 sklearn 风格分类器。
三个 RF 分别学习 Y>0、Y>1、Y>2，使用 `balanced_subsample`。
先记录原始概率的单调违反数，再按规范向下累积取最小值：

```text
q0* = q0
q1* = min(q0, q1)
q2* = min(q0, q1, q2)
p = [1-q0*, q0*-q1*, q1*-q2*, q2*]
```

四级概率归一化后，`argmax` 为默认等级；概率相同时取较低编号。
训练折只有一个二分类类别时也能输出正确的常数概率。
该修正保证概率有效，不保证提升所有指标，也不是概率校准。
相对名义 RF，结果包含累积分解和二分类类权重变化的共同影响，不能全归因于次序。

### 搜索与时间验证

3 组 RF 结构，包含 04 选出的结构，并分别降低深度、减小叶节点样本数。
每组运行 train 内 5 折 TimeSeriesSplit，按唯一 timestamp 分组，三个节点同刻一起划分。
按真实时间清除 `train timestamp + 3h >= fold validation start` 的训练行。
候选最终以 val Macro-F1、较低 MAOE、CV Macro-F1、较浅树依次选定。
固定超参数后，种子 42/43/44 只在 train 拟合，在 val 报均值及样本标准差。
最终保存种子 42。所有选择冻结并落盘后才产生一批 test 预测。

局限：上游 Top-25 用全部 train 标签筛选，内部 CV 不构成完整的嵌套验证。
val/test 未参与特征选择。本模块没有重做特征选择，保证与前面模型使用相同输入。

### 浓度区间

用 **M5 回归森林** 各树的浓度点预测取 5 / 50 / 95 百分位。
报告 PICP（真实浓度落在上下界之间的比例）、MPIW（平均带宽，µg/m³）。
这反映树之间的预测分散程度，**不是经过校准的 90% 预测区间，也不是分位数随机森林**。
示例图固定为 Sepang 最早 168 个测试小时，横轴为 t+3h，缺失小时留空，不选最好看的事件。

## 3. 给 6 号的概率接口

`models/ordinal_probs_val_test.npz` 可 `allow_pickle=False` 安全读取：

| 键 | 形状 | 说明 |
|---|---|---|
| `proba_val`, `proba_test` | (n,4) | 修正并归一化后，列顺序 L0/L1/L2/L3 |
| `y_val`, `y_test` | (n,) | 真实等级，0～3 |
| `ts_val`, `ts_test` | (n,) | 无时区的吉隆坡本地时间 t；不是目标时刻 t+3h |
| `station_id_val`, `station_id_test` | (n,) | 同刻多个节点，联合 timestamp 才能唯一定位 |

val n=16,656；test n=16,647。行序均为 timestamp、station_id。
**采用 spec_v3 的 `proba_val`，不是旧文档的 `val_probs`。**
高危概率为 `proba[:,2] + proba[:,3]`。阈值只能用 val 调整，test 用于冻结方案的最终报告。

```python
import numpy as np
z = np.load("models/ordinal_probs_val_test.npz", allow_pickle=False)
p = z["proba_val"]
y = z["y_val"]
high_risk_score = p[:, 2:].sum(axis=1)
```

## 4. 给 7 号的模型接口

先将仓库 `src` 加入模块查找路径，确保自定义分类器可被加载：

```python
import sys
from pathlib import Path
import joblib
import numpy as np
root = Path.cwd()  # 在仓库根目录运行
sys.path.insert(0, str(root / "src"))
ordinal = joblib.load(root / "models/ordinal_rf.joblib")
reg = joblib.load(root / "models/rf_regressor.joblib")
X = rows[ordinal.selected_features_].to_numpy(dtype=np.float32)
probability = ordinal.predict_proba(X)
level = probability.argmax(axis=1)
concentration = reg.predict(X)
fixed_level = np.digitize(concentration, reg.level_thresholds_fixed_)
tuned_level = np.digitize(concentration, reg.level_thresholds_val_tuned_)
tree_values = np.stack([t.predict(X) for t in reg.estimators_], axis=1)
band = np.quantile(tree_values, [0.05, 0.5, 0.95], axis=1).T
```

输入必须已有 25 个有限特征，不能直接传 API 原始字段。
模型是历史 CAMS 模式产品上的教学实验，区间和目标均不是地面监测真值。

## 5. 产物

- 代码：两份数字前缀入口 + `src/ordinal_forest.py`。
- 模型：`rf_regressor.joblib`、`ordinal_rf.joblib`，均要求 <20 MiB。
- 下游接口：`ordinal_probs_val_test.npz`。
- 证据：`metrics_stage5.json`、`stage5_selection.json`、`stage5_predictions.npz`。
- 图：`fig_ordinal_error_distribution.png`、`fig_interval_band.png`，300 dpi。
- 检查：`tests/test_stage5.py`、`tools/verify_stage5.py`。
- 本交接及 `reports/sec2_6_and_2_7.md`（数据、公式与自主撰写提纲）。

`stage5_predictions.npz` 额外保存 M5 预测、区间、M6 修正前累积概率和三级对照预测，
便于复核数字和重画图，无须重新预测 test。
模型、输入、代码及概率文件的 SHA256 均登记在 metrics JSON 中。

## 6. 规范中需要组长确认的三处文字

1. §4.2 的浓度区间把模型写成 M6；本实现使用能输出浓度的 M5，未改共享规范。
2. §4.4 的 M3 val 0.8744/0.2822 已满足 0.85/0.29 门槛，不能宣称本阶段首次满足。
3. §4.5 只有概率矩阵是 (n,4)，真实标签、时间戳都是一维。

这些处理均写入 `metrics_stage5.json -> spec_clarifications`，没有静默改契约。

## 7. 已完成结果速查

val 三种子均值：M5 校准切点的高危召回 0.8600、MAOE 0.2644，满足两道数值门槛。
M6 对应 0.8439 / 0.3028，未满足。

test 固定种子 42：

| 方案 | Macro-F1 | 高危召回 | MAOE | 跨两级误判 | 平均代价 |
|---|---:|---:|---:|---:|---:|
| M5 固定切点 | 0.7000 | 0.8413 | 0.2787 | 0.0100 | 0.7242 |
| M5 验证集校准切点 | 0.7078 | 0.8788 | 0.2873 | 0.0091 | 0.7083 |
| M6 有序 RF | 0.6839 | 0.8642 | 0.3169 | 0.0239 | 0.6616 |

本配置下，M5 校准方案相对 M6 的 F1、高危召回及跨级误判更好；M6 平均代价更低。
M6 的 Level 3 逐类召回为 0.8777，M5 为 0.7085；极端类别与总体序差存在取舍。
不据此声称所有有序分类方法都无用，也不将测试集结果用于继续改参数。

M6 单调违反率：val 3.48%、test 5.30%；修正后均为 0。
M5 树间区间 test PICP=0.7183、MPIW=20.7877 µg/m³，不支持“90% 真实浓度覆盖”的说法。

检查：4 项合成数据测试通过，基础数据 16/16 通过；保存结果复核通过，两张图已目视检查。
`check_prose.py` 没有硬性问题，但对中文数据笔记给了句长提示；它不验证原创性或课程许可。

模型文件各自小于 20 MiB，但两模型合计约 23 MB。完整交付压缩包供组内传递，
不等同于能直接上传 Moodle 的最终课程包；课程包可使用不含 joblib 的轻量包，模型按组长规范另存。

## 8. 可读性修改

删掉 `model_for`、`predict_for` 和 `regressor` 三层转接，训练处直接使用 sklearn 模型。
统计均值和标准差改成普通循环；四级概率分别写成 p0、p1、p2、p3；必要注释用简短中文解释。
完整训练入口的 `main()` 按读取、选参、验证、保存后测试、记录结果五步阅读。
部分重复的文字说明从训练入口移除，仍可在本交接查到。

未更改特征、候选参数、种子、时间折、类别权重、选参规则、切点或指标定义。
对照旧代码：合成数据上的训练、CV 和选参结果一致；全部 16,656 条验证预测等级一致，
概率最大绝对差约 5.6e-16（浮点累加误差）；已保存测试预测的后处理结果逐值一致。
没有重训真实模型或再次预测测试集。两个 joblib、概率/预测 NPZ、两张图和冻结记录的文件摘要均未改变。

`metrics_stage5.json` 保留原始训练时的 `code_sha256`，新增 `refactor` 字段登记当前代码和核对结果。
原版交付包保留；本次交付包文件名带 `_readable`。
