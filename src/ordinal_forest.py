"""把四级分类拆成三个“是否超过这一等级”的随机森林。"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted, validate_data


def cumulative_to_proba(cumulative):
    """修正三个累积概率，再还原 L0～L3 的概率。"""
    q = np.asarray(cumulative, dtype=float)
    if q.ndim != 2 or q.shape[1] != 3 or not np.isfinite(q).all():
        raise ValueError("cumulative probabilities must be finite with shape (n, 3)")
    if np.any((q < 0) | (q > 1)):
        raise ValueError("cumulative probabilities must be in [0, 1]")
    # 应满足 P(Y>0) >= P(Y>1) >= P(Y>2)，否则相减会产生负概率。
    q = np.minimum.accumulate(q, axis=1)
    p0 = 1 - q[:, 0]
    p1 = q[:, 0] - q[:, 1]
    p2 = q[:, 1] - q[:, 2]
    p3 = q[:, 2]
    p = np.column_stack((p0, p1, p2, p3))
    return p / p.sum(axis=1, keepdims=True)


class OrdinalRandomForest(ClassifierMixin, BaseEstimator):
    """保留 sklearn 接口，供保存模型和后面的看板直接调用。"""

    def __init__(self, n_estimators=100, max_depth=14, min_samples_leaf=20,
                 max_features=0.5, class_weight="balanced_subsample",
                 random_state=42, n_jobs=8):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        X, y = validate_data(self, X, y, dtype=np.float32)
        if not np.isin(y, [0, 1, 2, 3]).all():
            raise ValueError("ordinal labels must be integers in {0,1,2,3}")
        self.classes_ = np.arange(4)
        self.estimators_ = []
        for k in range(3):
            binary_y = (y > k).astype(np.int8)
            forest = RandomForestClassifier(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features, class_weight=self.class_weight,
                random_state=self.random_state, n_jobs=self.n_jobs,
            ).fit(X, binary_y)
            self.estimators_.append(forest)
        return self

    def predict_cumulative(self, X):
        check_is_fitted(self, "estimators_")
        X = validate_data(self, X, reset=False, dtype=np.float32)
        q = np.zeros((len(X), 3))
        for k, forest in enumerate(self.estimators_):
            # 早期时间折可能只有一个类别，不能固定取概率的第二列。
            positive = np.flatnonzero(forest.classes_ == 1)
            if len(positive):
                q[:, k] = forest.predict_proba(X)[:, positive[0]]
        return q

    def predict_proba(self, X):
        return cumulative_to_proba(self.predict_cumulative(X))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
