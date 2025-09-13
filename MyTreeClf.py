import numpy as np
import pandas as pd


class TreeNode:

    def __init__(
            self,
            feature: str = None,
            split_value: float = None,
            left_node: 'TreeNode' = None,
            right_node: 'TreeNode' = None,
            depth: int = 0,
            value: float = None
    ):
        self.feature = feature
        self.split_value = split_value
        self.left_node = left_node
        self.right_node = right_node
        self.depth = depth
        self.value = value

    def is_leave(self):
        return self.value is not None


class MyTreeClf:

    def __init__(
            self,
            max_depth: int = 5,
            min_samples_split: int = 2,
            max_leafs: int = 20,
            criterion: str = "entropy",
            bins=None
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.n = 0
        self.leafs_cnt = 0
        self.bins = bins
        self.criterion = criterion
        self.histogram = {}
        self.fi = {}
        self.root = None

    def _get_split_points(self, X: pd.DataFrame, feature: str) -> np.array:
        if self.bins is None:
            unique_values = np.sort(X[feature].values)
            return (unique_values[1:] + unique_values[:-1]) / 2
        if feature not in self.histogram:
            unique_values = X[feature].values
            self.histogram[feature] = np.histogram(unique_values, self.bins)[1][1:-1]
        return self.histogram[feature]

    def fit(self, X: pd.DataFrame, y: pd.Series):
        for feature in X.columns:
            self.fi[feature] = 0
        self.n = len(X)
        self.root = self._build_tree(0, X, y)

    def _get_feature_importance(self, y: pd.Series, left_idx: np.array, right_idx: np.array):
        left_y = y.loc[left_idx]
        right_y = y.loc[right_idx]
        start_criterion_value = self._get_criterion_value(y)
        left_criterion_value = self._get_criterion_value(left_y)
        right_criterion_value = self._get_criterion_value(right_y)

        return len(y) / self.n * (start_criterion_value - len(left_y) * left_criterion_value / len(y) - len(
            right_y) * right_criterion_value / len(y))

    def predict_proba(self, X: pd.DataFrame):
        result = []
        for _, row in X.iterrows():
            currNode: TreeNode = self.root
            while not currNode.is_leave():
                if row[currNode.feature] <= currNode.split_value:
                    currNode = currNode.left_node
                else:
                    currNode = currNode.right_node
            result.append(currNode.value)
        return np.array(result)

    def predict(self, X: pd.DataFrame):
        res = self.predict_proba(X)
        return (res >= 0.5).astype(int)

    def _build_tree(self, depth, X: pd.DataFrame, y: pd.Series) -> TreeNode:
        if self._get_entropy(y) == 0:
            return self._get_leave(y, depth)

        if depth == self.max_depth or len(y) < self.min_samples_split:
            return self._get_leave(y, depth)

        feature, split_point, _ = self.get_best_split(X, y)

        left_idx = X[X[feature] <= split_point].index
        right_idx = X[X[feature] > split_point].index

        potential_leaves = 0
        if len(left_idx) > 0:
            potential_leaves += 1
        if len(right_idx) > 0:
            potential_leaves += 1
        if depth != 0 and self.leafs_cnt + potential_leaves > self.max_leafs:
            return self._get_leave(y, depth)

        left_node = self._build_tree(depth + 1, X.loc[left_idx], y.loc[left_idx])
        right_node = self._build_tree(depth + 1, X.loc[right_idx], y.loc[right_idx])

        self.fi[feature] += self._get_feature_importance(y, left_idx, right_idx)

        return TreeNode(
            feature=feature,
            split_value=split_point,
            left_node=left_node,
            right_node=right_node,
            depth=depth
        )

    def _get_leave(self, y: pd.Series, depth: int) -> TreeNode:
        positive_class_count = (y == 1).sum()
        self.leafs_cnt += 1
        return TreeNode(
            value=positive_class_count / len(y),
            depth=depth
        )

    def get_best_split(self, X: pd.DataFrame, y: pd.Series):
        best_res = ("Feature name", -1, -1)
        for feature in X.columns:

            sorted_idx = X[feature].sort_values().index
            sorted_X: pd.DataFrame = X.loc[sorted_idx]
            sorted_y = y.loc[sorted_idx]

            split_points = self._get_split_points(X, feature)
            for split_point in split_points:
                y1 = sorted_y[sorted_X[feature] <= split_point]
                y2 = sorted_y[sorted_X[feature] > split_point]
                criterion_value = self._get_criterion_gain(y, y1, y2)
                if criterion_value > best_res[-1]:
                    best_res = (feature, split_point, criterion_value)
        return best_res

    def _get_criterion_gain(self, start_y: pd.Series, left_y: pd.Series, right_y: pd.Series):
        if self.criterion == "entropy":
            return self._get_information_gain(start_y, left_y, right_y)
        else:
            return self._get_gini_gain(start_y, left_y, right_y)

    def _get_criterion_value(self, y: pd.Series):
        if self.criterion == "entropy":
            return self._get_entropy(y)
        else:
            return self._get_gini_value(y)

    def _get_entropy(self, y: pd.Series):
        n = y.shape[0]
        p = y.value_counts() / n
        return -(np.sum(p * (np.log2(p + 1e-18))))

    def _get_information_gain(self, start_y: pd.Series, left_y: pd.Series, right_y: pd.Series):
        n = len(start_y)
        s0 = self._get_entropy(start_y)
        s1 = self._get_entropy(left_y)
        s2 = self._get_entropy(right_y)

        return s0 - len(left_y) / n * s1 - len(right_y) / n * s2

    def _get_gini_gain(self, start_y: pd.Series, left_y: pd.Series, right_y: pd.Series):
        n = len(start_y)
        curr_gini = self._get_gini_value(start_y)
        l_gini = self._get_gini_value(left_y)
        r_gini = self._get_gini_value(right_y)
        return curr_gini - len(left_y) * l_gini / n - len(right_y) * r_gini / n

    def _get_gini_value(self, y: pd.Series):
        n = len(y)
        p = y.value_counts() / n
        return 1 - np.sum(p ** 2)

    def print_tree(self):
        self._recursive_print_tree(self.root)

    def _recursive_print_tree(self, root: TreeNode, side: str = ""):
        if root is None:
            return
        if root.is_leave():
            print(f"{root.depth * ' '} {side} = {root.value}")
        else:
            print(f"{root.depth * ' '}{root.feature} | {root.split_value}")
            self._recursive_print_tree(root.left_node, "left")
            self._recursive_print_tree(root.right_node, "right")
