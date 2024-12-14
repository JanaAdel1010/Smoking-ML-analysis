import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

class Custom_RandomForest:
    def __init__(self, n_estimators=100, max_features='auto', random_state=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []

    def _bootstrap_sample(self, X, y):
        return resample(X, y, random_state=self.random_state)

    def _select_features(self, X):
        if self.max_features == 'auto':
            max_features = int(np.sqrt(X.shape[1]))
        elif isinstance(self.max_features, int):
            max_features = self.max_features  
        else:
            max_features = X.shape[1]

        features = np.random.choice(X.shape[1], max_features, replace=False)
        return features

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.trees = []

        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap_sample(X, y)

            features = self._select_features(X_sample)

            tree = DecisionTreeClassifier(max_features=len(features), random_state=self.random_state)
            tree.fit(X_sample[:, features], y_sample)

            self.trees.append((tree, features))

    def predict(self, X):
        predictions = np.zeros((X.shape[0], self.n_estimators))
        for i, (tree, features) in enumerate(self.trees):
            predictions[:, i] = tree.predict(X[:, features])

        majority_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=predictions)
        return majority_vote
