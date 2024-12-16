from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
import numpy as np


class Bagging(BaseEstimator, ClassifierMixin):
    def __init__(self, base_model=DecisionTreeClassifier(), n_estimators=100, max_samples=0.8, max_features=0.5):
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features

    def bootstrap(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, int(
            self.max_samples * n_samples), replace=True)
        return X.iloc[indices], y.iloc[indices]

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.models_ = []
        for _ in range(self.n_estimators):
            X_sample, y_sample = self.bootstrap(X, y)
            model = self.base_model
            model.fit(X_sample, y_sample)
            self.models_.append(model)
        return self

    def predict(self, X):
        predictions = np.zeros((self.n_estimators, X.shape[0]))
        for i, model in enumerate(self.models_):
            predictions[i] = model.predict(X)

        # Majority vote
        final_predictions = np.apply_along_axis(
            lambda x: np.bincount(x.astype(int)).argmax(), axis=0, arr=predictions
        )
        return final_predictions

    def get_params(self, deep=True):
        """Return hyperparameters for tuning."""
        return {"base_model": self.base_model, "n_estimators": self.n_estimators, "max_samples": self.max_samples, "max_features": self.max_features}

    def set_params(self, **params):
        """Set hyperparameters for tuning."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
