import numpy as np
class Custom_Bagging:
    def __init__(self, model, n):
        self.model = model
        self.n = n
        self.models = []

    def fit(self, X, y):
        self.models = [] 
        for _ in range(self.n):
            indices = np.random.choice(range(len(X)), size=len(X), replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            model = self.model()
            model.fit(X_sample, y_sample)
            self.models.append(model)

    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        majority_votes = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
        return majority_votes