import numpy as np
class Custom_Boosting:
    def __init__(self, model, n):
        self.model = model
        self.n = n
        self.models = []
        self.model_weights = []

    def fit(self, X, y):
        self.models = []  
        self.model_weights = []  
        sample_weights = np.ones(len(y)) / len(y)  

        for _ in range(self.n):
            model = self.model()
            model.fit(X, y, sample_weight=sample_weights)
            predictions = model.predict(X)

            error = np.sum(sample_weights * (predictions != y)) / np.sum(sample_weights)

            if error == 0:
                self.models.append(model)
                self.model_weights.append(1)
                break

            model_weight = 0.5 * np.log((1 - error) / error)
            self.models.append(model)
            self.model_weights.append(model_weight)

            sample_weights = sample_weights * np.exp(-model_weight * y * predictions)
            sample_weights /= np.sum(sample_weights)

    def predict(self, X):
        weighted_predictions = np.zeros(len(X))
        for model, weight in zip(self.models, self.model_weights):
            weighted_predictions += weight * model.predict(X)
        return np.sign(weighted_predictions)
