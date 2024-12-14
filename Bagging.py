import numpy as np
from sklearn.utils import resample
class Custom_Bagging:
    def __init__(self, base_model, n):
        self.base_model = base_model
        self.n = n
        self.models = []

    def fit(self, X, y):
        for _ in range(self.n):
            # Create a bootstrapped dataset
            X_resampled, y_resampled = resample(X, y, random_state=None)
            model = self.base_model()
            model.fit(X_resampled, y_resampled)
            self.models.append(model)

    def predict(self, X):
        predictions = np.zeros((len(X), self.n))
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X)
        
        # Majority voting (for classification)
        final_predictions = [np.bincount(predictions[i]).argmax() for i in range(len(X))]
        return np.array(final_predictions)