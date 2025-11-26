import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

class LogisticRegressionCustom:
    def __init__(self, lr=0.1, epochs=2000, class_weights=(1, 1)):
        self.lr = lr
        self.epochs = epochs
        self.class_weights = class_weights
        self.w = None
        self.b = None

    def sigmoid(self, z):
        # Ensure z is a numpy array
        z = np.asarray(z)
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, X, y):
        # Ensure X is a numpy array
        X = np.asarray(X)
        y = np.asarray(y)
        
        z = np.dot(X, self.w) + self.b
        w0, w1 = self.class_weights

        loss = (
            y * w1 * np.logaddexp(0, -z) +
            (1 - y) * w0 * (np.logaddexp(0, z))
        ).mean()

        return loss

    def compute_gradients(self, X, y):
        # Ensure X is a numpy array
        X = np.asarray(X)
        y = np.asarray(y)
        
        m = len(y)
        # Ensure z is an array, not a scalar
        z = np.dot(X, self.w) + self.b
        y_pred = self.sigmoid(z)

        w0, w1 = self.class_weights
        sample_weights = np.where(y == 1, w1, w0)

        dz = (y_pred - y) * sample_weights

        dw = (X.T @ dz) / m
        db = dz.mean()

        return dw, db

    def fit(self, X, y):
        # Ensure X is a 2D numpy array
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        y = np.asarray(y)
        
        # Initialize weights as numpy array
        self.w = np.zeros(X.shape[1], dtype=np.float64)
        self.b = 0.0

        for epoch in range(self.epochs):
            dw, db = self.compute_gradients(X, y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            if epoch % 200 == 0:
                loss = self.compute_loss(X, y)
                print(f"epoch {epoch}, loss={loss:.4f}")

        return self


    def predict(self, X):
        # Ensure X is a 2D numpy array
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        z = np.dot(X, self.w) + self.b
        return (self.sigmoid(z) >= 0.5).astype(int)

    def accuracy(self, X, y):
        preds = self.predict(X)
        return accuracy_score(y, preds)


    def confusion_matrix(self, X, y):
        preds = self.predict(X)
        return confusion_matrix(y, preds)
