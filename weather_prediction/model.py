import numpy as np

class GDRegressor:
    def __init__(self, alpha: float = 0.001, n_iter: int = 1000, progress: bool = False):
        self.alpha = alpha
        self.n_iter = n_iter
        self.progress = progress
        self.coef_ = None
        self.intercept_ = None
        self.loss_history = []

    def fit(self, X_train, y_train):
        X = np.asarray(X_train, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y_train, dtype=float).ravel()
        m, p = X.shape

        self.coef_ = np.zeros(p)
        self.intercept_ = 0.0
        self.loss_history = []

        for _ in range(self.n_iter):
            y_pred = X @ self.coef_ + self.intercept_
            error = y_pred - y

            grad_w = (1.0 / m) * (X.T @ error)
            grad_b = (1.0 / m) * np.sum(error)

            self.coef_ -= self.alpha * grad_w
            self.intercept_ -= self.alpha * grad_b

            self.loss_history.append((1.0 / (2 * m)) * np.sum(error ** 2))

        return self

    def predict(self, X_test):
        X = np.asarray(X_test, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X @ self.coef_ + self.intercept_




