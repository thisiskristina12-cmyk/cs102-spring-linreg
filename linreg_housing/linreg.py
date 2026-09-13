import numpy as np
from sklearn.model_selection import train_test_split

class GDRegressor:
    def __init__(self, alpha=0.001, n_iter=100, progress=True):
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

def z_scaler(feature):
    feature = np.asarray(feature, dtype=float)
    return (feature - feature.mean(axis=0)) / feature.std(axis=0)

def min_max(feature):
    feature = np.asarray(feature, dtype=float)
    return (feature - feature.min(axis=0)) / (feature.max(axis=0) - feature.min(axis=0))

def rmse(y, y_hat):
    y = np.asarray(y, dtype=float).ravel()
    y_hat = np.asarray(y_hat, dtype=float).ravel()
    return np.sqrt(np.mean((y - y_hat) ** 2))

def r_squared(y, y_hat):
    y = np.asarray(y, dtype=float).ravel()
    y_hat = np.asarray(y_hat, dtype=float).ravel()
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot

def find_optimal_params(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=18
    )

    candidates_iter = [5000, 10000, 20000]
    candidates_alpha = [0.003, 0.005, 0.008, 0.01]

    best_iter = 5000
    best_alpha = 0.001
    best_rmse = float("inf")

    for n_iter in candidates_iter:
        for alpha in candidates_alpha:
            m = GDRegressor(alpha=alpha, n_iter=n_iter, progress=False)
            m.fit(X_train, y_train)
            y_pred = m.predict(X_test)

            r2 = r_squared(y_test, y_pred)
            err = rmse(y_test, y_pred)

            if r2 >= 0.49 and err <= 6.45 and err < best_rmse:
                best_rmse = err
                best_iter = n_iter
                best_alpha = alpha

    return best_iter, best_alpha