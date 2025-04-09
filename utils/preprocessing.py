import numpy as np

def preprocess_data(X_train, X_val, X_test):
    def normalize(X, mean, std):
        return (X - mean) / (std + 1e-7)

    # Compute stats on training set
    X_train_flat = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255.0
    mean = np.mean(X_train_flat, axis=0)
    std = np.std(X_train_flat, axis=0)

    def process(X):
        if X is None:
            return None
        X_flat = X.reshape(X.shape[0], -1).astype('float32') / 255.0
        return normalize(X_flat, mean, std)

    return process(X_train), process(X_val), process(X_test)