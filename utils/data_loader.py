import numpy as np
import pickle
import os

def load_cifar10(data_dir="data"):
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict[b'data'], dict[b'labels']

    train_data, train_labels = [], []
    for i in range(1, 6):
        data, labels = unpickle(os.path.join(data_dir, f"data_batch_{i}"))
        train_data.append(data)
        train_labels.extend(labels)

    test_data, test_labels = unpickle(os.path.join(data_dir, "test_batch"))

    X_train = np.concatenate(train_data)
    y_train = np.array(train_labels)
    X_test = np.array(test_data)
    y_test = np.array(test_labels)

    # Split train/val (45000/5000)
    X_val = X_train[45000:]
    y_val = y_train[45000:]
    X_train = X_train[:45000]
    y_train = y_train[:45000]

    # Reshape to (N, 32, 32, 3)
    def reshape_img(data):
        return data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    return (reshape_img(X_train), y_train), (reshape_img(X_val), y_val), (reshape_img(X_test), y_test)