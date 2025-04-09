import numpy as np
from models.nn import ThreeLayerNN
from utils.data_loader import load_cifar10
from utils.preprocessing import preprocess_data

def test():
    print("Loading test data...")
    (X_train, _), _, (X_test, y_test) = load_cifar10()

    X_train_flat = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255.0
    mean = np.mean(X_train_flat, axis=0)
    std = np.std(X_train_flat, axis=0)

    X_test_flat = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255.0
    X_test = (X_test_flat - mean) / (std + 1e-7)

    model = ThreeLayerNN(
        input_dim=X_test.shape[1],
        hidden_dims=[512, 256],
        num_classes=10
    )
    model.load_weights("best_model.npz")

    print("Evaluating...")
    Z = model.forward(X_test)
    preds = np.argmax(Z, axis=1)
    accuracy = np.mean(preds == y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    test()
