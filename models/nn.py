import numpy as np
import os
from .layers import LinearLayer, ReLU, SoftmaxCrossEntropy

class ThreeLayerNN:
    def __init__(self, input_dim, hidden_dims, num_classes, reg_lambda=0.01):
        self.layers = [
            LinearLayer(input_dim, hidden_dims[0]),
            ReLU(),
            LinearLayer(hidden_dims[0], hidden_dims[1]),
            ReLU(),
            LinearLayer(hidden_dims[1], num_classes)
        ]
        self.loss_fn = SoftmaxCrossEntropy()
        self.reg_lambda = reg_lambda
        self.output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
        os.makedirs(self.output_dir, exist_ok=True)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def compute_loss(self, X, y):
        Z = self.forward(X)
        data_loss = self.loss_fn.forward(Z, y)
        reg_loss = 0.5 * self.reg_lambda * sum(
            np.sum(layer.W**2) for layer in self.layers if isinstance(layer, LinearLayer))
        return data_loss + reg_loss

    def backward(self, X, y):

        self.forward(X)
        

        dZ = self.loss_fn.backward(y)
        grads = []
        for layer in reversed(self.layers):
            if isinstance(layer, LinearLayer):
                dZ, grad_W, grad_b = layer.backward(dZ)
                grads.append((grad_W, grad_b))
            else:
                dZ = layer.backward(dZ)
        return grads[::-1]

    def update_params(self, grads, learning_rate):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, LinearLayer):
                grad_W, grad_b = grads[i//2]
                layer.W -= learning_rate * (grad_W + self.reg_lambda * layer.W)
                layer.b -= learning_rate * grad_b

    def save_weights(self, filename="model_weights.npz"):
        weights = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, LinearLayer):
                weights[f"W{i//2+1}"] = layer.W
                weights[f"b{i//2+1}"] = layer.b
        np.savez(os.path.join(self.output_dir, filename), **weights)

    def load_weights(self, filename="model_weights.npz"):
        data = np.load(os.path.join(self.output_dir, filename))
        for i, layer in enumerate(self.layers):
            if isinstance(layer, LinearLayer):
                layer.W = data[f"W{i//2+1}"]
                layer.b = data[f"b{i//2+1}"]
