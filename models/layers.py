import numpy as np

class LinearLayer:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)
        self.b = np.zeros((1, output_dim))
        self.X = None

    def forward(self, X):
        self.X = X
        return np.dot(X, self.W) + self.b

    def backward(self, dZ):
        grad_W = np.dot(self.X.T, dZ)
        grad_b = np.sum(dZ, axis=0, keepdims=True)
        dX = np.dot(dZ, self.W.T)
        return dX, grad_W, grad_b

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, Z):
        self.mask = (Z > 0)
        return Z * self.mask

    def backward(self, dA):
        return dA * self.mask

class SoftmaxCrossEntropy:
    def __init__(self):
        self.probs = None

    def forward(self, Z, y):
    
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))  
        self.probs = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)  
        m = y.shape[0]  
        
        loss = -np.sum(np.log(self.probs[np.arange(m), y])) / m
        return loss

    def backward(self, y):
       
        if self.probs is None:
            raise ValueError("Forward pass must be called before backward pass.")
        
        m = y.shape[0]
        dZ = self.probs.copy() 
        dZ[np.arange(m), y] -= 1  
        return dZ / m  
