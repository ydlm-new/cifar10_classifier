import numpy as np
from train import Trainer
from utils.data_loader import load_cifar10
from utils.preprocessing import preprocess_data
from models.nn import ThreeLayerNN  

class HyperparameterTuner(Trainer):
    def tune(self):

        (X_train, y_train), (X_val, y_val), _ = load_cifar10()
        X_train, X_val, _ = preprocess_data(X_train, X_val, None)
        

        param_grid = {
            'learning_rate': [0.1, 0.01, 0.001],
            'hidden_dims': [[512, 256], [256, 128]],
            'reg_lambda': [0.1, 0.01, 0.001]
        }
        
        best_acc = 0
        best_params = {}
        batch_size = 64
        num_epochs = 5  
        

        for lr in param_grid['learning_rate']:
            for h_dims in param_grid['hidden_dims']:
                for reg in param_grid['reg_lambda']:
                    print(f"\nTesting lr={lr}, h_dims={h_dims}, reg={reg}")
                    

                    model = ThreeLayerNN(
                        input_dim=X_train.shape[1],
                        hidden_dims=h_dims,
                        num_classes=10,
                        reg_lambda=reg
                    )
                    
 
                    for epoch in range(num_epochs):
                        indices = np.random.permutation(len(X_train))
                        for i in range(0, len(X_train), batch_size):
                            X_batch = X_train[indices[i:i+batch_size]]
                            y_batch = y_train[indices[i:i+batch_size]]
                            
                            _ = model.compute_loss(X_batch, y_batch)
                            
                            grads = model.backward(X_batch, y_batch)
                            
                            model.update_params(grads, lr)
                    
                    val_acc = self.evaluate(model, X_val, y_val)
                    print(f"Val Acc: {val_acc:.4f}")
                    
                    if val_acc > best_acc:
                        best_acc = val_acc
                        best_params = {
                            'learning_rate': lr,
                            'hidden_dims': h_dims,
                            'reg_lambda': reg
                        }
        
        print("\nBest Parameters:", best_params)
        print(f"Best Val Acc: {best_acc:.4f}")

if __name__ == "__main__":
    tuner = HyperparameterTuner()
    tuner.tune()
