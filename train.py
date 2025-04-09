import numpy as np
import matplotlib.pyplot as plt
import os
from models.nn import ThreeLayerNN
from utils.data_loader import load_cifar10
from utils.preprocessing import preprocess_data

class Trainer:
    def __init__(self):
        self.output_dir = os.path.join(os.path.dirname(__file__), "outputs")
        os.makedirs(self.output_dir, exist_ok=True)

    def evaluate(self, model, X, y):
        Z = model.forward(X)
        preds = np.argmax(Z, axis=1)
        return np.mean(preds == y)

    def plot_metrics(self, train_losses, val_accuracies):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(val_accuracies, label='Val Accuracy', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.savefig(os.path.join(self.output_dir, "training_metrics.png"))
        plt.close()

    def visualize_weights(self, model, n_filters=16):
        W1 = model.layers[0].W.reshape(32, 32, 3, -1)
        plt.figure(figsize=(10, 10))
        for i in range(min(n_filters, W1.shape[3])):
            plt.subplot(4, 4, i+1)
            w_img = W1[:, :, :, i]
            w_img = (w_img - w_img.min()) / (w_img.max() - w_img.min())
            plt.imshow(w_img)
            plt.axis('off')
        plt.savefig(os.path.join(self.output_dir, "first_layer_weights.png"))
        plt.close()

    def train(self):
        print("Loading data...")
        (X_train, y_train), (X_val, y_val), _ = load_cifar10()
        X_train, X_val, _ = preprocess_data(X_train, X_val, None)

        model = ThreeLayerNN(
            input_dim=X_train.shape[1],
            hidden_dims=[512, 256],
            num_classes=10,
            reg_lambda=0.01
        )

        batch_size = 64
        learning_rate = 0.01
        num_epochs = 50
        train_losses, val_accuracies = [], []

        print("Starting training...")
        for epoch in range(num_epochs):
            if epoch % 10 == 0 and epoch > 0:
                learning_rate *= 0.5
                print(f"Learning rate decayed to {learning_rate:.4f}")

            # Shuffle data
            indices = np.random.permutation(len(X_train))
            for i in range(0, len(X_train), batch_size):
                batch_idx = indices[i:i+batch_size]
                X_batch, y_batch = X_train[batch_idx], y_train[batch_idx]

                Z = model.forward(X_batch)

                _ = model.loss_fn.forward(Z, y_batch)

                grads = model.backward(X_batch, y_batch)

                model.update_params(grads, learning_rate)

            train_loss = model.compute_loss(X_train, y_train)
            val_acc = self.evaluate(model, X_val, y_val)
            train_losses.append(train_loss)
            val_accuracies.append(val_acc)

            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")

            if val_acc == max(val_accuracies):
                model.save_weights("best_model.npz")

        self.plot_metrics(train_losses, val_accuracies)
        model.load_weights("best_model.npz")
        self.visualize_weights(model)
        print("Training completed. Visualizations saved to outputs/")

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
