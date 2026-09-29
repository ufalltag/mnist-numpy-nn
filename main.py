import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist


class AI:
    def __init__(self, input_size=28 * 28, hidden_size=128, output_size=10, lr=0.01):
        self.weight_input_hidden = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weight_hidden_output = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
        self.bias_output = np.zeros((1, output_size))
        self.lr = lr

    def forward_propagation(self, x):
        x = x.reshape(-1, self.weight_input_hidden.shape[0])
        self.x = x
        self.hidden_raw = x @ self.weight_input_hidden + self.bias_hidden
        self.hidden = self.relu(self.hidden_raw)
        self.output = self.softmax(self.hidden @ self.weight_hidden_output + self.bias_output)
        return self.output

    def back_propagation(self, y_true):
        batch_size = self.x.shape[0]
        d_output = (self.output - self.one_hot_encoding(y_true)) / batch_size
        d_weight_hidden_output = self.hidden.T @ d_output
        d_bias_output = d_output.sum(axis=0, keepdims=True)

        d_hidden = (d_output @ self.weight_hidden_output.T) * self.relu_derivative(self.hidden_raw)
        d_weight_input_hidden = self.x.T @ d_hidden
        d_bias_hidden = d_hidden.sum(axis=0, keepdims=True)

        self.weight_hidden_output -= self.lr * d_weight_hidden_output
        self.bias_output -= self.lr * d_bias_output
        self.weight_input_hidden -= self.lr * d_weight_input_hidden
        self.bias_hidden -= self.lr * d_bias_hidden

    def train(self, X, Y, epochs=5, batch_size=32):
        X = X.reshape(X.shape[0], -1)
        for epoch in range(epochs):
            indices = np.random.permutation(X.shape[0])
            total_loss = 0
            for start in range(0, X.shape[0], batch_size):
                batch = indices[start:start + batch_size]
                output = self.forward_propagation(X[batch])
                total_loss += self.cross_entropy(output, Y[batch]) * len(batch)
                self.back_propagation(Y[batch])
            print(f'Эпоха {epoch + 1}/{epochs}, loss: {total_loss / X.shape[0]:.4f}')

    def predict(self, x):
        return np.argmax(self.forward_propagation(x), axis=1)

    def accuracy(self, X, Y):
        return np.mean(self.predict(X.reshape(X.shape[0], -1)) == Y)

    def one_hot_encoding(self, y_true):
        y_true = np.atleast_1d(y_true)
        one_hot = np.zeros((y_true.size, 10))
        one_hot[np.arange(y_true.size), y_true] = 1
        return one_hot

    def cross_entropy(self, output, y_true):
        y_true = np.atleast_1d(y_true)
        return -np.mean(np.log(output[np.arange(y_true.size), y_true] + 1e-12))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exp = np.exp(x - x.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)


if __name__ == '__main__':
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train.astype(np.float32) / 255
    X_test = X_test.astype(np.float32) / 255

    ai = AI(lr=0.1)
    ai.train(X_train, Y_train, epochs=5, batch_size=32)
    print(f'Точность на тесте: {ai.accuracy(X_test, Y_test) * 100:.2f}%')

    fig, axes = plt.subplots(1, 5, figsize=(10, 3))
    for ax, image, label in zip(axes, X_test[:5], Y_test[:5]):
        ax.imshow(image, cmap='gray')
        ax.set_title(f'pred: {ai.predict(image)[0]}, true: {label}')
        ax.axis('off')
    plt.show()
