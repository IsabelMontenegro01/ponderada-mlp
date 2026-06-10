import numpy as np
from mlp.activations import get_activation, softmax
from mlp.losses import cross_entropy_loss, cross_entropy_gradient


class MLP:
    """
    Multi-Layer Perceptron implementado do zero com NumPy.

    Arquitetura: [input] -> [hidden1] -> ... -> [hiddenN] -> [output (softmax)]

    Parâmetros
    ----------
    layer_sizes : list[int]
        Tamanhos de cada camada, incluindo entrada e saída.
        Ex: [784, 256, 128, 10] cria 2 camadas ocultas.
    activation : str
        Função de ativação das camadas ocultas ('relu', 'sigmoid', 'tanh').
    """

    def __init__(self, layer_sizes, activation="relu"):
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1  # número de camadas com parâmetros
        self.act_fn, self.act_derivative = get_activation(activation)

        self.weights = []
        self.biases = []
        self._init_weights()

    def _init_weights(self):
        """
        Inicialização He (Kaiming) para ReLU:
            std = sqrt(2 / n_in)
        
        Inicializar com zeros causa simetria: todos os neurônios recebem
        o mesmo gradiente e aprendem a mesma coisa. Com He, quebramos essa
        simetria e a rede consegue aprender representações diversas.
        """
        for i in range(self.n_layers):
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]
            std = np.sqrt(2.0 / n_in)
            W = np.random.randn(n_in, n_out) * std
            b = np.zeros((1, n_out))
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X):
        """
        Forward pass: propaga X pela rede camada por camada.
        
        Armazena as pré-ativações (z) e pós-ativações (a) para uso
        no backpropagation.

        Retorna a distribuição de probabilidades (softmax) da última camada.
        """
        self._cache = {"a": [X], "z": []}

        a = X
        for i in range(self.n_layers - 1):
            z = a @ self.weights[i] + self.biases[i]
            a = self.act_fn(z)
            self._cache["z"].append(z)
            self._cache["a"].append(a)

        # Última camada: linear + softmax
        z_out = a @ self.weights[-1] + self.biases[-1]
        a_out = softmax(z_out)
        self._cache["z"].append(z_out)
        self._cache["a"].append(a_out)

        return a_out

    def backward(self, y_true):
        """
        Backpropagation: calcula gradientes de todos os pesos e biases
        usando a regra da cadeia, de trás para frente.

        Para softmax + cross-entropy, o gradiente da última camada é:
            delta = y_pred - y_true  (simplificação algébrica)

        Para camadas ocultas com ReLU:
            delta_l = (delta_{l+1} @ W_{l+1}^T) * relu'(z_l)
        """
        grads_W = [None] * self.n_layers
        grads_b = [None] * self.n_layers

        y_pred = self._cache["a"][-1]

        # Gradiente da saída (softmax + cross-entropy combinados)
        delta = cross_entropy_gradient(y_pred, y_true)

        for i in reversed(range(self.n_layers)):
            a_prev = self._cache["a"][i]

            # Gradientes dos pesos e bias desta camada
            grads_W[i] = a_prev.T @ delta
            grads_b[i] = np.sum(delta, axis=0, keepdims=True)

            # Propaga o gradiente para a camada anterior (exceto na entrada)
            if i > 0:
                delta = delta @ self.weights[i].T
                delta *= self.act_derivative(self._cache["z"][i - 1])

        return grads_W, grads_b

    def update_params(self, optimizer, grads_W, grads_b):
        """Delega a atualização dos parâmetros ao otimizador."""
        all_params = self.weights + self.biases
        all_grads = grads_W + grads_b
        optimizer.update(all_params, all_grads)

    def predict(self, X):
        """Retorna o índice da classe com maior probabilidade."""
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def compute_loss(self, X, y_true):
        probs = self.forward(X)
        return cross_entropy_loss(probs, y_true)

    def compute_accuracy(self, X, y_labels):
        """
        y_labels: vetor de inteiros (não one-hot).
        """
        preds = self.predict(X)
        return np.mean(preds == y_labels)

    def train(self, X_train, y_train_oh, y_train,
              X_val, y_val,
              optimizer, epochs=20, batch_size=128, verbose=True):
        """
        Loop de treinamento completo com mini-batches.

        Parâmetros
        ----------
        X_train      : (N, 784) float32
        y_train_oh   : (N, 10) one-hot
        y_train      : (N,) inteiros
        X_val        : conjunto de validação
        y_val        : rótulos de validação (inteiros)
        optimizer    : instância de SGD, SGDMomentum ou Adam
        epochs       : número de épocas
        batch_size   : tamanho do mini-batch
        """
        history = {"train_loss": [], "train_acc": [], "val_acc": []}
        n = X_train.shape[0]

        for epoch in range(1, epochs + 1):
            # Embaralha os dados a cada época para quebrar correlações
            idx = np.random.permutation(n)
            X_shuf = X_train[idx]
            y_shuf = y_train_oh[idx]

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                X_batch = X_shuf[start:start + batch_size]
                y_batch = y_shuf[start:start + batch_size]

                # Forward
                self.forward(X_batch)

                # Loss do batch (para monitoramento)
                probs = self._cache["a"][-1]
                epoch_loss += cross_entropy_loss(probs, y_batch)
                n_batches += 1

                # Backward
                grads_W, grads_b = self.backward(y_batch)

                # Atualiza pesos
                self.update_params(optimizer, grads_W, grads_b)

            avg_loss = epoch_loss / n_batches
            train_acc = self.compute_accuracy(X_train, y_train)
            val_acc = self.compute_accuracy(X_val, y_val)

            history["train_loss"].append(avg_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            if verbose:
                print(f"Época {epoch:3d}/{epochs} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Treino: {train_acc:.4f} | "
                      f"Val: {val_acc:.4f}")

        return history