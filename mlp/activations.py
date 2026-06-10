import numpy as np


def relu(z):
    """ReLU: f(z) = max(0, z)"""
    return np.maximum(0, z)


def relu_derivative(z):
    """Derivada da ReLU: 1 se z > 0, 0 caso contrário."""
    return (z > 0).astype(float)


def sigmoid(z):
    """Sigmoid: f(z) = 1 / (1 + e^-z). Usada como opcional."""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)


def tanh(z):
    return np.tanh(z)


def tanh_derivative(z):
    return 1 - np.tanh(z) ** 2


def softmax(z):
    """
    Softmax numericamente estável: subtrai o máximo antes de exponenciar.
    Converte logits em distribuição de probabilidade para a saída.
    """
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


ACTIVATIONS = {
    "relu": (relu, relu_derivative),
    "sigmoid": (sigmoid, sigmoid_derivative),
    "tanh": (tanh, tanh_derivative),
}


def get_activation(name):
    if name not in ACTIVATIONS:
        raise ValueError(f"Ativação '{name}' desconhecida. Opções: {list(ACTIVATIONS.keys())}")
    return ACTIVATIONS[name]