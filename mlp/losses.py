import numpy as np


def cross_entropy_loss(y_pred, y_true):
    """
    Cross-entropy para saída softmax com rótulos one-hot.

    L = -1/N * sum(y_true * log(y_pred))

    Clampamos y_pred para evitar log(0).
    """
    n = y_true.shape[0]
    clipped = np.clip(y_pred, 1e-12, 1.0)
    return -np.sum(y_true * np.log(clipped)) / n


def cross_entropy_gradient(y_pred, y_true):
    """
    Gradiente da cross-entropy combinada com softmax.

    Quando usamos softmax + cross-entropy juntos, o gradiente simplifica para:
        dL/dz = (y_pred - y_true) / N

    Isso é uma das grandes conveniências matemáticas dessa combinação.
    """
    n = y_true.shape[0]
    return (y_pred - y_true) / n