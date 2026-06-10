import numpy as np


class SGD:
    """Stochastic Gradient Descent simples."""

    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def update(self, params, grads):
        """
        params e grads são listas paralelas de arrays NumPy.
        Atualização: w = w - lr * grad
        """
        for p, g in zip(params, grads):
            p -= self.lr * g


class SGDMomentum:
    """
    SGD com momento: acumula uma 'velocidade' na direção dos gradientes
    anteriores, o que ajuda a superar mínimos locais rasos e acelera
    a convergência em regiões de curvatura baixa.
    """

    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = None

    def update(self, params, grads):
        if self.velocity is None:
            self.velocity = [np.zeros_like(p) for p in params]

        for v, p, g in zip(self.velocity, params, grads):
            v *= self.momentum
            v -= self.lr * g
            p += v


class Adam:
    """
    Adam (Adaptive Moment Estimation): combina momento de 1ª e 2ª ordem.
    Mantém uma média móvel dos gradientes (m) e dos gradientes ao quadrado (v),
    ajustando a taxa de aprendizado individualmente para cada parâmetro.
    Geralmente converge mais rápido que o SGD puro.
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)

            # Correção de viés: nas primeiras iterações m e v tendem a zero
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)