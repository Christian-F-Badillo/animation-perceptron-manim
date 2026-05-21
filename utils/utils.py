import numpy as np
from .functionals import sigmoid, relu
from typing import Dict


def forward_propagation(X: np.ndarray, parameters: dict) -> tuple:
    """
    Calcula la inferencia de la red y guarda los tensores intermedios (cache)
    necesarios para calcular las derivadas más tarde.
    """
    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]
    W3, b3 = parameters["W3"], parameters["b3"]

    # Capa 1
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)

    # Capa 2
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)

    # Capa 3
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)
    return A3, cache


def backward_propagation_solution(
    parameters: Dict, cache: tuple, X: np.ndarray, Y: np.ndarray
) -> dict:
    """
    Solución. Calcula los gradientes de la función de costo con respecto a los parámetros.
    """
    m = X.shape[1]  # Número de ejemplos
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y

    dW3 = (1 / m) * (dZ3 @ A2.T)

    db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = W3.T @ dZ3

    derivada_relu_2 = Z2 > 0
    dZ2 = dA2 * derivada_relu_2

    dW2 = (1 / m) * (dZ2 @ A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = W2.T @ dZ2

    derivada_relu_1 = Z1 > 0
    dZ1 = dA1 * derivada_relu_1

    dW1 = (1 / m) * (dZ1 @ X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}


def initialize_parameters(n_x: int, n_h1: int, n_h2: int, n_y: int) -> dict:
    """Inicialización de pesos usando He (para ReLU) y Xavier (para Sigmoide)."""
    np.random.seed(42)
    parameters = {
        "W1": np.random.randn(n_h1, n_x) * np.sqrt(2.0 / n_x),
        "b1": np.zeros((n_h1, 1)),
        "W2": np.random.randn(n_h2, n_h1) * np.sqrt(2.0 / n_h1),
        "b2": np.zeros((n_h2, 1)),
        "W3": np.random.randn(n_y, n_h2) * np.sqrt(1.0 / n_h2),
        "b3": np.zeros((n_y, 1)),
    }
    return parameters


def update_parameters(parameters: dict, grads: dict, learning_rate: float) -> dict:
    """Aplica Descenso de Gradiente Clásico."""
    for i in range(1, 4):
        parameters[f"W{i}"] -= learning_rate * grads[f"dW{i}"]
        parameters[f"b{i}"] -= learning_rate * grads[f"db{i}"]
    return parameters
