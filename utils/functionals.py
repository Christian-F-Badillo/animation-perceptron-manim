import numpy as np


def sigmoid(Z: np.ndarray) -> np.ndarray:
    """Función de activación Sigmoide."""
    return 1 / (1 + np.exp(-Z))


def relu(Z: np.ndarray) -> np.ndarray:
    """Función de activación ReLU."""
    return np.maximum(0, Z)


def compute_cost(A3: np.ndarray, Y: np.ndarray) -> float:
    """
    Calcula la Entropía Cruzada Binaria.
    L(Y, A) = - (1/m) * sum(Y * log(A) + (1-Y) * log(1-A))
    """
    m = Y.shape[1]
    # Se añade un pequeño epsilon para evitar log(0)
    epsilon = 1e-15
    cost = -(1 / m) * np.sum(
        Y * np.log(A3 + epsilon) + (1 - Y) * np.log(1 - A3 + epsilon)
    )

    return float(np.squeeze(cost))
