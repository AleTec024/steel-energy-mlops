# src/utils/seeds.py

"""
Utilidades para fijar semillas aleatorias y hacer reproducibles
los experimentos de entrenamiento.
"""

import os
import random
import numpy as np

DEFAULT_SEED = int(os.getenv("SEED", 42))


def set_global_seed(seed: int = DEFAULT_SEED) -> int:
    """
    Fija la semilla de los generadores de números aleatorios
    usados en el proyecto.

    Parameters
    ----------
    seed : int
        Valor de semilla a utilizar.

    Returns
    -------
    int
        La semilla finalmente utilizada (por si se quiere loguear en MLflow).
    """
    random.seed(seed)
    np.random.seed(seed)
    return seed
