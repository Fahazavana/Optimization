import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange
from tqdm import tqdm


def gd(f, gradf, x0, maxit, lr, threshold=1e-8):
    """
        A simple vanilla Gradient Descent
    """
    history = {'loss': [], 'weights': []}
    pbar = trange(maxit, desc="Run GD", ascii=' =')
    for i in pbar:
        y = f(x0)
        dfx = gradf(x0)
        history['weights'].append(x0.copy())
        history['loss'].append(y)

        pbar.set_postfix(loss=f"{y:.3f}")
        x0 -= lr * dfx
        if jnp.allclose(gradf(x0), threshold):
            break

    return history
