import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm, trange

def gd(f, gradf, x0, maxit, lr, threshold=1e-5):
    """
        A simple vanilla Gradient Descent
    """
    threshold *= jnp.ones_like(x0)
    history = {'loss': [], 'weights': []}
    pbar = trange(maxit, desc="Vanilla Gradient Descent", ascii=' =')
    for _ in pbar:
        y = f(x0)
        dfx = gradf(x0)

        history['weights'].append(x0.copy())
        history['loss'].append(y)

        pbar.set_postfix(loss=f"{y:.3f}")
        x0 -= lr * dfx
        
        if jnp.allclose(gradf(x0), threshold):
            break
            
    history['loss'] = jnp.array(history['loss'])
    history['weights'] = jnp.array(history['weights'])
    return history
