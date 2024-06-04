import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange
from tqdm import tqdm

def newton(f,gradf, hessf, x0, maxit):
    """
        A simple vanilla Newton's method
    """
    history = {'weights': [], 'loss': []}
    pbar = trange(maxit, desc="Run Newton", ascii=' =')
    for k in pbar:
        y = f(x0)
        history['weights'].append(x0.copy())
        history['loss'].append(y)
        dfx0 = gradf(x0)
        hfx0 = hessf(x0)
        
        h = hfx0 + (1e-16) * jnp.eye(x0.size)
        x0 = jnp.linalg.solve(h, jnp.dot(h, x0) - dfx0)
        pbar.set_postfix(loss=f"{y:.3f}")
    return history