import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange
from tqdm import tqdm

def newtonpinv(f, gradf, hessf, x0, maxit):
    """
        Newton's methods, where we replace the inverse of the Hessian by
        its Pseudo-inverse.
    """

    history = {'weights': [x0], 'loss': [f(x0)]}
    pbar = trange(maxit, desc="Run Newton Pinv.", ascii=' =')
    for k in pbar:
        
        
        y = f(x0)
               
        dfx0 = gradf(x0)
        hfx0 = hessf(x0)
        h = hfx0 + 1e-18 * jnp.eye(x0.size)
        x0 -= jnp.dot(jnp.linalg.pinv(h, hermitian=True), dfx0)
        pbar.set_postfix(loss=f"{y:.3f}")
        history['weights'].append(x0.copy())
        history['loss'].append(f(x0)) 
        
    return history