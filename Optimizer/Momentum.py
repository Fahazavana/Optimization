import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm, trange

def momentum(f, gradf, x0, maxit, lr, beta=0.9, threshold=1e-5):
    """
        Momentum
    """
    threshold *= jnp.ones_like(x0)
    history = {'loss': [], 'weights': []}
    pbar = trange(maxit, desc="Momentum GD", ascii=' =')
    
    d0 = gradf(x0)
    for _ in pbar:
        y = f(x0)
        dfx = gradf(x0)
        d1 = beta*d0 - (1- beta)*dfx
        
        history['weights'].append(x0.copy())
        history['loss'].append(y)
        x0 += lr * d1
        d0 = d1
        pbar.set_postfix(loss=f"{y:.3f}", grad = f"({dfx[0]:.3f}, {dfx[1]:.3f})", x=f"({x0[0]:.3f}, {x0[1]:.3f})")
        if jnp.allclose(gradf(x0), threshold):
            break
            
    history['loss'] = jnp.array(history['loss'])
    history['weights'] = jnp.array(history['weights'])
    return history