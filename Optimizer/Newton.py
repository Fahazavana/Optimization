import numpy as jnp
from tqdm.auto import tqdm, trange
def newton(f,gradf, hessf, x0, maxit, threshold=1e-8):
    """
        A simple vanilla Newton's method
    """
    threshold *= jnp.ones_like(x0)
    history = {'weights': [], 'loss': []}
    pbar = trange(maxit, desc="Newton's", ascii=' =')
    for k in pbar:
        y = f(x0)
        history['weights'].append(x0.copy())
        history['loss'].append(y)
        dfx0 = gradf(x0)
        hfx0 = hessf(x0)
        
        h = hfx0 + (1e-8) * jnp.eye(x0.size)
        x0 = jnp.linalg.solve(h, jnp.dot(h, x0) - dfx0)
        pbar.set_postfix(loss=f"{y:.3f}")
        if jnp.allclose(gradf(x0), threshold):
            break
    history['loss'] = jnp.array(history['loss'])
    history['weights'] = jnp.array(history['weights'])
    return history