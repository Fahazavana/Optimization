import numpy as jnp
from tqdm import tqdm, trange
def newton(f,gradf, hessf, x0, maxit, threshold=1e-8):
    """
        A simple vanilla Newton's method
    """
    stop = 0
    threshold *= jnp.ones_like(x0)
    history = {'weights': [], 'loss': []}
    pbar = trange(maxit, desc="Newton's", ascii=' =')
    for k in pbar:
        y = f(x0)
        history['weights'].append(x0.copy())
        history['loss'].append(y)
        dfx = gradf(x0)
        hfx = hessf(x0)
        
        h = hfx + (1e-8) * jnp.eye(x0.size)
        x0 = jnp.linalg.solve(h, jnp.dot(h, x0) - dfx)
        pbar.set_postfix(loss=f"{y:.3f}", grad = f"({dfx[0]:.3f}, {dfx[1]:.3f})", x=f"({x0[0]:.3f}, {x0[1]:.3f})")
        if jnp.allclose(gradf(x0), threshold):
            stop+=1
            if stop>1:
                break
    history['loss'] = jnp.array(history['loss'])
    history['weights'] = jnp.array(history['weights'])
    return history