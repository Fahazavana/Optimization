import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm, trange


def agd(f, gradf, x, maxit, lr, threshold=1e-8):
    """
        Adaptive Gradient descent rules
    """
    stop =0
    threshold *= jnp.ones_like(x)
    history = {'loss': [], 'weights': []}
    pbar = trange(maxit, desc="Adaptive GD", ascii=' =')
    for k in pbar:
        y = f(x)
        dfx = gradf(x)
        for i in range(100):
            t= 1/(10**i)*lr
            if f(x - t * dfx) > f(x) - (t / 2)*jnp.linalg.norm(dfx, 2):
                lr = t
                
        history['weights'].append(x.copy())
        history['loss'].append(y)
        pbar.set_postfix(loss=f"{y:.3f}", grad = f"({dfx[0]:.3f}, {dfx[1]:.3f})", x=f"({x[0]:.3f}, {x[1]:.3f})")
        x -= lr*dfx
        if jnp.allclose(gradf(x), threshold):
            stop += 1
            if stop > 1:
                break
    history['loss'] = jnp.array(history['loss'])
    history['weights'] = jnp.array(history['weights'])
    return history