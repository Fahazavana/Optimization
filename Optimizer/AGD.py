import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange
from tqdm import tqdm

def agd(f, gradf, x, maxit, lr, threshold=1e-8):
    """
        Adaptive Gradient descent rules
    """
    history = {'loss': [], 'weights': []}
    pbar = trange(maxit, desc="Run GD")
    for k in pbar:
        y = f(x)
        dfx = gradf(x)
        for i in range(100):
            t= 1/(10**i)*lr
            if (f(x- t*dfx) > f(x) - (t/2)*jnp.linalg.norm(dfx, 2)):
                lr = t
                
        history['weights'].append(x.copy())
        history['loss'].append(y)
        pbar.set_postfix(loss=f"{y:.3f}")
        if jnp.allclose(dfx, threshold):
            break
        x -= lr*dfx
    return history