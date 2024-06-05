import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange
from tqdm import tqdm

def agd(f, gradf, x, maxit, lr, threshold=1e-8):
    """
        Adaptive Gradient descent rules
    """
    threshold *= jnp.ones_like(x0)
    history = {'loss': [], 'weights': []}
    pbar = trange(maxit, desc="Adaptive GD")
    for k in pbar:
        y = f(x)
        dfx = gradf(x)
        for i in range(100):
            t= 1/(10**i)*lr
            if (f(x- t*dfx) > f(x) - (t/2)*jnp.linalg.norm(dfx, 2)):
                lr = t
                
        history['weights'].append(x.copy())
        history['loss'].append(y)
        pbar.set_postfix(loss=f"{y:.3f}", grad = f"({dfx[0]:.3f}, {dfx[1]:.3f})", x=f"({x0[0]:.3f}, {x0[1]:.3f})")
        if jnp.allclose(dfx, threshold):
            break
        x -= lr*dfx
    return history