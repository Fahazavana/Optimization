import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm, trange

def adam(f, gradf, x0, maxit, eps=0.001, beta1=0.9, beta2=0.999, delta=1e-8, threshold=1e-5):
    threshold *= jnp.ones_like(x0)
    s = jnp.zeros_like(x0)
    r = jnp.zeros_like(x0)
    
    history = {'loss': [], 'weights': []}
    pbar = trange(maxit, desc="Adam", ascii=' =')
    
    for t in pbar:
        y = f(x0)
        dfx = gradf(x0)
        
        s = beta1 * s + (1 - beta1) * dfx
        r = beta2 * r + (1 - beta2) * dfx**2

        s_hat = s / (1 - beta1**(t + 1))
        r_hat = r / (1 - beta2**(t + 1))

        dtheta = -eps * s_hat / (jnp.sqrt(r_hat) + delta)
        
        history['weights'].append(x0)  # Convert to numpy for history storage
        history['loss'].append(y)

        pbar.set_postfix(loss=f"{y:.3f}", grad = f"({dfx[0]:.3f}, {dfx[1]:.3f})", x=f"({x0[0]:.3f}, {x0[1]:.3f})")
        x0 += dtheta
        if jnp.allclose(gradf(x0), threshold):
            break
            
    history['loss'] = np.array(history['loss'])
    history['weights'] = np.array(history['weights'])
    return history