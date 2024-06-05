import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm, trange

def rmsprop(f, gradf, x0, maxit, eps=1e-3, rho=1e-3, delta=1e-8, threshold=1e-5):
    threshold *= jnp.ones_like(x0)
    r =0
    history = {'loss': [], 'weights': []}
    pbar = trange(maxit, desc="RMSProp", ascii=' =')
    
    for t in pbar:
        y = f(x0)
        dfx = gradf(x0)
        r = rho*r + (1 - rho) * dfx**2
        dtheta = - (eps / (jnp.sqrt(r + delta))) * dfx
        history['weights'].append(x0)  # Convert to numpy for history storage
        history['loss'].append(y)
        x0 += dtheta
        
        pbar.set_postfix(loss=f"{y:.3f}", grad = f"({dfx[0]:.3f}, {dfx[1]:.3f})", x=f"({x0[0]:.3f}, {x0[1]:.3f})")
        if jnp.allclose(gradf(x0), threshold):
            break
            
    history['loss'] = np.array(history['loss'])
    history['weights'] = np.array(history['weights'])
    return history