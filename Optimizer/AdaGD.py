import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm, trange

def adagrad(f, gradf, x0, maxit, eps=1e-3, delta=1e-8, threshold=1e-5):
    threshold *= jnp.ones_like(x0)
    r =0
    history = {'loss': [], 'weights': []}
    pbar = trange(maxit, desc="AdaGrad", ascii=' =')
    
    for t in pbar:
        y = f(x0)
        dfx = gradf(x0)
        
        r += dfx**2

        dtheta = - (eps / (jnp.sqrt(r) + delta)) * dfx
        
        history['weights'].append(x0)  # Convert to numpy for history storage
        history['loss'].append(y)
        pbar.set_postfix(loss=f"{y:.3f}")
                    
        x0 += dtheta
        
        if jnp.allclose(gradf(x0), threshold):
            break
            
    history['loss'] = np.array(history['loss'])
    history['weights'] = np.array(history['weights'])
    return history