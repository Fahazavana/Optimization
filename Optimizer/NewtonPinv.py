import numpy as jnp
from tqdm import tqdm, trange

def newtonpinv(f, gradf, hessf, x0, maxit, threshold=1e-8):
    """
        Newton's methods, where we replace the inverse of the Hessian by
        its Pseudo-inverse.
    """
    stop =0
    threshold *= jnp.ones_like(x0)
    history = {'weights': [x0], 'loss': [f(x0)]}
    pbar = trange(maxit, desc="Newton's Pinv.", ascii=' =')
    for k in pbar:
        
        
        y = f(x0)
               
        dfx = gradf(x0)
        hfx = hessf(x0)
        h = hfx + 1e-18 * jnp.eye(x0.size)
        
        x0 -= jnp.dot(jnp.linalg.pinv(h, hermitian=True), dfx)
        history['weights'].append(x0.copy())
        history['loss'].append(f(x0)) 
        pbar.set_postfix(loss=f"{y:.3f}", grad = f"({dfx[0]:.3f}, {dfx[1]:.3f})", x=f"({x0[0]:.3f}, {x0[1]:.3f})")
        if jnp.allclose(gradf(x0), threshold):
            stop+=1
            if stop>1:
                break
    history['loss'] = jnp.array(history['loss'])
    history['weights'] = jnp.array(history['weights'])
    return history