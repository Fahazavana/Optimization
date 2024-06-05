import matplotlib.pyplot as plt
import numpy as np 
import jax.numpy as jnp

def static_plot(history, f, figsize=(15, 5)):
    fig = plt.figure(figsize=figsize)
    xy = history['weights']
    zz = history['loss']
    
    xmax = jnp.max(jnp.abs(xy[:, 0]))
    ymax = jnp.max(jnp.abs(xy[:, 1]))

    x = np.linspace(-xmax - 0.5, xmax+ 0.5, 100)
    y = np.linspace(-ymax - 1, ymax + 0.5, 100)
    X, Y = np.meshgrid(x, y)
    Z = f(jnp.stack([X, Y])).reshape(X.shape)

    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    ax1.plot(xy[:, 0], xy[:, 1], zz, '.-r', zorder=5, alpha=0.9)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f')
    
    ax2 = fig.add_subplot(132)
    ax2.contourf(X, Y, Z, cmap='viridis')
    CS = ax2.contour(X, Y, Z, linewidths=1, linestyles='dashed', colors='black')
    ax2.clabel(CS, CS.levels, inline=True, fontsize=5)
    
    ax2.plot(xy[:, 0], xy[:, 1], '.--', color='red')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    ax3 = fig.add_subplot(133)
    ax3.plot(zz)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel(r'$f(x)$')
    
    return ax1, ax2, ax3, plt