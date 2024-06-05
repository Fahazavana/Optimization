import matplotlib.pyplot as plt
import numpy as np 
import jax.numpy as jnp

def animed_plot(history, f, fig):
    xy = history['weights']
    zz = history['loss']
    
    xmax = abs(max(xy[:,0]))
    ymax = abs(max(xy[:,1]))
    
    x = np.linspace(xmax-0.5, xmax+0.5, 100)
    y = np.linspace(ymax-0.5, ymax+0.5, 100)
    X, Y = np.meshgrid(x, y)
    Z = f(jnp.stack([X, Y]))
    Z = Z.reshape(X.shape)


    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.25)
    line3d = ax.plot(xy[:, 0], xy[:, 1], zz, '.r')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_zlabel('f')
    
    # Plot the contour
    ax = fig.add_subplot(122)
    ax.contourf(X, Y, Z, cmap='viridis')
    line2d = ax.plot(xy[:, 0], xy[:, 1], '.--', color='red')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('2D Rastrigin Function Plot')

    # update the line plot:
    def update(frame):
        line2d[0].set_xdata(xy[:frame, 0])
        line2d[0].set_ydata(xy[:frame, 1])
        line3d[0].set_xdata(xy[:frame, 0])
        line3d[0].set_ydata(xy[:frame, 1])
        line3d[0].set_3d_properties(zz[:frame])
        return line2d, line3d
    return animation.FuncAnimation(fig=fig, func=update, frames=30, interval=30)