import jax.numpy as jnp


def rastrigin(x):
    """
        Rastrigin
    """
    r = x[0]**2 + x[1]**2
    cos = jnp.cos(2 * jnp.pi * x[0]) + jnp.cos(2 * jnp.pi * x[1])
    return 20 + r - 10*cos


def sphere(x):
    return jnp.sum(x**2)

def ackley(x):
    """
        Ackley
    """
    a = 20
    b = 0.2
    c = 2 * jnp.pi

    sum_squares = jnp.sum(x**2, axis=1)
    sum_cos = jnp.sum(jnp.cos(c * x), axis=1)

    A = -a * jnp.exp(-b * jnp.sqrt(0.5 * sum_squares))
    B = -jnp.exp(0.5 * sum_cos)

    return A + B + a + jnp.exp(1)