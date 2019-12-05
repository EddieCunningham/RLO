import jax.numpy as np
from jax import jit

@jit
def normal_log_pdf( x, mu, sigma ):
    dx = x - mu
    return -0.5*np.einsum( 'bi,ij,bj->b', dx, np.linalg.inv( sigma ), dx ) - 0.5*np.linalg.slogdet( sigma )[1] - 0.5*np.log( 2*np.pi )