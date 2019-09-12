import jax.numpy as np
from jax import jit

@jit
def normal_log_pdf( x, mu, sigma ):
    return -0.5*np.einsum( 'bi,ij,bj->b', x, np.linalg.inv( sigma ), x ) - 0.5*np.linalg.slogdet( sigma )[1] - 0.5*np.log( 2*np.pi )