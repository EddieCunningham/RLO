import jax.numpy as np
from util import normal_log_pdf
from jax import jit, vmap
from jax.scipy.special import logsumexp

@jit
def gaussian_mixture_log_pdf( x ):
    """ Log pdf for a basic mixture of gaussians
    Args:
        x - 2d batch of samples to evaluate
    Output:
        The logarithm of the pdf
    """

    # Must be in 2d
    assert x.shape[1] == 2

    # Define the mixture parameters
    components = 0.25*np.ones( 4 )
    mus = np.array( [ ( -4, -4 ), ( -4, 4 ), ( 4, -4 ), ( 4, 4 ) ] )
    sigmas = np.broadcast_to( np.eye( 2 )[None], ( 4, 2, 2 ) )
    sigma_invs = np.linalg.inv( sigmas )

    # Evaluate the log pdf for each componenet.  Vectorize over the gaussian parameters
    log_pdfs = vmap( normal_log_pdf, in_axes=( None, 0, 0 ), out_axes=1 )( x, mus, sigmas )
    log_pdfs += components[None,:]

    return logsumexp( log_pdfs, axis=1 )