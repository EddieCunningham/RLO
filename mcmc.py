import jax.numpy as np
from jax import random
from functools import partial
from util import normal_log_pdf
from jax import jit
import tensorflow as tf
import tensorflow_probability as tfp

def metropolis_hastings_proposals( unnormalized_log_pdf,
                                   proposal_noise,
                                   acceptance_noise,
                                   acceptance_prob=0.5,
                                   n_proposals=1000,
                                   x_dim=2,
                                   x_0=None ):
    """ Metropolis hastings algorithm with a gaussian proposal function
    Args:
        unnormalized_log_pdf - Function that computes the unnormalized
                               log_pdf for a batch of samples
        key                  - Jax random key
        acceptance_prob      - Acceptance probability
        n_proposals          - Number of times to propose a new point
        x_dim                - The dimension to work in
        x_0                  - The starting point
    Output:
        A batch of proposals with shape (n_proposals, x_dim)
    """
    pdf_type, log_pdf = unnormalized_log_pdf

    kernel = tfp.mcmc.HamiltonianMonteCarlo( target_log_prob_fn=log_pdf, step_size=0.5, num_leapfrog_steps=2 )

    states = tfp.mcmc.sample_chain( num_results=n_proposals,
                                    num_burnin_steps=0,
                                    current_state=tf.constant( [ -20.0, 20.0 ] ),
                                    kernel=kernel,
                                    trace_fn=None )
    return states.numpy()


    # states = tfp.mcmc.sample_chain(
    #     num_results=1000,
    #     num_burnin_steps=0,
    #     current_state=tf.zeros( 2 ),
    #     kernel=tfp.mcmc.HamiltonianMonteCarlo(
    #       target_log_prob_fn=unnormalized_log_pdf,
    #       step_size=0.5,
    #       num_leapfrog_steps=2),
    #     trace_fn=None)

    # print( states )
    # assert 0
    # return points

    # Create the proposals list and get the first point
    x_0 = np.zeros( x_dim ) if x_0 is None else x_0
    proposals = [ x_0 ]

    # Create the proposal distribution
    mu = np.zeros( x_dim )
    sigma = np.eye( x_dim )

    # Compute some stats for the first point
    current_x_likelihood_prob = unnormalized_log_pdf( x_0[None] )
    current_x_proposal_prob = normal_log_pdf( x_0[None], mu, sigma )

    # Run the metropolis hastings algorithm
    for i in range( n_proposals ):

        current_x = proposals[-1]

        # Sample the next proposal from a unit gaussian around the current point
        proposal_x = current_x + proposal_noise[i,:]

        # Compute the stats for the point
        proposal_x_likelihood_prob = unnormalized_log_pdf( proposal_x[None] )
        proposal_x_proposal_prob = normal_log_pdf( proposal_x[None], mu, sigma )

        # See if we should accept or reject
        pdf_ratio = proposal_x_likelihood_prob - current_x_likelihood_prob
        proposal_ratio = proposal_x_proposal_prob - current_x_proposal_prob

        # Accept or reject the new point
        if( pdf_ratio + proposal_ratio > 0 ):
            # Accept!
            current_x = proposal_x
            current_x_likelihood_prob = proposal_x_likelihood_prob
            current_x_proposal_prob = proposal_x_proposal_prob
        else:
            if( acceptance_noise[i] > acceptance_prob ):
                # Accept!
                current_x = proposal_x
                current_x_likelihood_prob = proposal_x_likelihood_prob
                current_x_proposal_prob = proposal_x_proposal_prob
            else:
                # Reject
                pass

        proposals.append( current_x )

    return proposals