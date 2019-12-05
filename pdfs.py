import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np

def gaussian_mixture_log_pdf():
    mix = 0.5
    bimix_gauss = tfd.Mixture(
      cat=tfd.Categorical( probs=[ mix, 1.-mix ] ),
      components=[
        tfd.MultivariateNormalDiag( loc=-1.5*tf.ones( 2 ), scale_diag=tf.ones( 2 ) ),
        tfd.MultivariateNormalDiag( loc=1.5*tf.ones( 2 ), scale_diag=tf.ones( 2 ) ),
    ] )
    log_pdf = lambda x: bimix_gauss.log_prob( x )
    return log_pdf

def unit_gaussian_log_pdf():
    dist = tfd.MultivariateNormalDiag( loc=tf.zeros( 2 ), scale_diag=tf.ones( 2 ) ),
    log_pdf = lambda x: dist.log_prob( x )
    return log_pdf