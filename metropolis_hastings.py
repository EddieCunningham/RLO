import jax.numpy as np
import numpy as onp
import pandas as pd
from jax import random
from mcmc import metropolis_hastings_proposals
from bokeh.models import ColumnDataSource
from bokeh.layouts import widgetbox
from bokeh.models.widgets import Slider

class MetropolisHastings():

    def __init__( self,
                  key,
                  log_pdf,
                  acceptance_prob,
                  n_proposals,
                  x_dim,
                  x_0=None ):
        """ Create a metropolis hastings object
        Args:
            key             - jax random key
            log_pdf         - The target distribution function
            acceptance_prob - Acceptance probability
            n_proposals     - Number of proposals to generate
            x_dim           - The dimension of the samples
            x_0             - The first sample
        Output:
            None
        """
        # Create the parameter glyphs
        self.algorithm_parameter_glyphs

        self.key             = key
        self.log_pdf         = log_pdf
        self.acceptance_prob = acceptance_prob
        self.n_proposals     = n_proposals
        self.x_dim           = x_dim
        self.x_0             = x_0
        self.proposals = ColumnDataSource()

    def update_proposals( self ):
        """ Update the class's array of proposals
        Args:
            None
        Output:
            None
        """
        self.key, key = random.split( self.key, 2 )

        # Get some base noise to work with
        keys = random.split( key, 2 )
        proposal_noise = random.normal( keys[0], shape=( self.n_proposals, self.x_dim ) )
        acceptance_noise = random.uniform( keys[1], shape=( self.n_proposals, ), minval=0, maxval=1 )

        # Generate new proposals
        proposals = metropolis_hastings_proposals( self.log_pdf,
                                                   proposal_noise,
                                                   acceptance_noise,
                                                   self.acceptance_prob,
                                                   self.n_proposals,
                                                   self.x_dim,
                                                   self.x_0 )
        proposals_df = pd.DataFrame( proposals, columns=( 'x', 'y' ) )

        # Update the column data source with the new samples
        self.proposals.data = proposals_df

        return self.proposals

    def get_proposals( self ):
        """ Get the proposal column data source
        Args:
            None
        Output:
            The column data source with the proposals
        """
        return self.proposals

    @property
    def algorithm_parameter_glyphs( self ):
        """ Get the algorithm specific parameters that we want to include
        Args:
            None
        Output:
            The column data source with the proposals
        """
        if( hasattr( self, 'parameters_widgetbox' ) == False ):
            self.acceptance_slider = Slider( start=0, end=1000, value=1, step=1, title='Acceptance Probability' )
            self.proposal_slider   = Slider( start=0, end=10000, value=1000, step=10, title='Number of proposals' )

            # WHen we change either of the sliders, rerun the algorithm
            def callback( attr, old, new ):
                self.acceptance_prob = self.acceptance_slider.value
                self.n_proposals     = self.proposal_slider.value
                self.update_proposals()

            self.acceptance_slider.on_change( 'value', callback )
            self.proposal_slider.on_change( 'value', callback )

            # Store the sliders in a widgetbox
            self.parameters_widgetbox = widgetbox( self.proposal_slider, self.acceptance_slider )

        return self.parameters_widgetbox

    def clear( self ):
        """ Reset the object
        Args:
            None
        Output:
            None
        """
        self.proposals = ColumnDataSource()

