# Necessary to make the paths work while we're in bokeh
import numpy as np
import os
import sys
sys.path.insert( 1, os.path.join( sys.path[0], '..' ) )

# Bokeh imports
from bokeh.plotting import figure
from bokeh.layouts import layout, column, row
from bokeh.models import ColumnDataSource, Div, HoverTool, CDSView, BooleanFilter
from bokeh.models.widgets import Slider, Select, TextInput
from bokeh.io import curdoc
from bokeh.models.widgets import Tabs, Panel
from bokeh.layouts import widgetbox

# Algorithm imports
from pdfs import gaussian_mixture_log_pdf
from metropolis_hastings import MetropolisHastings
from jax import random

def interactive( doc ):
    """ Function to do interactive plotting
    Args:
        doc - The bokeh document object
    Output:
        None
    """

    # Seed the random key
    key = random.PRNGKey( 0 )

    # Some constants for the moment.  Will clean up later
    height  = 800
    width   = 800
    x_range = ( -10, 10 )
    y_range = ( -10, 10 )
    tools = 'wheel_zoom,box_zoom,reset,pan,lasso_select,box_select'

    # Initialize the figure
    fig = figure( plot_height=height,
                  plot_width=width,
                  x_range=x_range,
                  y_range=y_range,
                  title='MCMC Samples',
                  tools=tools )

    # Make a slider to control the burn in value
    burn_in = Slider( start=1, end=10000, value=1, step=10, title='Burn In' )

    # Make a slider to control the sampling interval
    sample_interval = Slider( start=1, end=100, value=1, step=5, title='Sample Interval' )

    # Sample the data from a preset distribution
    mh = MetropolisHastings( key=key,
                             log_pdf=gaussian_mixture_log_pdf,
                             acceptance_prob=0.3,
                             n_proposals=10000,
                             x_dim=2,
                             x_0=None )
    samples_cds = mh.update_proposals()

    # Create a masking function that corresponds to the choice of burn in and sample interval values
    burn_in_mask = np.ones( mh.n_proposals ).astype( bool )
    burn_in_mask[:burn_in.value] = False

    sample_interval_mask = np.zeros( mh.n_proposals ).astype( bool )
    sample_interval_mask[::sample_interval.value] = True

    combined_mask = burn_in_mask&sample_interval_mask

    samples_view = CDSView( source=samples_cds, filters=[ BooleanFilter( combined_mask ) ] )

    # Create the circles glyph that will represent the samples
    sample = fig.circle( x='x',
                         y='y',
                         size=7,
                         source=samples_cds,
                         view=samples_view,
                         color='blue',
                         alpha=0.5 )

    # Create a callback function for the sliders
    def callback( attr, old, new ):
        # Re-create the masks again
        burn_in_mask = np.ones( mh.n_proposals ).astype( bool )
        burn_in_mask[:burn_in.value] = False

        sample_interval_mask = np.zeros( mh.n_proposals ).astype( bool )
        sample_interval_mask[::sample_interval.value] = True

        combined_mask = burn_in_mask&sample_interval_mask

        # Update the filter's mask
        samples_view.filters = [ BooleanFilter( combined_mask ) ]
        # samples_view.filters[0].booleans = combined_mask

    burn_in.on_change( 'value', callback )
    sample_interval.on_change( 'value', callback )

    # Create a slider to affect the acceptance probability
    doc.add_root( layout( row( fig, column( widgetbox( burn_in, sample_interval ), mh.algorithm_parameter_glyphs ) ) ) )

# if( __name__ == '__main__' ):
interactive( curdoc() )