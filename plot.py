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
from bokeh.models.widgets import Tabs, Panel, TextAreaInput, Dropdown
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
    height  = 600
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

    # Create the pdf contour
    def create_pdf( n_grid_points=200, pdf_kind='mixture' ):
        x = np.linspace( -25, 25, n_grid_points )
        y = np.linspace( -25, 25, n_grid_points )
        xx, yy = np.meshgrid( x, y )
        d = np.dstack( [ xx, yy ] ).reshape( ( -1, 2 ) )
        if( pdf_kind == 'mixture' ):
            log_pdfs = gaussian_mixture_log_pdf()
        else:
            assert 0
        pdfs = np.exp( log_pdfs( d ).numpy() ).reshape( ( n_grid_points, n_grid_points ) )
        return log_pdfs, pdfs
    log_pdfs, pdfs = create_pdf( n_grid_points=200, pdf_kind='mixture' )

    # Sample the data from a preset distribution
    mh = MetropolisHastings( key=key,
                             log_pdf=( 'gmm', log_pdfs ),
                             acceptance_prob=0.3,
                             n_proposals=2000,
                             x_dim=2,
                             x_0=None )
    samples_cds = mh.update_proposals()

    # must give a vector of image data for image parameter
    fig.image( image=[ pdfs ], x=-25, y=-25, dw=50, dh=50, palette="Viridis11" )

    # Make a slider to control the number of points
    n_points = Slider( start=1, end=mh.n_proposals, value=200, step=1, title='Number of points' )

    # Make a slider to control the burn in value
    burn_in = Slider( start=1, end=50, value=1, step=1, title='Burn In' )

    # Make a slider to control the sampling interval
    sample_interval = Slider( start=1, end=25, value=1, step=1, title='Sample Interval' )

    # Create a masking function that corresponds to the choice of burn in and sample interval values
    burn_in_mask = np.ones( mh.n_proposals ).astype( bool )
    burn_in_mask[:burn_in.value] = False

    n_points_mask = np.ones( mh.n_proposals ).astype( bool )
    n_points_mask[n_points.value:] = False

    sample_interval_mask = np.zeros( mh.n_proposals ).astype( bool )
    sample_interval_mask[::sample_interval.value] = True

    combined_mask = n_points_mask&burn_in_mask&sample_interval_mask

    samples_view = CDSView( source=samples_cds, filters=[ BooleanFilter( combined_mask ) ] )

    # Create the circles glyph that will represent the samples
    sample = fig.circle( x='x',
                         y='y',
                         size=7,
                         source=samples_cds,
                         view=samples_view,
                         color='red',
                         alpha=0.5 )

    n_points_question = 'What do you see when you increase the number of points?'
    n_points_answer = 'You should see all of the samples fit the pdf better'
    n_points_text = '%s\n\n%s'%( n_points_question, n_points_answer )

    burn_in_question = 'What do you see when you increase the burnin slider?'
    burn_in_answer = 'Small burn-in values should make the first few points farther from the target distribution'
    burn_in_text = '%s\n\n%s'%( burn_in_question, burn_in_answer )

    interval_question = 'How are points added to the different clusters when sample interval is low (1)?  What about when it is high (50)?'
    interval_answer = 'Small sample intervals should add points in one cluster at a time while large values will add points to both clusters'
    interval_text = '%s\n\n%s'%( interval_question, interval_answer )

    # Create textboxes for the user to write in
    n_points_text_box = TextAreaInput( value=n_points_text, rows=5, title='Effect of n_points' )
    burn_in_text_box = TextAreaInput( value=burn_in_text, rows=5, title='Effect of burn-in' )
    n_points_and_interval_text_box = TextAreaInput( value=interval_text, rows=5, title='Effect of sample interval' )

    # Create the pdf dropdown
    # pdf_dropdown = Dropdown( label="", button_type="warning", menu=menu )

    helper_1 = 'Number of points = 2000, Burn-in = 10, Sample interval = ?'
    helper_2 = 'Number of points = ?, Burn-in = 10, Sample interval = 15'
    helper_3 = 'Number of points = 200, Burn-in = ?, Sample interval = 1'
    helper_4 = 'Number of points = ?, Burn-in = 100, Sample interval = ?'

    helper_text = """<font size="4"><b>Suggested Configurations</b></font>
                     We recommend these configurations: \n\n%s\n\n%s\n\n%s\n\n%s"""%( helper_1, helper_2, helper_3, helper_4 )
    helper_div = Div( text=helper_text, width=400, height=100 )

    # Create helper code
    explanation_div = Div( text="""<font size="4"><b>MCMC explanation</b></font>
                       The goal of MCMC is to sample from a target probability distribution function.  In this tool, the red
                       points represent samples generated by the MCMC algorithm and the heatmap represents the target distribution.
                       When everything is working correctly, the red points should align with the heatmap""", width=400, height=125 )

    n_points_explanation = Div( text='<font size="4"><b>Number of Points Slider</b></font>This is the number of points that the algorithm will generate', width=400, height=80 )
    burn_in_explanation = Div( text='<font size="4"><b>Burn In Slider</b></font>This throws out the first N steps that are generated.  Ideally, points that are not representative of the distribution (due to bad initialization) will be discarded.', width=400, height=80 )
    sample_interval_explanation = Div( text='<font size="4"><b>Sample Interval Slider</b></font>Keep every N\'th point.  This parameter affects the position of consecutive points.  Try changing # of points with different fixed sample intervals.', width=400, height=80 )

    # Create a callback function for the sliders
    def callback( attr, old, new ):
        # Re-create the masks again
        n_points_mask = np.ones( mh.n_proposals ).astype( bool )
        n_points_mask[n_points.value:] = False

        burn_in_mask = np.ones( mh.n_proposals ).astype( bool )
        burn_in_mask[:burn_in.value] = False

        sample_interval_mask = np.zeros( mh.n_proposals ).astype( bool )
        sample_interval_mask[::sample_interval.value] = True

        combined_mask = n_points_mask&burn_in_mask&sample_interval_mask

        # Update the filter's mask
        samples_view.filters = [ BooleanFilter( combined_mask ) ]

    n_points.on_change( 'value', callback )
    burn_in.on_change( 'value', callback )
    sample_interval.on_change( 'value', callback )

    # Create a slider to affect the acceptance probability
    doc.add_root( layout( row( column( fig, column( row( n_points_text_box, burn_in_text_box ), n_points_and_interval_text_box ) ),
        column( explanation_div, helper_div, n_points_explanation, burn_in_explanation, sample_interval_explanation, widgetbox( n_points, burn_in, sample_interval ) ) ) ) )

interactive( curdoc() )