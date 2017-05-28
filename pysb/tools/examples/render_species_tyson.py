from pysb.examples.tyson_oscillator import model
from pysb.tools.cytoscape_app.model_visualization import run_visualization

run_visualization(model, render_type='species')
