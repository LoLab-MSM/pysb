from __future__ import print_function
import csv
import numpy as np
from earm.lopez_embedded import model
from pysb.tools.cytoscape_app.model_visualization_mons_norm import run_visualization

# type Bax cluster: 5400
# type Bak cluster: 4052

# type1 Bid cluster: 3905
# type2 Bid cluster: 2415

f = open('/Users/dionisio/PycharmProjects/tropical/examples/EARM/parameters_5000/pars_embedded_2415.txt')
data = csv.reader(f)
parames = [float(i[1]) for i in data]
tspan = np.linspace(0, 20160, 500)
run_visualization(model, tspan, parameters=parames, render_type='flux', save_video=False, verbose=True)

