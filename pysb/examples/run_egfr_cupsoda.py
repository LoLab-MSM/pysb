from chen_sorger_2009_egfr_simplified.erbb_exec import model
from pysb_cupsoda import *
import numpy as np
import matplotlib.pyplot as plt

observables = ['obsAKTPP', 'obsErbB1_P_CE', 'obsERKPP']
colors = ['b', 'g', 'r']

expdata = np.load('/Users/lopezlab/git/egfr/chen_sorger_2009_egfr_simplified/experimental_data/experimental_data_A431_highEGF_unnorm.npy')
exptimepts = [0, 149, 299, 449, 599, 899, 1799, 2699, 3599, 7199]

tspan = np.linspace(0, 8000, 8001)

# Sub best-fit parameters
import pickle
with open('/Users/lopezlab/git/egfr/chen_sorger_2009_egfr_simplified/calibration_files/A431_highEGF_erbb_simplified_fitted_values_no_ERKPP_fit.p', 'rb') as handle:
     fittedparams = pickle.loads(handle.read())
for p in model.parameters:
    if p.name in fittedparams:
        p.value = fittedparams[p.name]

#####
# from pysb.integrate import odesolve
# x = odesolve(model, tspan, verbose=True)
# for i in range(len(observables)):
#     plt.plot(tspan, x[observables[i]], lw=3, label=observables[i], color=colors[i])
# #     plt.plot(exptimepts, expdata[:,i], '*', lw=0, ms=12, mew=0, mfc=colors[i])
# plt.legend(loc=0)
# plt.ylim(ymin=1)
# plt.ylabel('molecules')
# plt.xlabel('time')
# plt.show()
# quit()
#####

set_cupsoda_path("/Users/lopezlab/cupSODA") #FIXME: should search for cupSODA in standard locations
solver = CupsodaSolver(model, tspan, atol=1e-12, rtol=1e-6, max_steps=20000, verbose=True)

n_sims = 1

# Rate constants
param_values = np.ones((n_sims, len(model.parameters_rules())))
for i in range(len(param_values)):
    for j in range(len(param_values[i])):
        param_values[i][j] *= model.parameters_rules()[j].value

# Initial concentrations
y0 = np.zeros((n_sims, len(model.species)))
for i in range(len(y0)):
    for ic in model.initial_conditions:
        for j in range(len(y0[i])):
            if str(ic[0]) == str(model.species[j]):
                y0[i][j] = ic[1].value
                break

solver.run(param_values, y0) #, outdir=os.path.join(outdir,'NSAMPLES_'+str(n_samples))) #obs_species_only=False, load_conc_data=False)
        
