from pysb.importers.boolean import model_from_boolean
from pysb.bng import run_ssa
import numpy as np
import matplotlib.pyplot as plt

modes = ['GSP', 'ROA', 'GA']

# For GSP, output is vs. time
# For ROA, output is vs. round
# For GA, output is vs. both time and round

for mode in modes:

    model = model_from_boolean('ABC_example.txt', mode=mode)

    n_sims = 1000
    tspan = None
    rounds = None
     
    if mode in ['GSP','GA']:
        tspan = np.linspace(0, 10, 101)
    if mode in ['ROA','GA']:
        rounds = 10
        output_step_interval = len(model.monomers)*2
        max_sim_steps = output_step_interval*rounds

    if tspan is not None:
        print(mode, '-- TIME')
        trajectories = []
        for n in range(n_sims):
            print(n)
            x = run_ssa(model, t_end=tspan[-1], n_steps=len(tspan)-1)
            trajectories.append(x)

        # transpose to aid in plotting
        a_traj = np.array([tr['A_True_obs'] for tr in trajectories]).T
        b_traj = np.array([tr['B_True_obs'] for tr in trajectories]).T
        c_traj = np.array([tr['C_True_obs'] for tr in trajectories]).T

        # plot the mean concentrations at each time point
        plt.figure('%s -- TIME' % mode)  
        plt.plot(tspan, a_traj.mean(axis=1), lw=2, label="A (%s)" % mode)
        plt.plot(tspan, b_traj.mean(axis=1), lw=2, label="B (%s)" % mode)
        plt.plot(tspan, c_traj.mean(axis=1), lw=2, label="C (%s)" % mode)
        plt.xlabel('Time (au)')
        plt.ylabel('Frequency')
        plt.legend(loc=0)
        
    if rounds is not None:
        print(mode, '-- ROUNDS')
        trajectories = []
        for n in range(n_sims):
            print(n)
            x = run_ssa(model, t_end=output_step_interval*100, n_steps=1,
                        output_step_interval=output_step_interval, 
                        max_sim_steps=max_sim_steps)
            trajectories.append(x)

        # transpose to aid in plotting
        a_traj = np.array([tr['A_True_obs'] for tr in trajectories]).T
        b_traj = np.array([tr['B_True_obs'] for tr in trajectories]).T
        c_traj = np.array([tr['C_True_obs'] for tr in trajectories]).T
        
        # plot the mean concentrations at each update round
        plt.figure('%s -- ROUNDS' % mode)
        plt.plot(range(rounds+1), a_traj.mean(axis=1), '-o', lw=2, ms=12, label="A (%s)" % mode)
        plt.plot(range(rounds+1), b_traj.mean(axis=1), '-o', lw=2, ms=12, label="B (%s)" % mode)
        plt.plot(range(rounds+1), c_traj.mean(axis=1), '-o', lw=2, ms=12, label="C (%s)" % mode)
        plt.xlabel('Round')
        plt.ylabel('Frequency')
        plt.legend(loc=0)

plt.show()
