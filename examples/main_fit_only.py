import sys, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.constants as sp
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from pylab import *

#sys.path.append(os.path.abspath('../src'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.reaction_mechanism_generator import ReactionMechanismGenerator as rmg
from src.reaction_mechanism_generator import reaction_system


#from mech_plot import plot_mechanism
matplotlib.rcParams["font.family"] = "Times New Roman"
#plt.rcParams.update({"text.usetex": True})
matplotlib.rcParams['figure.figsize'] = 6,6/1.82
matplotlib.rcParams['axes.linewidth'] = 1.5
matplotlib.rcParams['figure.figsize'] = 8,8/1.62

def normalize_probs(probs):
    return probs / np.sum(probs)

def derivatives(t, concentration, k0, k1, k2):
    E, S, ES, P = concentration
    dEdt =  -k0*E*S + k1*ES + k2*ES
    dSdt =  -k0*E*S + k1*ES
    dESdt =  k0*E*S - (k1 + k2)*ES
    dPdt = k2*ES
    return [dEdt, dSdt, dESdt, dPdt]


# Seed the random number generators for reproducibility
np.random.seed(100)

# Constants
total_time = 6
Dt = 1e-2

# Rate constants
k0 = 1
k1 = 0.1
k2 = 1
ks = (k0, k1, k2)
print("true rate constants = ", *ks)

# Initial distribution and concentration
init_dist = np.array([0.40, 1.00, 0.00,0.00])
init_conc = init_dist*10
print("true initial concentrations = ", init_conc)
time = np.arange(0, total_time+Dt, Dt)

# Solve the system using solve_ivp
sol = solve_ivp(derivatives, [0, time[-1]], init_conc, t_eval=time, args=ks)

sol_time = sol.t
concentrations = sol.y

t_test = sol_time
x_test = concentrations.T
print("true final concentrations = ", x_test[-1])

data = np.concatenate((sol_time.reshape(-1, 1), concentrations.T), axis=1)
# Save the data to a file
np.savetxt('MM_data.txt', data, header='Time A B', fmt='%.6e')
