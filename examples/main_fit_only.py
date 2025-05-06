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
    A, B, C, D = concentration
    dAdt =  -k0*A
    dBdt =  k0*A - k1*B
    dCdt =  k1*B - k2*C
    dDdt =  k2*C
    return [dAdt, dBdt, dCdt, dDdt]


# Seed the random number generators for reproducibility
np.random.seed(100)

# Constants
Num_particles = 1
total_time = 100000
Dt = 1e1

# Rate constants
k0 = 6.312e-5
k1 = 1.262e-4
k2 = 3.156e-4
ks = (k0, k1, k2)
print(*ks)
print("true rate constants = ", *ks)

# Initial distribution and concentration
init_dist = normalize_probs(np.array([1.00, 0.00, 0.00, 0.00]))
init_conc = init_dist * 2
time = np.arange(0, total_time+Dt, Dt)

# Solve the system using solve_ivp
sol = solve_ivp(derivatives, [0, time[-1]], init_conc, t_eval=time, args=ks)

sol_time = sol.t
concentrations = sol.y

t_test = sol_time
x_test = concentrations.T

data = np.concatenate((sol_time.reshape(-1, 1), concentrations.T), axis=1)
# Save the data to a file
np.savetxt("Linear_data.txt", data)
