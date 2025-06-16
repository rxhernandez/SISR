import sys, os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib

from SISR.reaction_mechanism_generator import ReactionMechanismGenerator as rmg, reaction_system

#from mech_plot import plot_mechanism
matplotlib.rcParams["font.family"] = "Times New Roman"
#plt.rcParams.update({"text.usetex": True})
matplotlib.rcParams['figure.figsize'] = 6,6/1.82
matplotlib.rcParams['axes.linewidth'] = 1.5
matplotlib.rcParams['figure.figsize'] = 8,8/1.62

# Create a function that imports the file "Sequential_data.txt" and returns t_test and x_test
def import_data(filename):
    data = np.loadtxt(filename)
    t_test = data[:, 0]
    x_test = data[:, 1:]
    return t_test, x_test


t_test, x_test = import_data("input_data/Sequential_data.txt")
concentrations = x_test.T
init_conc = concentrations[:, 0]

plt.figure()
step = 1
for i in range(concentrations.shape[0]):
    plt.plot(t_test[::step], concentrations[i][::step],'-',markersize = 11,markeredgewidth = 1.0,markeredgecolor = 'w', lw=2.0)
plt.xlabel("Time (s)", fontsize=20)
plt.ylabel("Concentration (mmol/L)", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(["A", "B", "C", "D"], fontsize=20)
plt.grid()
plt.savefig("Sequential_concentration.png", dpi=300, bbox_inches='tight')

print("True Reaction Mechanism")
print("A -> B, 6.312e-5")
print("B -> C, 1.262e-4")
print("C -> D, 3.156e-4")

# rmg inputs
number_of_generations = 10
mechanisms_per_generation = 100
min_rxns_per_mech = 3
max_rxns_per_mech = 3
fraction_of_mechanisms_passed_on = 0.1


mechanism_generator = rmg(order = 2, num_generations = number_of_generations, num_mech_per_gen = mechanisms_per_generation, max_rxns_per_mech = max_rxns_per_mech,min_rxns_per_mech = min_rxns_per_mech, from_previous_generation = fraction_of_mechanisms_passed_on, verbosity = 2)
mechanism_generator = mechanism_generator.fit(x_test,t_test)

###########################################################################
ks_fit = mechanism_generator.rate_constants
rm = list(mechanism_generator.reaction_mechanism)
print("Predicted rates = ", ks_fit)
print("Predicted mechanism = ", rm)
print("Mean Square Error = ", mechanism_generator.mse)

def odes(t, y):
    return reaction_system(t, y, rm, ks_fit)

# Solve the system using solve_ivp
sol = solve_ivp(odes, [0, t_test[-1]], init_conc, t_eval=t_test)
concentrations = sol.y

plt.figure()
for i in range(concentrations.shape[0]):
    plt.plot(sol.t, concentrations[0],ls = '--', lw = 3)

plt.xlabel("Time (s)", fontsize=20)
plt.ylabel("Concentration (mol/L)", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(["A", "B", "C", "D"], fontsize=20)
plt.grid()
plt.savefig("Sequential_concentration_fit.png", dpi=300, bbox_inches='tight')

# Output data file as txt with column 1 with time and next columns with concentrations
output_data = np.column_stack((sol.t, concentrations.T))
np.savetxt('fitted_sequential_data.txt', output_data, header='Time A B', fmt='%.6e')
