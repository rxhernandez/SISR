import sys, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.constants as sp
from scipy.optimize import least_squares
#sys.path.append(os.path.abspath('../../4-nonlin/rxn_kin_gen_algo'))
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from pylab import *
from rxn_mech_gen_gtc import reaction_mechanism_generator as rmg
from rxn_mech_gen_gtc import reaction_system
#from mech_plot import plot_mechanism
matplotlib.rcParams["font.family"] = "Times New Roman"
#plt.rcParams.update({"text.usetex": True})
matplotlib.rcParams['figure.figsize'] = 6,6/1.82
matplotlib.rcParams['axes.linewidth'] = 1.5
matplotlib.rcParams['figure.figsize'] = 8,8/1.62

def normalize_probs(probs):
    return probs / np.sum(probs)

def derivatives(t, concentration, k0, k1, k2, k3, k4):
    A, B = concentration
    dAdt =  -k0*2*A*A - k2*A + k2*2*A - k3*A*B 
    dBdt =  -k1*2*B*B - k3*A*B +k3*2*A*B -k4*B
    return [dAdt, dBdt]

#[[2,0,0,0,2,0],[0,1,0,0,0,1],[0,1,1,2,0,0]]

# Seed the random number generators for reproducibility
np.random.seed(100)

# Constants
Num_particles = 1
total_time = 20
#Dt = 5e-2
Dt = 2e-1

# Rate constants
k0 = 0.1
k1 = 0.1
k2 = 1
k3 = 1
k4 = 1
ks = (k0, k1, k2,k3,k4)
print(*ks)
print("true rate constants = ", *ks)

# Initial distribution and concentration
init_dist = normalize_probs(np.array([1.00, 1.00]))
init_conc = init_dist * 2
time = np.arange(0, total_time+Dt, Dt)

# Solve the system using solve_ivp
sol = solve_ivp(derivatives, [0, time[-1]], init_conc, t_eval=time, args=ks)

sol_time = sol.t
concentrations = sol.y
#print(sol_time.shape,concentrations.shape)

t_test = sol_time
x_test = concentrations.T

plt.figure()

step = int(total_time/Dt/20)
step = 1
print(step)
    
plt.plot(sol_time[::step], concentrations[0][::step],'-',markersize = 11,markeredgewidth = 1.0,markeredgecolor = 'w', lw=2.0,color="k")
plt.plot(sol_time[::step], concentrations[1][::step],'-',markersize = 11,markeredgewidth = 1.0,markeredgecolor = 'w', lw=2.0,color="k")
#plt.show()
#sys.exit()
plt.show()

out_file=open("./true_data.txt", "w") #windows
i = 0
while i<len(sol_time):
	print(sol_time[i],concentrations[0][i],concentrations[1][i], file = out_file)
	i+=1
#############################################
out_file.close()																																   
#############################################


# rmg inputs
number_of_generations = 20
mechanisms_per_generation = 2000
min_rxns_per_mech = 2
max_rxns_per_mech = 2
fraction_of_mechanisms_passed_on = 0.1


mechanism_generator = rmg(order = 2, num_generations = number_of_generations, num_mech_per_gen = mechanisms_per_generation, max_rxns_per_mech = max_rxns_per_mech,min_rxns_per_mech = min_rxns_per_mech, from_previous_generation = fraction_of_mechanisms_passed_on)
mechanism_generator, mech_list = mechanism_generator.fit(x_test,t_test)

###########################################################################
ks = list(mech_list[1])
print("Predicted rates = ", ks)

rm = list(mech_list[2])

def odes(t, y):
    return reaction_system(t, y, rm, ks)  


# Solve the system using solve_ivp
sol = solve_ivp(odes, [0, time[-1]], init_conc, t_eval=time)

sol_time = sol.t
concentrations = sol.y


plt.plot(sol_time, concentrations[0],ls = '--', lw = 3, label = "A")    
plt.plot(sol_time, concentrations[1],ls = '--', lw = 3, label = "B")    

out_file=open("./fit_data.txt", "w") #windows
i = 0
while i<len(sol_time):
	print(sol_time[i],concentrations[0][i],concentrations[1][i], file = out_file)
	i+=1
#############################################
out_file.close()

     
        
							  
									

																   
																			
														  
plt.show()