import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.optimize import brentq


h_bar = 1 #Reduced planck's constant [J * s]
L = 2 #Width of finite potential box [m]
m = 1 #Mass of particle within infinite potential [kg] 
V_0 = 50 #Potential energy of finite potential well
n_f = 1000 #Number of frames [1]
n_x = 3000 #Number of x-intervals [1]
x_max = 3 * L #x value range [m]
t_tot = 5 #Total time of simulation [s]

x = np.linspace(-x_max, x_max, n_x)

#Potential well function

def V(x):
    return np.where((x >= -L/2) & (x <= L/2), 0, V_0)

#Initial state of wavefunction- can be altered, but the less bound it is by the potential well, the less accurate the simulation

def PSI0(x):
    k0 = 5 
    """Directly related to energy of initial packet- as this value increases, less and less 
    of the initial wavepacket can be represented by bound states, making simulation less accurate"""
    return np.exp(-(x)**2/(2 * (0.2 * L)**2)) * np.exp(1j * k0 * x)

#Normalizes the initial wavefunction

def PSI0NORM(x):
    return PSI0(x)/np.sqrt(np.trapezoid(np.abs(PSI0(x))**2, x))

#Solves numerically for the energy solutions to the Schrodinger equation 

def energy_even(E_n):
    kappa1 = np.sqrt(2 * m * E_n)/h_bar
    kappa2 = np.sqrt(2 * m * (V_0 - E_n))/h_bar
    return kappa1 * np.tan(kappa1 * L/2) - kappa2

def energy_odd(E_n):
    kappa1 = np.sqrt(2 * m * E_n)/h_bar
    kappa2 = np.sqrt(2 * m * (V_0 - E_n))/h_bar
    return kappa1 * 1/np.tan(kappa1 * L/2) + kappa2

e_poss = np.linspace(10**(-6), V_0 - 10**(-6), 3000)

guess_even = energy_even(e_poss)
guess_odd = energy_odd(e_poss)

E_n_even = []
E_n_odd = []
E_n_even = np.array(E_n_even)
E_n_odd  = np.array(E_n_odd)

#Searches for when the transcendental equations are 0, to solve for the energy solutions, then solving for the values of kappa1 and kappa2 for both the even and odd solutions

for i in range(len(guess_even) - 1):
    if ((guess_even[i] > 0 and guess_even[i + 1] < 0) or (guess_even[i] < 0 and guess_even[i + 1] > 0)):
        root = brentq(energy_even, e_poss[i], e_poss[i + 1])
        E_n_even = np.append(E_n_even, root)

for i in range(len(guess_odd) - 1):
    if ((guess_odd[i] > 0 and guess_odd[i + 1] < 0) or (guess_odd[i] < 0 and guess_odd[i + 1] > 0)):
        root = brentq(energy_odd, e_poss[i], e_poss[i + 1])
        E_n_odd = np.append(E_n_odd, root)

kappa1_even = np.sqrt(2 * m * E_n_even)/h_bar
kappa2_even = np.sqrt(2 * m * (V_0 - E_n_even))/h_bar

kappa1_odd = np.sqrt(2 * m * E_n_odd)/h_bar
kappa2_odd = np.sqrt(2 * m * (V_0 - E_n_odd))/h_bar

#Defines the bound state eigenfunctions (the solutions to the Schrodinger equation) via kappa1 and kappa2

def psi_e_even(x, kappa1, kappa2):
    x = np.atleast_1d(x)
    psi_e = np.where(x < -L/2
                          , np.cos(kappa1 * L/2) * np.exp(kappa2 * (L/2 + x))
                          , np.where(x > L/2
                                     , np.cos(kappa1 * L/2) * np.exp(-kappa2 * (x - L/2))
                                     , np.cos(kappa1 * x)))
    psi_e /= np.sqrt(2 * (1/kappa2 * (np.cos(kappa1 * L/2))**2 + L/2 + np.sin(kappa1 * L)/(2 * kappa1)))
    return psi_e

def psi_e_odd(x, kappa1, kappa2):
    psi_e = np.where(x < -L/2
                          , -np.sin(kappa1 * L/2) * np.exp(kappa2 * (L/2 + x))
                          , np.where(x > L/2
                                     , np.sin(kappa1 * L/2) * np.exp(-kappa2 * (x - L/2))
                                     , np.sin(kappa1 * x)))
    psi_e /= np.sqrt(2 * (1/kappa2 * (np.sin(kappa1 * L/2))**2 + L/2 - np.sin(kappa1 * L)/(2 * kappa1)))
    return psi_e

#Solves for all coefficients of the eigenfunctions to be superposed 

c_n_even = [np.trapezoid(np.conj(psi_e_even(x, kappa1_even[i], kappa2_even[i])) * PSI0NORM(x), x)
               for i in range(len(E_n_even))]
c_n_odd  = [np.trapezoid(np.conj(psi_e_odd(x, kappa1_odd[i], kappa2_odd[i])) * PSI0NORM(x), x)
               for i in range(len(E_n_odd))]

#Combines all c_n and E_n solutions in two lists in order of energy 

c_n_list = []
c_n_sq_list = []
E_n_list = []

min_len = min(len(c_n_even), len(c_n_odd))
for i in range(min_len):
    c_n_list.append(c_n_even[i])
    c_n_list.append(c_n_odd[i])
    E_n_list.append(E_n_even[i])
    E_n_list.append(E_n_odd[i])
c_n_list.extend(c_n_even[min_len:])
c_n_list.extend(c_n_odd[min_len:])
E_n_list.extend(E_n_even[min_len:])
E_n_list.extend(E_n_odd[min_len:])

for i in c_n_list:
    c_n_sq_list.append(np.abs(i)**2)

#Filters E_n and C_n lists for values greater than 0.001

E_n_filtered = np.array(E_n_list)[c_n_sq_list > 0.001 * np.max(c_n_sq_list)]
c_n_sq_filtered = np.array(c_n_sq_list)[c_n_sq_list > 0.001 * np.max(c_n_sq_list)]

#Defines the time-dependent wavefunction psi, superposing the even and odd wavefunctions with time dependent factor 

def PSI(x, t):
    psi_x_t = 0 + 0j
    for i in range(len(c_n_even)):
        psi_x_t += c_n_even[i] * psi_e_even(x, kappa1_even[i], kappa2_even[i]) * np.exp(-1j * E_n_even[i]/h_bar * t)
    for i in range(len(c_n_odd)):
        psi_x_t += c_n_odd[i] * psi_e_odd(x, kappa1_odd[i], kappa2_odd[i]) * np.exp(-1j * E_n_odd[i]/h_bar * t)
    return psi_x_t

#Finds norm-squared of psi

def PSI2(x, t):
    return np.abs(PSI(x, t))**2

t = np.linspace(0, t_tot, n_f)

#Creates norm-squared psi, E_n, and c_n^2 graphs

fig, axis = plt.subplot_mosaic([["PSI2", "PSI2"]
                          , ["c_n2", "E_n"]]
                          , gridspec_kw = {'width_ratios' : [1, 1]}
                          , figsize = (12, 8)
                          )

#Plots energy data on E_n and c_n^2 plots

axis["PSI2"].plot(x, V(x) / (V_0) * 2/3, alpha=0.5, linewidth=1.5, label='V(x)', color = 'black')
axis["PSI2"].fill_between(x, V(x) / V_0 * 2/3, alpha=0.5, linewidth=1.5, color = (133/255, 74/255, 175/255))
axis["PSI2"].set_xlim(-x_max, x_max)
axis["PSI2"].set_ylim(0, 1)
axis["E_n"].set_xlim(0, x_max)
axis["E_n"].set_ylim(-0.1 * max(E_n_list), 1.1 * max(E_n_list))

axis["c_n2"].bar(E_n_filtered, c_n_sq_filtered, width = 0.3)

for i in E_n_list:
    axis["E_n"].plot(x, np.full(len(x), i))

#Sets axes and titles of graphs

axis["E_n"].set_ylabel("Energy of state [J]")

axis["PSI2"].set_title("|\N{Greek Capital Letter Psi}(x, t)|\N{Superscript Two}")
axis["c_n2"].set_title("Probability of measuring each energy state")
axis["E_n"].set_title("Allowed energy states")

plt.subplots_adjust(hspace=0.4)

#Defines values to be changed in animation

animated_PSI2, = axis["PSI2"].plot([], [])

prob_check = axis["PSI2"].text(0.98
    , 0.95
    , ""
    , transform=axis["PSI2"].transAxes
    , ha="right"
    , va="top"
)

time_check = axis["PSI2"].text(0.98
    , 0.85
    , ""
    , transform=axis["PSI2"].transAxes
    , ha="right"
    , va="top"
)

#Updates each frame with time-dependent data

def update_data(frame):
    animated_PSI2.set_data(x, PSI2(x, t[frame]))
    x_int = np.linspace(5 * -x_max, 5 * x_max, 9000)
    area = np.trapezoid(PSI2(x_int, t[frame]), x_int)
    prob_check.set_text(f"P_tot \u2248 {round(area, 2)}")
    time_check.set_text(f"t = {round(t[frame], 3)}")

    return animated_PSI2, prob_check, time_check

animation_system = FuncAnimation(fig = fig
                                 , func = update_data
                                 , frames = len(t)
                                 , interval = 20
                                 , repeat = False
                                 , blit = True)

axis["PSI2"].grid()
axis["E_n"].grid()

plt.show()