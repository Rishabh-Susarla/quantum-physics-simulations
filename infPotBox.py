import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.integrate import quad


h_bar = 1 #Reduced planck's constant [J * s]
L = 3 #Width of infinite potential box [m]
m = 1 #Mass of particle within infinite potential [kg] 
n_f = 2000 #Number of frames [1]
n_x = 200 #Number of x-intervals [1]
t_tot = 5 #Total time of simulation [s]
N_0 = 1 #Starting eigenfunction (helpful for approximating higher energy states) [1]
N = 50 #Number of eigenfunctions to superpose [1]

x = np.linspace(0, L, n_x)

#Initial state of wavefunction- can be altered, but the more complex it is the more eigenfunctions must be superposed to get accurate results

def PSI0(x):
    return np.exp(-(x - L/2)**2/(2 * (0.1 * L)**2)) * np.exp(1j * 10 * x)
##  np.exp(-(x - L/2)**2/(2 * (0.1 * L)**2)) * np.exp(1j * 10 * x)
##  np.sqrt(2/L) * np.sin(3 * np.pi/L * x) 
##  np.sqrt(2/L) * np.sin(3 * np.pi/L * x) + np.sqrt(2/L) * np.sin(4 * np.pi/L * x)
##  L**(-2) * (x - L/2)**2 * np.exp(1j * 6 * x)

#Normalizes initial wavefunction

area = quad(lambda x: np.abs(PSI0(x))**2, 0, L)[0]

def PSI0NORM(x):
    return PSI0(x)/np.sqrt(area)

#Defines eigenfunction to be superposed

def psi_e(x, n):
    return np.sqrt(2/L) * np.sin(n * np.pi/L * x)

#Defines frequency of time dependence

def w(n):
    return n**2 * np.pi**2 * h_bar/(2 * m * L**2)

#Defines function to generate coefficients of each superposed eigenfunction

def c_n(n):
    real = quad(lambda x: np.real(psi_e(x, n) * PSI0NORM(x)), 0, L)[0]
    imaginary = quad(lambda x: np.imag(psi_e(x, n) * PSI0NORM(x)), 0, L)[0]
    return real + 1j * imaginary

#Creates list of all coefficients

c_n_list = [c_n(i) for i in range(N_0, N_0 + N)]

#Defines time-dependent wavefunction psi as superposition of all Schrodinger equation solutions, with time dependent factor

def PSI(x, t):
    psi_x_t = 0 + 0j
    for i in range(N):
        psi_x_t += c_n_list[i] * psi_e(x, i + N_0) * np.exp(-1j * w(i + N_0) * t)
    return psi_x_t

#Finds norm-squared of psi

def PSI2(x, t):
    return np.abs(PSI(x, t))**2

t = np.linspace(0, t_tot, n_f)

#Creates norm-squared of psi graph, as well as graphs for the real and imaginary parts of psi 

fig, axis = plt.subplot_mosaic([["PSI2", "PSI2"]
                          , ["RePSI", "ImPSI"]]
                          , gridspec_kw = {'width_ratios' : [1, 1]}
                          )

axis["PSI2"].set_xlim(-0.5 * L, 1.5 * L)
axis["PSI2"].set_ylim(0, L)
axis["RePSI"].set_xlim(-0.5 * L, 1.5 * L)
axis["RePSI"].set_ylim(-0.5 * L, 0.5 * L)
axis["ImPSI"].set_xlim(-0.5 * L, 1.5 * L)
axis["ImPSI"].set_ylim(-0.5 * L, 0.5 * L)

axis["PSI2"].axvspan(-0.5 * L, 0, alpha=0.2, color='purple', label='Potential Well')
axis["RePSI"].axvspan(-0.5 * L, 0, alpha=0.2, color='purple', label='Potential Well')
axis["ImPSI"].axvspan(-0.5 * L, 0, alpha=0.2, color='purple', label='Potential Well')
axis["PSI2"].axvspan(L, 1.5 * L, alpha=0.2, color='purple', label='Potential Well')
axis["RePSI"].axvspan(L, 1.5 * L, alpha=0.2, color='purple', label='Potential Well')
axis["ImPSI"].axvspan(L, 1.5 * L, alpha=0.2, color='purple', label='Potential Well')

axis["PSI2"].plot(np.full(len(x), 0), np.full(len(x), np.linspace(0, L, n_x)), color = 'black', alpha = 0.2)
axis["RePSI"].plot(np.full(len(x), 0), np.full(len(x), np.linspace(-0.5 * L, 0.5 * L, n_x)), color = 'black', alpha = 0.2)
axis["ImPSI"].plot(np.full(len(x), 0), np.full(len(x), np.linspace(-0.5 * L, 0.5 * L, n_x)), color = 'black', alpha = 0.2)
axis["PSI2"].plot(np.full(len(x), L), np.full(len(x), np.linspace(0, L, n_x)), color = 'black', alpha = 0.2)
axis["RePSI"].plot(np.full(len(x), L), np.full(len(x), np.linspace(-0.5 * L, 0.5 * L, n_x)), color = 'black', alpha = 0.2)
axis["ImPSI"].plot(np.full(len(x), L), np.full(len(x), np.linspace(-0.5 * L, 0.5 * L, n_x)), color = 'black', alpha = 0.2)

#Sets axes and titles of graphs

axis["PSI2"].set_title("|\N{Greek Capital Letter Psi}(x, t)|\N{Superscript Two}")
axis["RePSI"].set_title("Re(\N{Greek Capital Letter Psi}(x, t))")
axis["ImPSI"].set_title("Im(\N{Greek Capital Letter Psi}(x, t))")

#Defines values to be changed in animation

plt.subplots_adjust(hspace=0.4)

animated_PSI2, = axis["PSI2"].plot([], [])
animated_RePSI, = axis["RePSI"].plot([], [])
animated_ImPSI, = axis["ImPSI"].plot([], [])

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
    animated_RePSI.set_data(x, np.real(PSI(x, t[frame])))
    animated_ImPSI.set_data(x, np.imag(PSI(x, t[frame])))

    area = quad(lambda xx: PSI2(xx, t[frame]), 0, L)[0]
    prob_check.set_text(f"P_tot \u2248 {round(area, 2)}")
    time_check.set_text(f"t = {round(t[frame], 3)}")

    return animated_PSI2, animated_RePSI, animated_ImPSI, prob_check, time_check

animation_system = FuncAnimation(fig = fig
                                 , func = update_data
                                 , frames = len(t)
                                 , interval = 5
                                 , repeat = False
                                 , blit = True)

axis["PSI2"].grid()
axis["RePSI"].grid()
axis["ImPSI"].grid()

plt.show()
