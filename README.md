# quantum-physics-simulations
A collection of simulations visualizing the analytic time evolution of wavefunctions in potential wells, using numerical solutions of the Schrödinger equation to represent individual energy eigenstates.

- Infinite potential well: Demonstrates discrete energy eigenstates and their time-dependent phase evolution, under an infinite potential boundary; designed to build intuition regarding the relation between energy eigenfunctions and time evolution of an initial wavefunction. 
- Finite potential well: Displays the time-evolution of an initial wavefunction and its allowed energy values, including the probability weights of each under a finite potential well; designed to display my analytical solution to the potential energy well problem, as well as to build further intuition regarding wavefunctions and their decomposition into their constituent eigenfunctions. 

Both run with Python 3.11+, with matplotlib, numpy, and scipy installed.
<table>
  <tr>
  <td> <img src = "https://github.com/user-attachments/assets/30caf4d4-d991-4034-8cd5-e42de6302c0d"
         width = 600
         height = 400/> </td>
  </tr>
</table>
<sub> (The time evolution as a Gaussian wavepacket bounces between the infinite potential walls; notice the time-symmetric nature of the motion, as well as the classical equivalent of a particle bouncing from wall to another in a box.) </sub>
<br>
<table>
  <tr>
    <br>
    <td> <img src = "https://github.com/user-attachments/assets/509d200c-e942-43ac-a493-7a2c6e1befc2"
           width = 600
           height = 400/> </td>
  </tr>
</table>
<sub> (The time evolution as a Gaussian wavepacket moves in a finite potential well; notice the leakage of the probability distribution into classically forbidden regions. Furthermore, the total probability decreasing over time can be attributed to more of the wavefunction being a representation of unbound states, eigenfunctions with energy greater than V_0, which are not represented in the simulation.) </sub>

