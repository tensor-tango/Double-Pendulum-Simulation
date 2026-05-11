# Double Pendulum Simulation

Real-time simulation of a chaotic double pendulum system, written in MATLAB and Python.
More detailed scientific- and engineering-based information of the system, numerical algorithm, motion equations and total summary can be found in dedicated file.

## Demonstration

Below is demonstrated Python version of simulation
![Python-Simulation](animation.gif)

Below is demonstrated MATLAB version of simulation
![MATLAB-Simulation](MATLAB_anim.gif)

## Main features

Main features:
- RK4 (Runge-Kutta of the 4th order) numerical integrator
- Real-time calculation of the motion's equations
- Real-time simulation rendering 
- Kinetic, potential and total system's energies analysis and live visualization
- Phase portrait 
- Tracking of chaotic trajectory of the second mass 

## Models descriptions and notices

### Computational notices
The masses of the rods L_1 and L_2 are omitted and assumed to be massless, masses m_1 and m_2 are treated as the point masses, the whole system is in 2D due to pendulum simplification

### Physical model

System consists of two connected pendulums, moving in a two-dimensional plane

### Numerical algorithm

The choice of the used numerical algorith bases on the most efficient and accurate algorithm exactly for this problem. Main factors were:
- Accuracy under conditions of non-linearity and chaos
- Exprected computational needs
- animation adaptivity
- accuracy under conditions of energy loss and motion depression
- convenience under one-step computation
