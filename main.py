import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import time

# -------------------------
# Initial parameters
# -------------------------

g = 9.81
m1 = 1
m2 = 1
L1 = 1
L2 = 1

gamma1 = 0.02
gamma2 = 0.08

dt = 0.01

state = np.array([
    np.pi/2,
    0,
    np.pi/2 + 0.01,
    0
])

# -------------------------
# Energy calculations
# -------------------------

def energies(y):

    theta1, omega1, theta2, omega2 = y

    x1 = L1*np.sin(theta1)
    y1 = -L1*np.cos(theta1)

    x2 = x1 + L2*np.sin(theta2)
    y2 = y1 - L2*np.cos(theta2)

    v1_sq = (L1*omega1)**2
    v2_sq = v1_sq + (L2*omega2)**2 + 2*L1*L2*omega1*omega2*np.cos(theta1-theta2)

    T = 0.5*m1*v1_sq + 0.5*m2*v2_sq
    V = m1*g*y1 + m2*g*y2

    return T, V, T+V

# -------------------------
# Motion equations
# -------------------------

def derivatives(y):

    theta1, omega1, theta2, omega2 = y

    delta = theta1 - theta2
    M = m1 + m2
    alpha = m1 + m2*np.sin(delta)**2

    dtheta1 = omega1
    dtheta2 = omega2

    domega1 = (
        -np.sin(delta)*(m2*L1*omega1**2*np.cos(delta) + m2*L2*omega2**2)
        -g*(M*np.sin(theta1) - m2*np.sin(theta2)*np.cos(delta))
    )/(L1*alpha)

    domega2 = (
        np.sin(delta)*(M*L1*omega1**2 + m2*L2*omega2**2*np.cos(delta))
        +g*(M*np.sin(theta1)*np.cos(delta) - M*np.sin(theta2))
    )/(L2*alpha)

    domega1 -= gamma1*omega1
    domega2 -= gamma2*omega2

    return np.array([dtheta1, domega1, dtheta2, domega2])

# -------------------------
# RK4
# -------------------------

def rk4_step(y, dt):

    k1 = derivatives(y)
    k2 = derivatives(y + 0.5*dt*k1)
    k3 = derivatives(y + 0.5*dt*k2)
    k4 = derivatives(y + dt*k3)

    return y + dt*(k1 + 2*k2 + 2*k3 + k4)/6

# -------------------------
# GUI setup
# -------------------------

plt.style.use("default")

fig = plt.figure(figsize=(14,8))
plt.subplots_adjust(hspace=0.35, wspace=0.25)

ax_sim = plt.subplot(221)
ax_energy = plt.subplot(222)
ax_phase = plt.subplot(223)
ax_info = plt.subplot(233)

# -------------------------
# Pendulum GUI elements
# -------------------------

ax_sim.set_title("Double Pendulum")
ax_sim.set_xlabel("x [m]")
ax_sim.set_ylabel("y [m]")

ax_sim.set_xlim(-3,3)
ax_sim.set_ylim(-3,3)
ax_sim.set_aspect("equal")

line1, = ax_sim.plot([],[], lw=3)
line2, = ax_sim.plot([],[], lw=3)
mass = ax_sim.scatter([],[], s=80)

trace_points = []
trace = LineCollection([], linewidth=2, cmap="plasma")
ax_sim.add_collection(trace)

# -------------------------
# Energy GUI elements
# -------------------------

t_values = []
T_values = []
V_values = []
E_values = []

lineT, = ax_energy.plot([],[], color="red", label="Kinetic")
lineV, = ax_energy.plot([],[], color="green", label="Potential")
lineE, = ax_energy.plot([],[], color="cyan", label="Total")

ax_energy.legend()
ax_energy.set_title("Energy")
ax_energy.set_xlabel("time [s]")
ax_energy.set_ylabel("energy [J]")

# -------------------------
# Phase portrait GUI elements
# -------------------------

phase_points_x = []
phase_points_y = []

phase_line, = ax_phase.plot([],[], color="orange")

ax_phase.set_title("Phase portrait θ1 vs ω1")
ax_phase.set_xlabel("θ₁ [rad]")
ax_phase.set_ylabel("ω₁ [rad/s]")

# -------------------------
# Parameters and initial conditions GUI elements
# -------------------------

ax_info.axis("off")

params_text = f"""
System parameters

m1 = {m1}kg
m2 = {m2}kg

L1 = {L1}m
L2 = {L2}m

g  = {g}m/s²

γ1 = {gamma1} 1/s
γ2 = {gamma2} 1/s

dt = {dt}s
"""
initial_text = f"""
Initial conditions

θ1 = {state[0]:.2f}rad 
ω1 = {state[1]:.2f}rad/s

θ2 = {state[2]:.2f}rad
ω2 = {state[3]:.2f}rad/s
"""

ax_info.text(
    0.5,
    -0.15,
    params_text,
    transform=ax_info.transAxes,
    fontsize=12,
    verticalalignment="top",
    family="monospace"
)

ax_info.text(
    -0.5,
    -0.15,
    initial_text,
    transform=ax_info.transAxes,
    fontsize=12,
    verticalalignment="top",
    family="monospace"
)

# -------------------------
# Initial energy
# -------------------------

_,_,E0 = energies(state)

max_trace = 2000
t = 0

# -------------------------
# Main loop
# -------------------------

while True:

    start = time.time()

    state[:] = rk4_step(state, dt)

    theta1, omega1, theta2, _ = state

    x1 = L1*np.sin(theta1)
    y1 = -L1*np.cos(theta1)

    x2 = x1 + L2*np.sin(theta2)
    y2 = y1 - L2*np.cos(theta2)

    # pendulum
    line1.set_data([0,x1],[0,y1])
    line2.set_data([x1,x2],[y1,y2])
    mass.set_offsets([[x1,y1],[x2,y2]])

    trace_points.append((x2,y2))
    if len(trace_points) > max_trace:
        trace_points.pop(0)

    if len(trace_points) > 1:

        segments = [
            [trace_points[i], trace_points[i+1]]
            for i in range(len(trace_points)-1)
        ]

        trace.set_segments(segments)
        trace.set_array(np.linspace(0,1,len(segments)))

    # Energy
    T,V,E = energies(state)

    t_values.append(t)
    T_values.append(T)
    V_values.append(V)
    E_values.append(E)

    lineT.set_data(t_values,T_values)
    lineV.set_data(t_values,V_values)
    lineE.set_data(t_values,E_values)

    ax_energy.relim()
    ax_energy.autoscale_view()

    # phase portrait
    phase_points_x.append(theta1)
    phase_points_y.append(omega1)

    phase_line.set_data(phase_points_x, phase_points_y)

    ax_phase.relim()
    ax_phase.autoscale_view()

    t += dt

    plt.draw()
    plt.pause(0.001)

    elapsed = time.time() - start
    if elapsed < dt:
        time.sleep(dt - elapsed)