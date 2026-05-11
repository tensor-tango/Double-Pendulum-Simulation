clear;
close all;
clc;

% initial parameters

%free-fall acceleration
g = 9.81;

%mass of the bobs
m1 = 1;
m2 = 1;

%length of the arms
L1 = 1;
L2 = 1;

%friction in joints
gamma1 = 0.02;
gamma2 = 0.08;

%step for the RK
dt = 0.01;

%initial angle parameters
state = [
    pi/2;
    0;
    pi/2 + 0.01;
    0
];

initial_state = state;


% GUI 


figure('Name', 'Double Pendulum Simulation', 'NumberTitle', 'off');
set(gcf, 'Position', [100 100 1400 800]);


% simulation plot
 

ax_sim = subplot(2, 2, 1);
hold(ax_sim, 'on');
grid(ax_sim, 'on');
axis(ax_sim, 'equal');

title(ax_sim, 'Double Pendulum');
xlabel(ax_sim, 'x [m]');
ylabel(ax_sim, 'y [m]');

xlim(ax_sim, [-3 3]);
ylim(ax_sim, [-3 3]);

line1 = plot(ax_sim, [0 0], [0 0], 'b', 'LineWidth', 3);
line2 = plot(ax_sim, [0 0], [0 0], 'r', 'LineWidth', 3);

mass1 = scatter(ax_sim, 0, 0, 100, 'b', 'filled');
mass2 = scatter(ax_sim, 0, 0, 100, 'r', 'filled');

trace_x = [];
trace_y = [];

trace = plot(ax_sim, NaN, NaN, 'Color', [0.6 0 0.8], 'LineWidth', 1.3);

max_trace = 2000;


% energy plot


ax_energy = subplot(2, 2, 2);
hold(ax_energy, 'on');
grid(ax_energy, 'on');

title(ax_energy, 'Energy');
xlabel(ax_energy, 'time [s]');
ylabel(ax_energy, 'energy [J]');

lineT = plot(ax_energy, NaN, NaN, 'r', 'LineWidth', 1.5);
lineV = plot(ax_energy, NaN, NaN, 'g', 'LineWidth', 1.5);
lineE = plot(ax_energy, NaN, NaN, 'c', 'LineWidth', 1.5);

legend(ax_energy, {'Kinetic', 'Potential', 'Total'}, 'Location', 'best');

t_values = [];
T_values = [];
V_values = [];
E_values = [];


% phase portrait


ax_phase = subplot(2, 2, 3);
hold(ax_phase, 'on');
grid(ax_phase, 'on');

title(ax_phase, 'Phase portrait \theta_1 vs \omega_1');
xlabel(ax_phase, '\theta_1 [rad]');
ylabel(ax_phase, '\omega_1 [rad/s]');

xlim(ax_phase, [-pi pi]);
ylim(ax_phase, [-8 8]);

phase_x = [];
phase_y = [];

phase_line = plot(ax_phase, NaN, NaN, 'Color', [1 0.5 0], 'LineWidth', 1.5);


% parameters and initial conditions visualization


ax_info = subplot(2, 2, 4);
axis(ax_info, 'off');

params_text = sprintf([ ...
    'System parameters\n\n', ...
    'm1 = %.2f kg\n', ...
    'm2 = %.2f kg\n\n', ...
    'L1 = %.2f m\n', ...
    'L2 = %.2f m\n\n', ...
    'g  = %.2f m/s^2\n\n', ...
    '\\gamma1 = %.3f 1/s\n', ...
    '\\gamma2 = %.3f 1/s\n\n', ...
    'dt = %.4f s\n'], ...
    m1, m2, L1, L2, g, gamma1, gamma2, dt);

initial_text = sprintf([ ...
    'Initial conditions\n\n', ...
    '\\theta1 = %.2f rad\n', ...
    '\\omega1 = %.2f rad/s\n\n', ...
    '\\theta2 = %.2f rad\n', ...
    '\\omega2 = %.2f rad/s\n'], ...
    initial_state(1), initial_state(2), initial_state(3), initial_state(4));

text(ax_info, 0.05, 0.95, params_text, ...
    'Units', 'normalized', ...
    'FontSize', 12, ...
    'FontName', 'Courier New', ...
    'VerticalAlignment', 'top');

text(ax_info, 0.55, 0.95, initial_text, ...
    'Units', 'normalized', ...
    'FontSize', 12, ...
    'FontName', 'Courier New', ...
    'VerticalAlignment', 'top');


% initial energy


[~, ~, E0] = energies(state, g, m1, m2, L1, L2);

t = 0;


% main simulation loop


while ishandle(gcf)

    tic;

    % RK4 integration step
    state = rk4_step(state, dt, g, m1, m2, L1, L2, gamma1, gamma2);

    theta1 = state(1);
    omega1 = state(2);
    theta2 = state(3);
    omega2 = state(4);

    % angles to coordinates conversion

    x1 = L1 * sin(theta1);
    y1 = -L1 * cos(theta1);

    x2 = x1 + L2 * sin(theta2);
    y2 = y1 - L2 * cos(theta2);

    % animation update

    set(line1, ...
        'XData', [0 x1], ...
        'YData', [0 y1]);

    set(line2, ...
        'XData', [x1 x2], ...
        'YData', [y1 y2]);

    set(mass1, ...
        'XData', x1, ...
        'YData', y1);

    set(mass2, ...
        'XData', x2, ...
        'YData', y2);

    % second mass trace update

    trace_x(end + 1) = x2;
    trace_y(end + 1) = y2;

    if length(trace_x) > max_trace
        trace_x(1) = [];
        trace_y(1) = [];
    end

    set(trace, ...
        'XData', trace_x, ...
        'YData', trace_y);

    % energy plot update

    [T, V, E] = energies(state, g, m1, m2, L1, L2);

    t_values(end + 1) = t;
    T_values(end + 1) = T;
    V_values(end + 1) = V;
    E_values(end + 1) = E;

    set(lineT, ...
        'XData', t_values, ...
        'YData', T_values);

    set(lineV, ...
        'XData', t_values, ...
        'YData', V_values);

    set(lineE, ...
        'XData', t_values, ...
        'YData', E_values);

    % showing only last 20 seconds of energy plot
    xlim(ax_energy, [max(0, t - 20), max(20, t)]);

    all_energy = [T_values, V_values, E_values];

    if ~isempty(all_energy)
        ymin = min(all_energy);
        ymax = max(all_energy);

        if ymin ~= ymax
            ylim(ax_energy, [ymin - 1, ymax + 1]);
        end
    end

    % phase portrait update

    % angle normalization to [-pi, pi]
    theta1_plot = atan2(sin(theta1), cos(theta1));

    phase_x(end + 1) = theta1_plot;
    phase_y(end + 1) = omega1;

    set(phase_line, ...
        'XData', phase_x, ...
        'YData', phase_y);

    % fixed axes are for no autoscale 
    xlim(ax_phase, [-pi pi]);
    ylim(ax_phase, [-8 8]);

    % advencement in time

    t = t + dt;

    drawnow limitrate;

    elapsed = toc;

    if elapsed < dt
        pause(dt - elapsed);
    end
end

% energy calculations

function [T, V, E] = energies(y, g, m1, m2, L1, L2)

theta1 = y(1);
omega1 = y(2);
theta2 = y(3);
omega2 = y(4);

x1 = L1 * sin(theta1);
y1 = -L1 * cos(theta1);

x2 = x1 + L2 * sin(theta2);
y2 = y1 - L2 * cos(theta2);

v1_sq = (L1 * omega1)^2;

v2_sq = v1_sq ...
    + (L2 * omega2)^2 ...
    + 2 * L1 * L2 * omega1 * omega2 * cos(theta1 - theta2);

T = 0.5 * m1 * v1_sq + 0.5 * m2 * v2_sq;

V = m1 * g * y1 + m2 * g * y2;

E = T + V;

end

% Motion equations

function dydt = derivatives(y, g, m1, m2, L1, L2, gamma1, gamma2)

theta1 = y(1);
omega1 = y(2);
theta2 = y(3);
omega2 = y(4);

delta = theta1 - theta2;

M = m1 + m2;
alpha = m1 + m2 * sin(delta)^2;

dtheta1 = omega1;
dtheta2 = omega2;

domega1 = ...
    ( ...
    -sin(delta) * ...
    (m2 * L1 * omega1^2 * cos(delta) + m2 * L2 * omega2^2) ...
    - g * ...
    (M * sin(theta1) - m2 * sin(theta2) * cos(delta)) ...
    ) ...
    / ...
    (L1 * alpha);

domega2 = ...
    ( ...
    sin(delta) * ...
    (M * L1 * omega1^2 + m2 * L2 * omega2^2 * cos(delta)) ...
    + g * ...
    (M * sin(theta1) * cos(delta) - M * sin(theta2)) ...
    ) ...
    / ...
    (L2 * alpha);

% Damping
domega1 = domega1 - gamma1 * omega1;
domega2 = domega2 - gamma2 * omega2;

dydt = [
    dtheta1;
    domega1;
    dtheta2;
    domega2
];

end

% RK4 step

function y_next = rk4_step(y, dt, g, m1, m2, L1, L2, gamma1, gamma2)

k1 = derivatives(y, g, m1, m2, L1, L2, gamma1, gamma2);
k2 = derivatives(y + 0.5 * dt * k1, g, m1, m2, L1, L2, gamma1, gamma2);
k3 = derivatives(y + 0.5 * dt * k2, g, m1, m2, L1, L2, gamma1, gamma2);
k4 = derivatives(y + dt * k3, g, m1, m2, L1, L2, gamma1, gamma2);

y_next = y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6;

end
