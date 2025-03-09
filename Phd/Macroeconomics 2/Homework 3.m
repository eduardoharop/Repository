clc;
clear all;
close all;

p.n = 1/3;
p.alpha = 0.36;
p.beta = 0.99;
p.delta = 0.025;
p.rho = 0.95;
p.sigma = 0.01;

k_n = (p.alpha/((1/p.beta)-(1-p.delta)))^(1/(1-p.alpha));
B = ((1-p.alpha)/p.n) * (k_n^p.alpha/(k_n^p.alpha - p.delta*k_n));
c = ((1-p.alpha)/B)*(k_n^p.alpha);
k = p.n*(k_n);
y = k^p.alpha * p.n^(1-p.alpha);
i = p.delta*k;

%% Part 2 DYNARE
cd 'C:\dynare\6.2\matlab'
addpath("C:\dynare\6.2\matlab")
dynare rbc_macro2_hw

%Part 2b


load('sim_order1.mat');
time = (1:length(sim_order1))';
% Example if the order is: c, k, n, y, z
c_order1 = sim_order1(:, strmatch('c', M_.endo_names, 'exact'));
k_order1 = sim_order1(:, strmatch('k', M_.endo_names, 'exact'));
n_order1 = sim_order1(:, strmatch('n', M_.endo_names, 'exact'));
z_order1 = sim_order1(:, strmatch('z', M_.endo_names, 'exact'));
y_order1 = sim_order1(:, strmatch('y', M_.endo_names, 'exact'));
k_n_order1 = sim_order1(:, strmatch('k_n', M_.endo_names, 'exact'));

load('sim_order2.mat');
c_order2 = sim_order2(:, strmatch('c', M_.endo_names, 'exact'));
k_order2 = sim_order2(:, strmatch('k', M_.endo_names, 'exact'));
n_order2 = sim_order2(:, strmatch('n', M_.endo_names, 'exact'));
z_order2 = sim_order2(:, strmatch('z', M_.endo_names, 'exact'));
y_order2 = sim_order2(:, strmatch('y', M_.endo_names, 'exact'));
k_n_order2 = sim_order2(:, strmatch('k_n', M_.endo_names, 'exact'));

figure;

subplot(2,3,1);
plot(time, c_order1, 'b', time, c_order2, 'r--');
title('Consumption (Order 1 vs Order 2)');
legend('Order 1', 'Order 2');
grid on;

subplot(2,3,2);
plot(time, k_order1, 'b', time, k_order2, 'r--');
title('Capital (Order 1 vs Order 2)');
legend('Order 1', 'Order 2');
grid on;

subplot(2,3,3);
plot(time, n_order1, 'b', time, n_order2, 'r--');
title('Labor (Order 1 vs Order 2)');
legend('Order 1', 'Order 2');
grid on;

subplot(2,3,4);
plot(time, y_order1, 'b', time, y_order2, 'r--');
title('Output (Order 1 vs Order 2)');
legend('Order 1', 'Order 2');
grid on;

subplot(2,3,5);
plot(time, k_n_order1, 'b', time, k_n_order2, 'r--');
title('Average Labor Productivity (Order 1 vs Order 2)');
legend('Order 1', 'Order 2');
grid on;

subplot(2,3,6);
plot(time, z_order1, 'b', time, z_order2, 'r--');
title('Productivity (Order 1 vs Order 2)');
legend('Order 1', 'Order 2');
grid on;

sgtitle('Simulated Series from First- and Second-Order Perturbations');

%Part 2c
load('moments_order1.mat')
disp(moments_order1.mean)

load('moments_order2.mat')
disp(moments_order2.mean)


%% Part 3 
disp(oo_.dr.order_var) %coincides with the names in M_.endo_names
rng(123);

SS = oo_.dr.ys;
A = oo_.dr.ghx;
B = oo_.dr.ghu;
del2 = oo_.dr.ghs2;
C = oo_.dr.ghxx;
D = oo_.dr.ghuu;
E = oo_.dr.ghxu;

% Generate shocks
T = 10000;   
sigma = 0.01;
shocks = sigma * randn(T, 1);

nvars = size(SS, 1);
PER1 = zeros(nvars, T); % First-order solution
PER2 = zeros(nvars, T); % Second-order solution
PER1(:,1) = SS';
PER2(:,1) = SS';

for t = 2:T
    % First-order perturbation (PER1)
    lag1 = PER1(:,t-1) - SS;
    PER1(:, t) = SS + A*lag1([4 5],:) + B * shocks(t);
    
    lag2 = PER2(:,t-1) - SS;
    PER2(:, t) = SS + 0.5 * del2 + A * lag2([4 5],:) + B * shocks(t) ...
                + 0.5 * C * kron(lag2([4 5],:), lag2([4 5],:)) ...
                + 0.5 * D * kron(shocks(t), shocks(t)) ...
                + E * kron(lag2([4 5],:), shocks(t));
    
end

time = (1:T)';

figure;

subplot(2,3,1);
plot(time, PER1(6, :), 'b', ...
     time, PER2(6, :), 'r--');
title('Consumption');
legend('Order 1', 'Order 2');
grid on;

subplot(2,3,2);
plot(time, PER1(4, :), 'b', ...
     time, PER2(4, :), 'r--');
title('Capital');
legend('Order 1', 'Order 2');
grid on;

subplot(2,3,3);
plot(time, PER1(7, :), 'b', ...
     time, PER2(7, :), 'r--');
title('Labor');
legend('Order 1', 'Order 2');
grid on;

subplot(2,3,4);
plot(time, PER1(1, :), 'b', ...
     time, PER2(1, :), 'r--');
title('Output');
legend('Order 1', 'Order 2');
grid on;

subplot(2,3,5);
plot(time, PER1(5, :), 'b', ...
     time, PER2(5, :), 'r--');
title('Technology');
legend('Order 1', 'Order 2');
grid on;

subplot(2,3,6);
plot(time, PER1(3, :), 'b', ...
     time, PER2(3, :), 'r--');
title('Average Labor Productivity');
legend('Order 1', 'Order 2');
grid on;

sgtitle('Simulated Series from First- and Second-Order Perturbations');

%% Part 4
% After simulating PER1 and PER2 in Problem 3:
burn_in = 200;
T_total = 10000;
sim_periods = (burn_in + 1):T_total;

% Extract variables for PER1 and PER2 (adjust indices based on Dynare's order_var)
% Example order: [c, k, n, y, theta]
c_PER1 = PER1(6, sim_periods);
k_PRE_PER1 = PER1(4, sim_periods - 1); % k_t is k_{t+1} from t-1
n_PER1 = PER1(7, sim_periods);
theta_PER1 = PER1(5, sim_periods);
k_plus1_PER1 = PER1(4, sim_periods); % k_{t+1}

% Repeat for PER2
c_PER2 = PER2(6, sim_periods);
k_PRE_PER2 = PER2(4, sim_periods - 1);
n_PER2 = PER2(7, sim_periods);
theta_PER2 = PER2(5, sim_periods);
k_plus1_PER2 = PER2(4, sim_periods);

alpha = 0.36; 
delta = 0.025;

%Resdiuals from the Budget Constraint (R_BC)
R_BC_PER1 = ((1 - delta)*k_PRE_PER1 + theta_PER1 .* k_PRE_PER1.^alpha .* n_PER1.^(1 - alpha)) ...
    ./ (c_PER1 + k_plus1_PER1) - 1;

R_BC_PER2 = ((1 - delta)*k_PRE_PER2 + theta_PER2 .* k_PRE_PER2.^alpha .* n_PER2.^(1 - alpha)) ...
    ./ (c_PER2 + k_plus1_PER2) - 1;

%Resdiuals from the Marginal Utility of Leisure (R_MUL)
B = 2.58; % From calibration
u1_PER1 = 1 ./ c_PER1;
f2_PER1 = (1 - alpha) * theta_PER1 .* k_PRE_PER1.^alpha .* n_PER1.^(-alpha);
R_MUL_PER1 = (u1_PER1 .* f2_PER1) / (B) - 1;

u1_PER2 = 1 ./ c_PER2;
f2_PER2 = (1 - alpha) * theta_PER2 .* k_PRE_PER2.^alpha .* n_PER2.^(-alpha);
R_MUL_PER2 = (u1_PER2 .* f2_PER2) / (B) - 1;

%Compute Residuals for Euler Equation (R_EE)
epsilon_nodes = [1, -1]; % Nodes for ε ~ N(0,1)
weights = [0.5, 0.5]; % Weights for two-node quadrature
beta = 0.99; rho = 0.95; sigma = 0.01;

R_EE_PER1 = zeros(1, length(sim_periods));
R_EE_PER2 = zeros(1, length(sim_periods));

% Extract steady-state values
SS_k = SS(4); % Capital in steady state
SS_theta = SS(5); % Theta in steady state

for t = 1:length(sim_periods)
    % For PER1
    current_theta = theta_PER1(t);
    current_c = c_PER1(t);
    current_k_plus1 = k_plus1_PER1(t);
    
    sum_terms_PER1 = 0;
    for j = 1:2
        theta_plus1_j = current_theta^rho * exp(sigma * epsilon_nodes(j));
        dev_k = current_k_plus1 - SS_k;
        dev_theta = theta_plus1_j - SS_theta;
        
        % First-order decision rule for c and n at t+1
        c_plus1_j = SS(6) + oo_.dr.ghx(6,1)*dev_k + oo_.dr.ghx(6,2)*dev_theta;
        n_plus1_j = SS(7) + oo_.dr.ghx(7,1)*dev_k + oo_.dr.ghx(7,2)*dev_theta;
        
        term_j = (current_c / c_plus1_j) * (1 - delta + alpha * theta_plus1_j * (current_k_plus1 / n_plus1_j)^(alpha - 1));
        sum_terms_PER1 = sum_terms_PER1 + weights(j) * term_j;
    end
    R_EE_PER1(t) = beta * sum_terms_PER1 - 1;
    
    % Repeat for PER2 (include second-order terms)
    current_theta = theta_PER2(t);
    current_c = c_PER2(t);
    current_k_plus1 = k_plus1_PER2(t);
    
    sum_terms_PER2 = 0;
    for j = 1:2
        theta_plus1_j = current_theta^rho * exp(sigma * epsilon_nodes(j));
        dev_k = current_k_plus1 - SS_k;
        dev_theta = theta_plus1_j - SS_theta;
        
        % Second-order decision rule
        state_dev = [dev_k; dev_theta];
        kron_state = kron(state_dev, state_dev);
        c_plus1_j = SS(6) + 0.5*oo_.dr.ghs2(6) + oo_.dr.ghx(1,:)*state_dev ...
            + 0.5*oo_.dr.ghxx(6,:)*kron_state;
        n_plus1_j = SS(7) + 0.5*oo_.dr.ghs2(7) + oo_.dr.ghx(7,:)*state_dev ...
            + 0.5*oo_.dr.ghxx(7,:)*kron_state;
        
        term_j = (current_c / c_plus1_j) * (1 - delta + alpha * theta_plus1_j * (current_k_plus1 / n_plus1_j)^(alpha - 1));
        sum_terms_PER2 = sum_terms_PER2 + weights(j) * term_j;
    end
    R_EE_PER2(t) = beta * sum_terms_PER2 - 1;
end

%Results

% For PER1
RBC_mean_PER1= mean(abs(R_BC_PER1));
RBC_max_PER1= max(abs(R_BC_PER1));
REE_mean_PER1= mean(abs(R_EE_PER1));
REE_max_PER1= max(abs(R_EE_PER1));
RMUL_mean_PER1= mean(abs(R_MUL_PER1));
RMUL_max_PER1= max(abs(R_MUL_PER1));

disp("Results for PER1");
fprintf('Mean |Resid BC| (log10): %.4f\n', log10(RBC_mean_PER1));
fprintf('Max |Resid BC| (log10): %.4f\n', log10(RBC_max_PER1));
fprintf('Mean |Resid EE| (log10): %.4f\n', log10(REE_mean_PER1));
fprintf('Max |Resid EE| (log10): %.4f\n', log10(REE_max_PER1));
fprintf('Mean |Resid MUL| (log10): %.4f\n', log10(RMUL_mean_PER1));
fprintf('Max |Resid MUL| (log10): %.4f\n', log10(RMUL_max_PER1));

% For PER2

RBC_mean_PER2= mean(abs(R_BC_PER2));
RBC_max_PER2= max(abs(R_BC_PER2));
REE_mean_PER2= mean(abs(R_EE_PER2));
REE_max_PER2= max(abs(R_EE_PER2));
RMUL_mean_PER2= mean(abs(R_MUL_PER2));
RMUL_max_PER2= max(abs(R_MUL_PER2));

disp("Results for PER2");
fprintf('Mean |Resid BC| (log10): %.4f\n', log10(RBC_mean_PER2));
fprintf('Max |Resid BC| (log10): %.4f\n', log10(RBC_max_PER2));
fprintf('Mean |Resid EE| (log10): %.4f\n', log10(REE_mean_PER2));
fprintf('Max |Resid EE| (log10): %.4f\n', log10(REE_max_PER2));
fprintf('Mean |Resid MUL| (log10): %.4f\n', log10(RMUL_mean_PER2));
fprintf('Max |Resid MUL| (log10): %.4f\n', log10(RMUL_max_PER2));
