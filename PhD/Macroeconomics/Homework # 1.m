clc;
clear all;
close all;
%% Initialize

% parametes
alpha = 0.3;
beta  = 0.7;
sigma = 2;
eta = 0.5;
phi = 0.001;
delta = 0.1;

k = 0.005:0.005:0.25; % grid for state variable, capital stock
k_0 = 0.045;
k_ind = zeros(50,1);
n_ind = zeros(50,1);
N = zeros(50,50);
C = zeros(50,50);
V = zeros(50,50);
Y = zeros(50,50);

[~, k0_index] = min(abs(k - k_0));

% Arrays to store the time series for capital, labor, and consumption
k_series = k_0;   % Capital stock over time (initialize with k_0)
l_series = [];      % Labor over time
c_series = []; % Consumption over time
y_series = []; % Consumption over time


% Initial guess for the value function
V0 = zeros(50,1);
V1 = zeros(50,1);
 
function l = labor(n,k_t,k_t_1,alpha,phi,delta,sigma,eta) 
    l = ((1-alpha)*k_t^alpha* n^-alpha) / (k_t^alpha * n^(1-alpha) - k_t_1 + (1 + phi - delta)*k_t) - sigma * n^eta;
end

%% Iteration

% Initialize tol and iter
tol  = 1;
iter = 0;

% loop 
while tol > 1e-4
    for i = 1:50
        k_t = k(i);
        for j =1:50
            k_t_1 = k(j);
            fun = @(n) labor(n,k_t,k_t_1,alpha,phi,delta,sigma,eta);
            N(i,j) = fsolve(fun, 0.5);  % Initial guess for n_t
            if N(i,j) < 0 
            N(i,j) = realmin;
            end
            if N(i,j) >= 1 
            N(i,j) = 1-realmin;
            end
            Y(i,j) = k(i)^alpha * N(i,j)^(1-alpha);
            C(i,j) = Y(i,j) - k(j) + (1 + phi - delta) * k(i); % Compute consumption c_t
            V(i,j) = log(C(i,j)) - sigma*(N(i,j)^(1+eta))/(1+eta) + beta*V0(j);
        end
        [V1(i),k_ind(i)]=max(V(i,:));
        n_ind(i) = max(N(i,:));
    end
    % Track time series for optimal path starting from k_0
    k_opt = k(k_ind(k0_index));      % Optimal capital for the current period
    n_opt = N(k0_index, k_ind(k0_index));  % Optimal labor for the current period
    c_opt = C(k0_index, k_ind(k0_index));  % Optimal consumption for the current period
    y_opt = Y(k0_index, k_ind(k0_index));  % Optimal consumption for the current period
    
    % Append optimal values to the time series
    k_series = [k_series; k_opt];
    l_series = [l_series; n_opt];
    c_series = [c_series; c_opt];
    y_series = [y_series; y_opt];
    
    k0_index = k_ind(k0_index);  % Move to the next period's capital index
    tol = norm(V1-V0);
    V0  = V1;
    iter = iter + 1;
end
%% first plot

fig1 = figure; 

% value function: 
subplot(1,3,1);
plot(k,V1);
xlabel('k');
ylabel('value function');
xlim([0,k(end)]);

% capital policy function:
subplot(1,3,2);
plot(k,k(k_ind));
xlabel('k');
ylabel('capital policy function');
xlim([0,k(end)]);

% labor policy function:
subplot(1,3,3);
plot(k,n_ind);
xlabel('k');
ylabel('capital policy function');
xlim([0,k(end)]);


% Save the figure
saveas(fig1, 'Value_and_Policy_Functions.png');

%% second plot 

fig2 = figure; 

% time series of capital: 
subplot(1,4,1);
plot(1:1:iter,k_series(1:30));
xlabel('time');
ylabel('Time series of K');
xlim([0,iter]);

% time series of labor: 
subplot(1,4,2);
plot(1:1:iter,l_series);
xlabel('time');
ylabel('Time series of Labor');
xlim([0,iter]);

% time series of consumption: 
subplot(1,4,3);
plot(1:1:iter,c_series);
xlabel('time');
ylabel('Time series of Consumption');
xlim([0,iter]);

% time series of output: 
subplot(1,4,4);
plot(1:1:iter,y_series);
xlabel('time');
ylabel('Time series of Output');
xlim([0,iter]);
