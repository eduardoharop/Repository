clc;
clear all;
close all;

cd 'C:\Users\eduar\OneDrive\Escritorio\Ph.D\First Year\Fall 2024\Macro\Homework\Homework #5'
data = readtable('mydata.xlsx');

%% Data Preparation
% Calculate the GDP deflator
data.GDP_deflator = (data.GDPAtCurrentPrices_B__ ./ data.GDPAtConstantPrices_B__) * 100;
% Calculate real investment using the GDP deflator
data.real_investment  = (data.InvestmentAtCurrentPrices_B__ ./ data.GDP_deflator) * 100;

% Estimate parameter delta
p.delta = mean(data.consumptionOfFixedCapital_B__ ./ data.GDPAtConstantPrices_B__);

data.capital_stock = zeros(length(data.GDPAtConstantPrices_B__), 1);

% The initial capital stock is the first data point for the capital stock
data.capital_stock(1) = 28144.202; % Replace with your actual initial capital stock data if available
% Now, update the capital stock for each period
for t = 2:length(data.GDPAtConstantPrices_B__) % Start from the second period since the first is the initial value
    data.capital_stock(t) = (1 - p.delta) * data.capital_stock(t - 1) + data.real_investment(t - 1);
end


%% Question 1
% Initialize variables for results
variableNames = data.Properties.VariableNames;
numVars = numel(variableNames);

% Preallocate storage for RBC facts
stdDevs = zeros(numVars, 1);
correlations = zeros(numVars, 1);
prevCorrs = zeros(numVars, 5); % 5 previous periods
futureCorrs = zeros(numVars, 5); % 5 future periods

% Specify lambda for HP filter (1600 for quarterly data)
lambda = 1600;

% Loop through each variable to deseasonalize and process
for i = 1:numVars
    % Extract variable
    variable = data.(variableNames{i});
    
    % Skip if the variable is non-numeric (e.g., dates or categories)
    if ~isnumeric(variable) || any(isnan(variable))
        continue;
    end
    
    % Apply HP filter to remove trend
    [trend, cyclical] = hpfilter(variable, lambda);
    
    % Store the detrended series back into the dataset
    detrended_variable = variable - trend; 
    
    % Deseasonalize (remove seasonal component) - Using moving average
    % Adjust window size (e.g., 4 for quarterly data)
    window = 4; 
    deseasonalized_variable = detrended_variable - movmean(detrended_variable, window, 'omitnan');
    
    data.(['detrended_' variableNames{i}]) = deseasonalized_variable;
    
    % Compute RBC facts
    stdDevs(i) = std(deseasonalized_variable, 'omitnan'); % Standard deviation
    % Compute correlations for 5 previous periods (lags)
    for lag = 1:5
        lagged_var = [NaN(lag, 1); deseasonalized_variable(1:end-lag)]; % Lagged version
        prevCorrs(i, lag) = corr(deseasonalized_variable, lagged_var, 'Rows', 'complete');
    end
    correlations(i) = corr(deseasonalized_variable, data.detrended_GDPAtConstantPrices_B__, 'Rows', 'complete'); % Correlation with Output
    for lead = 1:5
        lead_var = [deseasonalized_variable(lead+1:end); NaN(lead, 1)]; % Lead version
        futureCorrs(i, lead) = corr(deseasonalized_variable, lead_var, 'Rows', 'complete');
    end
end
variableNames = data.Properties.VariableNames;
variableNames = variableNames(2:19);
stdDevs = stdDevs(2:19);
prevCorrs = prevCorrs(2:19,:);
correlations = correlations(2:19);
futureCorrs = futureCorrs(2:19,:);

% Create a table for RBC facts
rbc_facts = table(variableNames', stdDevs, prevCorrs(:,5),prevCorrs(:,4),prevCorrs(:,3),prevCorrs(:,2),prevCorrs(:,1), correlations,futureCorrs(:,1),futureCorrs(:,2),futureCorrs(:,3),futureCorrs(:,4),futureCorrs(:,5), ...
    'VariableNames', {'Variable', 'StdDev','x(-5)','x(-4)','x(-3)','x(-2)','x(-1)' ,'x', 'x(1)','x(2)','x(3)','x(4)','x(5)'});

% Display the results
disp(rbc_facts);

% Save detrended data and RBC facts to a new Excel file
writetable(data, 'processed_data.xlsx');
writetable(rbc_facts, 'rbc_facts.xlsx');

disp('Processed data and RBC facts saved as Excel files.');

%% Question 2 a)

%%Calibrate other parameters

%Estimating alpha
p.alpha  = mean(data.laborCompensation_B__)/mean(data.GDPAtConstantPrices_B__);

%Estimating psi
data.Z_t =  data.GDPAtConstantPrices_B__ ./ (data.capital_stock.^(p.alpha) .* data.HoursWorked.^(1-p.alpha));

for t = 1:(178-1)
    consumption_temp = data.GDPAtConstantPrices_B__(t) + (1-p.delta)*data.capital_stock(t) - data.capital_stock(t+1);
    data.consumption(t) = consumption_temp;
end
data.consumption(end) = NaN; % Set the last value to NaN

data.psi = ((1-p.alpha).*data.GDPAtConstantPrices_B__)./(data.consumption.*data.HoursWorked);
data.psi(end) = NaN; % Set the last value to NaN

p.psi = nanmean(data.psi);

%Estimating beta
upper_limit = 0.99;

for t = 1:(178-2)
    beta_temp = data.consumption(t+1) / (data.consumption(t) * (p.alpha * data.Z_t(t) * data.capital_stock(t)^(p.alpha-1) * data.HoursWorked(t)^(1-p.alpha) + (1 - p.delta)));
    
    % Apply the upper limit boundary condition
    if beta_temp > upper_limit
        beta_temp = upper_limit;
    end
    
    data.beta_values(t) = beta_temp;
end

% Calibrate p.beta by taking the average of the calculated values
p.beta = mean(data.beta_values);

% Calibrate z_t parameters
data.detrended_Z_t = real(data.detrended_GDPAtConstantPrices_B__ ./ (data.detrended_capital_stock.^(p.alpha) .* data.detrended_HoursWorked.^(1-p.alpha)));
p.rho = .1094536; %parameter estimated in stata
p.sigma = 19.66201 ; %parameter estimated in stata


%% Question 2 b)
[fl_ar1_persistence, fl_shk_std, it_disc_points, bl_verbose, it_std_bound] = ...
    deal(p.rho, p.sigma, 10, true, 3);
ffy_tauchen(fl_ar1_persistence, fl_shk_std, it_disc_points, bl_verbose,it_std_bound);

N=10;
it_std_bound = 3;

[z_grid, P] = ffy_tauchen(p.rho, p.sigma, N, true, it_std_bound);
writematrix(P, 'transition_matrix.xlsx');

%% Question 2 c)

% --- Step 1: Define Model Parameters ---
% From previous steps (calibrated AR(1) parameters)

sigma = 19.662010000000000; 
beta = 0.973871067644482; 
delta = 0.116984253257508; 
alpha = 0.362899982548457; 
psi = 0.009657400456219; 

% Extract z_min and z_max from the grid z_grid
z_min = min(z_grid);
z_max = max(z_grid);

% Capital grid
k_min = 0.5;        % Minimum capital
k_max = 5;          % Maximum capital
num_k = 100;        % Number of grid points
k_grid = linspace(k_min, k_max, num_k);

% Initialize value function
V = zeros(N, num_k);      % Value function for each z state and k
V_new = zeros(N, num_k);    % Placeholder for updates
policy_k = zeros(N, num_k); % Policy function for capital
policy_c = zeros(N, num_k); % Policy function for c_t

% Tolerance for convergence
tol = 1e-6;
diff = 1;

while diff > tol
    for iz = 1:N % Loop over z states
        z = z_grid(iz);
        for ik = 1:num_k % Loop over capital grid
            k = k_grid(ik);
            % Resource constraint to find feasible consumption
            feasible_k = k_grid; % Next period's capital
            feasible_c = z * k^alpha - feasible_k + (1 - delta) * k;

            % Labor policy function
            feasible_n = ((1 - alpha) * z * k^alpha ./ (psi * feasible_c)).^(1 / (1 - alpha));

            % Utility
            feasible_u = log(max(feasible_c, 1e-8)) - psi;

            % Output
            feasible_y = z * k.^alpha * feasible_n.^(1-alpha);

            % Bellman equation
            expected_value = P(iz, :) * V; % Expected continuation value
            total_value = feasible_u + beta * expected_value;

            % Optimal choice
            [V_new(iz, ik), idx_opt] = max(total_value);
            policy_k(iz, ik) = feasible_k(idx_opt);
            policy_c(iz, ik) = feasible_c(idx_opt);
            policy__n(iz, ik) = feasible_n(idx_opt);
            policy_y(iz, ik) = feasible_y(idx_opt);
        end
    end
    % Update difference for convergence
    diff = max(abs(V_new(:) - V(:)));
    V = V_new;
end

T = 20000; % Length of simulation

% Initialize simulation
z_sim = zeros(1, T); % Simulated z_t
z_sim(1) = z_grid(randi(N)); % Random initial state

% Simulate the Markov process
for t = 2:T
    current_state = find(z_grid == z_sim(t-1)); % Find current state's index
    next_state = randsample(N, 1, true, P(current_state, :)); % Draw next state
    z_sim(t) = z_grid(next_state); % Record next state
end

% Map simulated z_t to V
value_series = zeros(1, T); % Preallocate value series
for t = 1:T
    [~, closest_index] = min(abs(z_grid - z_sim(t))); % Closest grid point
    value_series(t) = V(closest_index); % Extract corresponding value
end

% Initialize simulation arrays

% Preallocate arrays for c_t, n_t, and k_t
c_sim = zeros(1, T); % Consumption
n_sim = zeros(1, T); % Labor
k_sim = zeros(1, T); % Capital
y_sim = zeros(1, T); % Output

for t = 1:T
    % Find the closest z_state to z_sim(t) in the coarse grid (z_grid)
    [~, closest_index] = min(abs(z_grid - z_sim(t))); % Find closest z_index in z_grid
    z_closest = z_grid(closest_index);  % Closest z value
   
    % Access the corresponding policy values from the feasible arrays
    c_sim(t) = feasible_c(closest_index);  % Consumption policy
    n_sim(t) = feasible_n(closest_index);  % Labor policy
    k_sim(t) = feasible_k(closest_index);  % Capital policy
    y_sim(t) = feasible_y(closest_index);  % Output policy
end

%Plots
% Time index for plotting
time = 1:T;

% Plot z_t
figure;
subplot(5,1,1);
plot(time, z_sim, 'b');
title('Time Series for z_t');
xlabel('Time');
ylabel('z_t');

% Plot c_t
subplot(5,1,2);
plot(time, c_sim, 'r');
title('Time Series for c_t');
xlabel('Time');
ylabel('c_t');

% Plot n_t
subplot(5,1,3);
plot(time, n_sim, 'g');
title('Time Series for n_t');
xlabel('Time');
ylabel('n_t');

% Plot k_t
subplot(5,1,4);
plot(time, k_sim, 'm');
title('Time Series for k_t');
xlabel('Time');
ylabel('k_t');
%ylim([0 5]);

% Plot y_t
subplot(5,1,5);
plot(time, y_sim, 'b');
title('Time Series for y_t');
xlabel('Time');
ylabel('y_t');


% Adjust layout for better visibility
sgtitle('Simulated Time Series');

%% Question 2 d)
T_discard = 10000;
z = z_sim(T_discard:end);
k = k_sim(T_discard:end);
c = c_sim(T_discard:end);
n = n_sim(T_discard:end);
y = y_sim(T_discard:end);
sim = table(z',k',c',n',y');
sim.Properties.VariableNames = {'z', 'k', 'c', 'n', 'y'};

% Initialize variables for results
variableNames = sim.Properties.VariableNames;
numVars = numel(variableNames);

% Preallocate storage for RBC facts
stdDevs = zeros(numVars, 1);
correlations = zeros(numVars, 1);
prevCorrs = zeros(numVars, 5); % 5 previous periods
futureCorrs = zeros(numVars, 5); % 5 future periods


% Loop through each variable to deseasonalize and process
for i = 1:numVars
    % Extract variable
    variable = sim.(variableNames{i});

    % Compute RBC facts
    stdDevs(i) = std(variable, 'omitnan'); % Standard deviation
    % Compute correlations for 5 previous periods (lags)
    for lag = 1:5
        lagged_var = [NaN(lag, 1); variable(1:end-lag)]; % Lagged version
        prevCorrs(i, lag) = corr(variable, lagged_var, 'Rows', 'complete');
    end
    correlations(i) = corr(variable, sim.y, 'Rows', 'complete'); % Correlation with Output
    for lead = 1:5
        lead_var = [variable(lead+1:end); NaN(lead, 1)]; % Lead version
        futureCorrs(i, lead) = corr(variable, lead_var, 'Rows', 'complete');
    end
end


% Create a table for RBC facts
rbc_facts_sim = table(variableNames', stdDevs, prevCorrs(:,5),prevCorrs(:,4),prevCorrs(:,3),prevCorrs(:,2),prevCorrs(:,1), correlations,futureCorrs(:,1),futureCorrs(:,2),futureCorrs(:,3),futureCorrs(:,4),futureCorrs(:,5), ...
    'VariableNames', {'Variable', 'StdDev','x(-5)','x(-4)','x(-3)','x(-2)','x(-1)' ,'x', 'x(1)','x(2)','x(3)','x(4)','x(5)'});

% Display the results
disp(rbc_facts_sim);
writetable(rbc_facts_sim, 'rbc_facts_sim.xlsx');

%% Question  3)
cd 'C:\dynare\6.2\matlab'
addpath("C:\dynare\6.2\matlab")
dynare rbc_hw