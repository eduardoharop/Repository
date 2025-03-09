clc;
clear all;
close all;

%% Initialize

% Parameters
p.alpha = 0.33;
p.beta  = 0.95;
p.delta  = 0.1;
z_L   = 9;
z_H   = 11;
A     = [0.8 0.2; 0.2 0.8];

k = 0.01:0.01:10; % Grid for state variable, capital stock
Lk    = length(k);
z     = [z_L, z_H];
Lz    = length(z);

% Initializing value function matrices
k_ind = zeros(Lz, Lk); 
c     = zeros(Lz,Lk,Lk);
V     = zeros(Lz, Lk, Lk);
V0    = zeros(Lz, Lk);
V1    = zeros(Lz, Lk);

%% Find C and
for z_ind = 1:Lz %states
   for  i = 1:Lk %k
    for j = 1:Lk %k'
        c(z_ind,i,j) = z(z_ind)*10*k(i)^p.alpha-k(j)+(1-p.delta)*k(i);
        if c(z_ind,i,j) < 0
            c(z_ind,i,j) = realmin;
        end
    end
   end
end

%% Value Function Iteration Question a
tol  = 1;
iter = 0;

while tol > 1e-4
    EV = A*V0;
    for z_ind = 1:Lz % states
        for i = 1:Lk % capital today
            for j = 1:Lk % capital tmrw
                V(z_ind, i, j) = log(c(z_ind, i, j)) + p.beta * EV(z_ind, j);
            end
            [V1(z_ind,i),k_ind(z_ind,i)]=max(V(z_ind,i,:));
        end
    end  
    % Check for convergence
    tol = norm(V1 - V0);
    V0  = V1;
    iter = iter + 1;
end

%% Policy Functions  

kp = zeros(Lz, Lk);
c  = zeros(Lz, Lk);
for z_ind = 1:Lz
    kp(z_ind,:) = k(k_ind(z_ind,:));
    c(z_ind,:) = z(z_ind) * k.^p.alpha - kp(z_ind,:) +(1-p.delta)*k;
    if c(z_ind,:) < 0
        c(z_ind,:) = realmin;
    end 
end

%% Plot Value and Policy Functions Question b

fig1 = figure; 

subplot(1,3,1)
hold on
grid on
box on
plot(k, V1(1,:), 'LineWidth', 2)
plot(k, V1(2,:), 'LineWidth', 2)
legend('z_L', 'z_H', 'Location', 'northwest')
title('Value Function')
xlim([0, max(k)])

subplot(1,3,2)
hold on
grid on
box on
plot(k, c(1,:), 'LineWidth', 2)
plot(k, c(2,:), 'LineWidth', 2)
legend('z_L', 'z_H', 'Location', 'northwest')
title('Consumption Policy Function')
xlim([0, max(k)])

subplot(1,3,3)
hold on
grid on
box on
plot(k, kp(1,:), 'LineWidth', 2)
plot(k, kp(2,:), 'LineWidth', 2)
legend('z_L', 'z_H', 'Location', 'northwest')
title('Capital Policy Function')
xlim([0, max(k)])

% Save the figure
saveas(fig1, 'Value_and_Policy_Functions.png');

%% MCMC Simulation for Steady-State Distribution Question c

% Set initial values
k_t        = zeros(1, 20000);   % Extended to allow a better distribution
z_index    = ones(1, 20000);    % To store productivity states
k_t(1)     = 1;             % Initial capital stock
z_index(1) = 1;                % Initial productivity state
rand_nums  = zeros(1, 20000);   % To store generated random numbers

% Simulate the Markov chain
for t = 2:20000
    % Calculate the index for the current capital stock
    k_index = find(abs(k - k_t(t-1))<1e-6);

   % Update capital stock using the policy function
    k_t(t) = kp(z_index(t-1),k_index);
    
    % Generate a random number for state transition
    rand_nums(t) = rand;
    
% Determine the current productivity state index
if z_index(t-1) == 1
    if rand_nums(t) <= 0.8
        z_index(t) = 1; % Stay in z_L
    else
        z_index(t) = 2; % Switch to z_H
    end
elseif z_index(t-1) == 2
    if rand_nums(t) <= 0.8
        z_index(t) = 2; % Stay in z_H
    else
        z_index(t) = 1; % Switch to z_L
    end
end
end
% Plot the histogram of capital stock after burn-in period
fig2 = figure;
histogram(k_t(10001:end), 20); % Adjusted bin size for better visualization
title('Steady State Distribution of Capital');
xlabel('Capital Stock');
ylabel('Frequency');

% Save the histogram figure
saveas(fig2, 'Steady_State_Distribution_of_Capital.png');
