clc;
clear all;
close all;
%% Parameters
beta = 0.95;    % Discount factor
alpha = 0.3;    % Capital share in production
delta = 0.1;    % Depreciation rate
eta = 0.4;      % Past consumption weight
gamma = 2;      % Utility parameter

%% Grid Setup
k_ss = (alpha/((1/beta)-(1-alpha)))^(1/(1-alpha));
c_ss = k_ss^(alpha) - delta*k_ss;
k_min = 0.5*k_ss;    % Minimum capital
k_max = 1.5*k_ss;      % Maximum capital
num_k = 100;    % Number of grid points

k = linspace(k_min, k_max, num_k);  % Capital grid
k_ind = zeros(100,1);
c_ind = c_ss * ones(num_k, 1);
C = zeros(100,100); 
V = zeros(100,100);
V0 = zeros(100,1);
V1 = zeros(100,1);
policy_k = zeros(num_k, 1);  
policy_c = zeros(num_k, 1);  

%% Find C 

for i=1:100 %k
    for j=1:100 %k'
        C(i,j) = k(i)^alpha +(1-delta)*k(i)-k(j);
        if C(i,j) < 0 
            C(i,j) = -realmax;
        end
    end
end


%% Value Function Iteration 
tolerance = 1;
iter = 0;

while tolerance > 1e-4 && iter < 20000
    for i = 1:num_k
         for j = 1:num_k
         V(i,j) =(C(i,j)-eta*c_ind(i))^(gamma+1)/(gamma+1) +beta*V0(j);
         end
    [V1(i),k_ind(i)]=max(V(i,:));
    policy_k(i) = k(k_ind(i));
    policy_c(i) = C(i, k_ind(i));
    end
    c_ind = policy_c;
    tolerance = max(abs(V1 - V0));
    V0  = V1;
    iter = iter + 1;
    
end

%% Plots

fig1 = figure; 

% value function: 
subplot(1,3,1);
plot(k,V1);
xlabel('k');
ylabel('value function');
xlim([0,k(end)]);

% capital policy function:
subplot(1,3,2);
plot(k,policy_k);
xlabel('k');
ylabel('capital policy function');
xlim([0,k(end)]);

% consumption policy function:
subplot(1,3,3);
plot(k,policy_c);
xlabel('c');
ylabel('consumption policy function');
xlim([0,k(end)]);