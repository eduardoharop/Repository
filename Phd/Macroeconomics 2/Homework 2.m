%% Second degree Polynomial and 1-node MC (converges in 2559 iterations)
clc;
clear all;
close all;
%Parameters
alpha = 0.33;      % Capital share
beta = 0.96;      % Discount factor
delta = 0.1;      % Depreciation rate
gamma = 2;        % Risk aversion
rho = 0.7;        % Persistence of theta
sigma = 0.2;     % Volatility of shocks
T = 50000;         % Simulation length for PEA
xi = 0.4;         % Damping parameter
tol = 1e-6;       % Convergence tolerance
% Initialize productivity shocks (pre-compute)
rng(1);                           % Seed for reproducibility
eps = sigma * randn(T, 1);        % Shocks for PEA simulation
ln_theta = zeros(T, 1);           % Precompute ln(theta_t)
ln_theta(1) = log(1);             % Initial condition (steady state theta=1)
for t = 2:T
    ln_theta(t) = rho * ln_theta(t-1) + eps(t);
end
theta = exp(ln_theta);
kss= (alpha/((1/beta)-(1-delta)))^(1/(1 - alpha));  % Steady state
css= kss^alpha - delta*kss;
b0 = log(css^(-gamma)*(1-delta+alpha*(kss)^(alpha-1))); 
options = optimoptions('lsqnonlin', 'Display', 'off', 'Algorithm', 'levenberg-marquardt');
b = [b0; 0; 0; 0; 0; 0];  % Initial guess (adjust as needed)
%b = [0.5952; -0.812; -1.0858; 0; 0; 0];  % Initial guess (adjust as needed)

% Iterative PEA
converged = false;
iter = 0;
diff = 1;
while ~converged 
    % Step 1: Simulate capital and consumption
    k = zeros(T, 1);
    c = zeros(T, 1);
    k(1) = kss;
    
    for t = 1:T-1
        ln_k = log(k(t));
        ln_theta_t = ln_theta(t);
        % Compute Psi using second-degree polynomial
        Psi = b(1) + b(2)*ln_k + b(3)*ln_theta_t + b(4)*(ln_k)^2 + b(5)*ln_k*ln_theta_t + b(6)*(ln_theta_t)^2;
        c(t) = max((beta * exp(Psi))^(-1/gamma),1e-3);
        k(t+1) = max((1 - delta)*k(t) + theta(t)*k(t)^alpha - c(t),1e-3);
    end
    
    % Step 2: Compute expectations e_t (two-node integration)
    e = zeros(T-1, 1);
    for t = 1:T-1
        % Draw two shocks for two-node MC
        eps1 = sigma * randn;
        ln_theta_next1 = rho * ln_theta(t) + eps1;
        theta_next1 = exp(ln_theta_next1);
        
        % Compute next-period consumption (approximate)
        ln_k_next = log(k(t+1));
        Psi_next1 = b(1) + b(2)*ln_k_next + b(3)*ln_theta_next1 + ...
                    b(4)*(ln_k_next)^2 + b(5)*ln_k_next*ln_theta_next1 + b(6)*(ln_theta_next1)^2;
        c_next1 = max((beta * exp(Psi_next1))^(-1/gamma),1e-3);
        
        % Euler equation expectation
        RHS1 = (1 - delta) + alpha * theta_next1 * k(t+1)^(alpha - 1);
        e(t) =  max(beta * (c_next1)^(-gamma) * RHS1, 1e-6);
    end
    
    % Step 3: Update coefficients via NLLS
     X = [ones(T-1,1), log(k(1:T-1)), ln_theta(1:T-1), ...
         (log(k(1:T-1))).^2, log(k(1:T-1)).*ln_theta(1:T-1), (ln_theta(1:T-1)).^2];
    residual_fun = @(b) e - exp(X*b); % Residual vector for lsqnonlin
    b_new = real(lsqnonlin(residual_fun, b, [], [], options)); % Use lsqnonlin

    % Step 4: Dampen update
    b_update = (1 - xi) * b + xi * b_new;
    diff = norm(b_update - b)^2;
    fprintf('Iteration %d: Tolerance = %.6f\n', iter, diff);
    if norm(b_update - b)^2 < tol
        converged = true;
    end
    b = b_update;
    iter = iter+1;
end

% Residual evaluation (after convergence)
I = 10200; 
discard = 200;
eps_long = sigma * randn(I, 1);
ln_theta_long = zeros(I, 1);
ln_theta_long(1) = log(1);
for t = 2:I
    ln_theta_long(t) = rho * ln_theta_long(t-1) + eps_long(t);
end
theta_long = exp(ln_theta_long);

% Simulate capital/consumption
k_long = zeros(I, 1);
c_long = zeros(I, 1);
k_long(1) = k(1);  % Steady-state initial
for t = 1:I-1
    ln_k = log(k_long(t));
    ln_theta_t = ln_theta_long(t);
    Psi = b(1) + b(2)*ln_k + b(3)*ln_theta_t + ...
          b(4)*(ln_k)^2 + b(5)*ln_k*ln_theta_t + b(6)*(ln_theta_t)^2;
    c_long(t) = max((beta * exp(Psi))^(-1/gamma),1e-3);
    k_long(t+1) = max((1 - delta)*k_long(t) + theta_long(t)*k_long(t)^alpha - c_long(t),1e-3);
end

% Compute residuals
resid_BC = zeros(I, 1);
for t = 1:I-1
    resid_BC(t) = ((1 - delta)*k_long(t) + theta_long(t)*k_long(t)^alpha) / ...
           (c_long(t) + k_long(t+1)) - 1;    
end
resid_EE = zeros(I, 1);
for t = 1:I-1
    % Two-node MC for expectation
    eps1 = sigma * randn;
    eps2 = sigma * randn;
    ln_theta_next1 = rho * ln_theta_long(t) + eps1;
    ln_theta_next2 = rho * ln_theta_long(t) + eps2;
    theta_next1 = exp(ln_theta_next1);
    theta_next2 = exp(ln_theta_next2);
    
    % Next-period consumption
    ln_k_next = log(k_long(t+1));
    Psi_next1 = b(1) + b(2)*ln_k_next + b(3)*ln_theta_next1 + ...
                b(4)*(ln_k_next)^2 + b(5)*ln_k_next*ln_theta_next1 + b(6)*(ln_theta_next1)^2;
    c_next1 = max((beta * exp(Psi_next1))^(-1/gamma),1e-3);
    
    Psi_next2 = b(1) + b(2)*ln_k_next + b(3)*ln_theta_next2 + ...
                b(4)*(ln_k_next)^2 + b(5)*ln_k_next*ln_theta_next2 + b(6)*(ln_theta_next2)^2;
    c_next2 = max((beta * exp(Psi_next2))^(-1/gamma),1e-3);
    
    RHS1 = (1 - delta) + alpha * theta_next1 * k_long(t+1)^(alpha - 1);
    RHS2 = (1 - delta) + alpha * theta_next2 * k_long(t+1)^(alpha - 1);
    resid_EE(t) = beta * 0.5 * ( (c_next1/c_long(t))^(-gamma)*RHS1 + (c_next2/c_long(t))^(-gamma)*RHS2 ) - 1;
end

% Log residuals
mean_abs_resid_BC = log10(mean(abs(resid_BC(discard:I-1))));
max_abs_resid_BC = log10(max(abs(resid_BC(discard:I-1))));
mean_abs_resid_EE = log10(mean(abs(resid_EE(discard:I-1))));
max_abs_resid_EE = log10(max(abs(resid_EE(discard:I-1))));

fprintf('Mean |Resid BC| (log10): %.4f\n', mean_abs_resid_BC);
fprintf('Max |Resid BC| (log10): %.4f\n', max_abs_resid_BC);
fprintf('Mean |Resid EE| (log10): %.4f\n', mean_abs_resid_EE);
fprintf('Max |Resid EE| (log10): %.4f\n', max_abs_resid_EE);


%% First degree Polynomial and 2-node MC (converges at 608th iter)
clc;
clear all;
close all;
%Parameters
alpha = 0.33;      % Capital share
beta = 0.96;      % Discount factor
delta = 0.1;      % Depreciation rate
gamma = 2;        % Risk aversion
rho = 0.7;        % Persistence of theta
sigma = 0.2;     % Volatility of shocks
T = 20000;         % Simulation length for PEA
xi = 0.4;         % Damping parameter
tol = 1e-6;       % Convergence tolerance
% Initialize productivity shocks (pre-compute)
rng(1);                           % Seed for reproducibility
eps = sigma * randn(T, 1);        % Shocks for PEA simulation
ln_theta = zeros(T, 1);           % Precompute ln(theta_t)
ln_theta(1) = log(1);             % Initial condition (steady state theta=1)
for t = 2:T
    ln_theta(t) = rho * ln_theta(t-1) + eps(t);
end
theta = exp(ln_theta);
kss= (alpha/((1/beta)-(1-delta)))^(1/(1 - alpha));  % Steady state
css= kss^alpha - delta*kss;
b0 = log(css^(-gamma)*(1-delta+alpha*(kss)^(alpha-1))); 
options = optimoptions('lsqnonlin', 'Display', 'off', 'Algorithm', 'levenberg-marquardt');
b = [b0; 0; 0];  % Initial guess (adjust as needed)
%b = [0.69; -0.9514; -0.8466];  % Initial guess (adjust as needed)

% Iterative PEA
converged = false;
iter = 0;
diff = 1;
while ~converged 
    % Step 1: Simulate capital and consumption
    k = zeros(T, 1);
    c = zeros(T, 1);
    k(1) = kss;
    
    for t = 1:T-1
        ln_k = log(k(t));
        ln_theta_t = ln_theta(t);
        % Compute Psi using second-degree polynomial
        Psi = b(1) + b(2)*ln_k + b(3)*ln_theta_t;
        c(t) = max((beta * exp(Psi))^(-1/gamma),1e-3);
        k(t+1) = max((1 - delta)*k(t) + theta(t)*k(t)^alpha - c(t),1e-3);
    end
    
    % Step 2: Compute expectations e_t (two-node integration)
    e = zeros(T-1, 1);
    for t = 1:T-1
        % Draw two shocks for two-node MC
        eps1 = sigma * randn;
        eps2 = sigma * randn;
        ln_theta_next1 = rho * ln_theta(t) + eps1;
        ln_theta_next2 = rho * ln_theta(t) + eps2;
        theta_next1 = exp(ln_theta_next1);
        theta_next2 = exp(ln_theta_next2);
        
        % Compute next-period consumption (approximate)
        ln_k_next = log(k(t+1));
        Psi_next1 = b(1) + b(2)*ln_k_next + b(3)*ln_theta_next1;
        c_next1 = max((beta * exp(Psi_next1))^(-1/gamma),1e-3);
        
        Psi_next2 = b(1) + b(2)*ln_k_next + b(3)*ln_theta_next2;
        c_next2 = max((beta * exp(Psi_next2))^(-1/gamma),1e-3);
        
        % Euler equation expectation
        RHS1 = (1 - delta) + alpha * theta_next1 * k(t+1)^(alpha - 1);
        RHS2 = (1 - delta) + alpha * theta_next2 * k(t+1)^(alpha - 1);
        e(t) =  max(beta * 0.5 * ((c_next1)^(-gamma) * RHS1 + (c_next2)^(-gamma) * RHS2 ), 1e-6);
    end
    
    % Step 3: Update coefficients via NLLS
     X = [ones(T-1,1), log(k(1:T-1)), ln_theta(1:T-1)];
    residual_fun = @(b) e - exp(X*b); % Residual vector for lsqnonlin
    b_new = real(lsqnonlin(residual_fun, b, [], [], options)); % Use lsqnonlin

    % Step 4: Dampen update
    b_update = (1 - xi) * b + xi * b_new;
    diff = norm(b_update - b)^2;
    fprintf('Iteration %d: Tolerance = %.6f\n', iter, diff);
    if norm(b_update - b)^2 < tol
        converged = true;
    end
    b = b_update;
    iter = iter+1;
end

% Residual evaluation (after convergence)
I = 10200; 
discard = 200;
eps_long = sigma * randn(I, 1);
ln_theta_long = zeros(I, 1);
ln_theta_long(1) = log(1);
for t = 2:I
    ln_theta_long(t) = rho * ln_theta_long(t-1) + eps_long(t);
end
theta_long = exp(ln_theta_long);

% Simulate capital/consumption
k_long = zeros(I, 1);
c_long = zeros(I, 1);
k_long(1) = k(1);  % Steady-state initial
for t = 1:I-1
    ln_k = log(k_long(t));
    ln_theta_t = ln_theta_long(t);
    Psi = b(1) + b(2)*ln_k + b(3)*ln_theta_t;
    c_long(t) = (beta * exp(Psi))^(-1/gamma);
    k_long(t+1) = (1 - delta)*k_long(t) + theta_long(t)*k_long(t)^alpha - c_long(t);
end

% Compute residuals
resid_BC = zeros(I, 1);
for t = 1:I-1
    resid_BC(t) = ((1 - delta)*k_long(t) + theta_long(t)*k_long(t)^alpha) / ...
           (c_long(t) + k_long(t+1)) - 1;    
end
resid_EE = zeros(I, 1);
for t = 1:I-1
    % Two-node MC for expectation
    eps1 = sigma * randn;
    eps2 = sigma * randn;
    ln_theta_next1 = rho * ln_theta_long(t) + eps1;
    ln_theta_next2 = rho * ln_theta_long(t) + eps2;
    theta_next1 = exp(ln_theta_next1);
    theta_next2 = exp(ln_theta_next2);
    
    % Next-period consumption
    ln_k_next = log(k_long(t+1));
    Psi_next1 = b(1) + b(2)*ln_k_next + b(3)*ln_theta_next1;
    c_next1 = (beta * exp(Psi_next1))^(-1/gamma);
    
    Psi_next2 = b(1) + b(2)*ln_k_next + b(3)*ln_theta_next2;
    c_next2 = (beta * exp(Psi_next2))^(-1/gamma);
    
    RHS1 = (1 - delta) + alpha * theta_next1 * k_long(t+1)^(alpha - 1);
    RHS2 = (1 - delta) + alpha * theta_next2 * k_long(t+1)^(alpha - 1);
    resid_EE(t) = beta * 0.5 * ( (c_next1/c_long(t))^(-gamma)*RHS1 + (c_next2/c_long(t))^(-gamma)*RHS2 ) - 1;
end

% Log residuals
mean_abs_resid_BC = log10(mean(abs(resid_BC(discard:I-1))));
max_abs_resid_BC = log10(max(abs(resid_BC(discard:I-1))));
mean_abs_resid_EE = log10(mean(abs(resid_EE(discard:I-1))));
max_abs_resid_EE = log10(max(abs(resid_EE(discard:I-1))));

fprintf('Mean |Resid BC| (log10): %.4f\n', mean_abs_resid_BC);
fprintf('Max |Resid BC| (log10): %.4f\n', max_abs_resid_BC);
fprintf('Mean |Resid EE| (log10): %.4f\n', mean_abs_resid_EE);
fprintf('Max |Resid EE| (log10): %.4f\n', max_abs_resid_EE);



%% Second degree Polynomial and 2-node MC (converges at 959th iteration)
clc;
clear all;
close all;
%Parameters
alpha = 0.33;      % Capital share
beta = 0.96;      % Discount factor
delta = 0.1;      % Depreciation rate
gamma = 2;        % Risk aversion
rho = 0.7;        % Persistence of theta
sigma = 0.2;     % Volatility of shocks
T = 50000;         % Simulation length for PEA
xi = 0.5;         % Damping parameter
tol = 1e-6;       % Convergence tolerance
% Initialize productivity shocks (pre-compute)
rng(1);                           % Seed for reproducibility
eps = sigma * randn(T, 1);        % Shocks for PEA simulation
ln_theta = zeros(T, 1);           % Precompute ln(theta_t)
ln_theta(1) = log(1);             % Initial condition (steady state theta=1)
for t = 2:T
    ln_theta(t) = rho * ln_theta(t-1) + eps(t);
end
theta = exp(ln_theta);
kss= (alpha/((1/beta)-(1-delta)))^(1/(1 - alpha));  % Steady state
css= kss^alpha - delta*kss;
b0 = log(css^(-gamma)*(1-delta+alpha*(kss)^(alpha-1))); 
options = optimoptions('lsqnonlin', 'Display', 'off', 'Algorithm', 'levenberg-marquardt');
b = [b0; 0; 0; 0; 0; 0];  % Initial guess (adjust as needed)


% Iterative PEA
converged = false;
iter = 0;
diff = 1;
while ~converged 
    % Step 1: Simulate capital and consumption
    k = zeros(T, 1);
    c = zeros(T, 1);
    k(1) = kss;
    
    for t = 1:T-1
        ln_k = log(k(t));
        ln_theta_t = ln_theta(t);
        % Compute Psi using second-degree polynomial
        Psi = b(1) + b(2)*ln_k + b(3)*ln_theta_t + b(4)*(ln_k)^2 + b(5)*ln_k*ln_theta_t + b(6)*(ln_theta_t)^2;
        c(t) = max((beta * exp(Psi))^(-1/gamma),1e-3);
        k(t+1) = max((1 - delta)*k(t) + theta(t)*k(t)^alpha - c(t),1e-3);
    end
    
    % Step 2: Compute expectations e_t (two-node integration)
    e = zeros(T-1, 1);
    for t = 1:T-1
        % Draw two shocks for two-node MC
        eps1 = sigma * randn;
        eps2 = sigma * randn;
        ln_theta_next1 = rho * ln_theta(t) + eps1;
        ln_theta_next2 = rho * ln_theta(t) + eps2;
        theta_next1 = exp(ln_theta_next1);
        theta_next2 = exp(ln_theta_next2);
        
        % Compute next-period consumption (approximate)
        ln_k_next = log(k(t+1));
        Psi_next1 = b(1) + b(2)*ln_k_next + b(3)*ln_theta_next1 + ...
                    b(4)*(ln_k_next)^2 + b(5)*ln_k_next*ln_theta_next1 + b(6)*(ln_theta_next1)^2;
        c_next1 = max((beta * exp(Psi_next1))^(-1/gamma),1e-3);
        
        Psi_next2 = b(1) + b(2)*ln_k_next + b(3)*ln_theta_next2 + ...
                    b(4)*(ln_k_next)^2 + b(5)*ln_k_next*ln_theta_next2 + b(6)*(ln_theta_next2)^2;
        c_next2 = max((beta * exp(Psi_next2))^(-1/gamma),1e-3);
        
        % Euler equation expectation
        RHS1 = (1 - delta) + alpha * theta_next1 * k(t+1)^(alpha - 1);
        RHS2 = (1 - delta) + alpha * theta_next2 * k(t+1)^(alpha - 1);
        e(t) =  max(beta * 0.5 * ((c_next1)^(-gamma) * RHS1 + (c_next2)^(-gamma) * RHS2 ),1e-6);
    end
    
    % Step 3: Update coefficients via NLLS
     X = [ones(T-1,1), log(k(1:T-1)), ln_theta(1:T-1), ...
         (log(k(1:T-1))).^2, log(k(1:T-1)).*ln_theta(1:T-1), (ln_theta(1:T-1)).^2];
    residual_fun = @(b) e - exp(X*b); % Residual vector for lsqnonlin
    b_new = real(lsqnonlin(residual_fun, b, [], [], options)); % Use lsqnonlin

    % Step 4: Dampen update
    b_update = (1 - xi) * b + xi * b_new;
    diff = norm(b_update - b)^2;
    fprintf('Iteration %d: Tolerance = %.6f\n', iter, diff);
    if norm(b_update - b)^2 < tol
        converged = true;
    end
    b = b_update;
    iter = iter+1;
    xi = max(0.1, 0.8 / (1 + iter/500));
end

% Residual evaluation (after convergence)
I = 10200; 
discard = 200;
eps_long = sigma * randn(I, 1);
ln_theta_long = zeros(I, 1);
ln_theta_long(1) = log(1);
for t = 2:I
    ln_theta_long(t) = rho * ln_theta_long(t-1) + eps_long(t);
end
theta_long = exp(ln_theta_long);

% Simulate capital/consumption
k_long = zeros(I, 1);
c_long = zeros(I, 1);
k_long(1) = k(1);  % Steady-state initial
for t = 1:I-1
    ln_k = log(k_long(t));
    ln_theta_t = ln_theta_long(t);
    Psi = b(1) + b(2)*ln_k + b(3)*ln_theta_t + ...
          b(4)*(ln_k)^2 + b(5)*ln_k*ln_theta_t + b(6)*(ln_theta_t)^2;
    c_long(t) = max((beta * exp(Psi))^(-1/gamma),1e-3);
    k_long(t+1) = max((1 - delta)*k_long(t) + theta_long(t)*k_long(t)^alpha - c_long(t),1e-3);
end

% Compute residuals
resid_BC = zeros(I, 1);
for t = 1:I-1
    resid_BC(t) = ((1 - delta)*k_long(t) + theta_long(t)*k_long(t)^alpha) / ...
           (c_long(t) + k_long(t+1)) - 1;    
end
resid_EE = zeros(I, 1);
for t = 1:I-1
    % Two-node MC for expectation
    eps1 = sigma * randn;
    eps2 = sigma * randn;
    ln_theta_next1 = rho * ln_theta_long(t) + eps1;
    ln_theta_next2 = rho * ln_theta_long(t) + eps2;
    theta_next1 = exp(ln_theta_next1);
    theta_next2 = exp(ln_theta_next2);
    
    % Next-period consumption
    ln_k_next = log(k_long(t+1));
    Psi_next1 = b(1) + b(2)*ln_k_next + b(3)*ln_theta_next1 + ...
                b(4)*(ln_k_next)^2 + b(5)*ln_k_next*ln_theta_next1 + b(6)*(ln_theta_next1)^2;
    c_next1 = max((beta * exp(Psi_next1))^(-1/gamma),1e-3);
    
    Psi_next2 = b(1) + b(2)*ln_k_next + b(3)*ln_theta_next2 + ...
                b(4)*(ln_k_next)^2 + b(5)*ln_k_next*ln_theta_next2 + b(6)*(ln_theta_next2)^2;
    c_next2 = max((beta * exp(Psi_next2))^(-1/gamma),1e-3);
    
    RHS1 = (1 - delta) + alpha * theta_next1 * k_long(t+1)^(alpha - 1);
    RHS2 = (1 - delta) + alpha * theta_next2 * k_long(t+1)^(alpha - 1);
    resid_EE(t) = beta * 0.5 * ( (c_next1/c_long(t))^(-gamma)*RHS1 + (c_next2/c_long(t))^(-gamma)*RHS2 ) - 1;
end

% Log residuals
mean_abs_resid_BC = log10(mean(abs(resid_BC(discard:I-1))));
max_abs_resid_BC = log10(max(abs(resid_BC(discard:I-1))));
mean_abs_resid_EE = log10(mean(abs(resid_EE(discard:I-1))));
max_abs_resid_EE = log10(max(abs(resid_EE(discard:I-1))));

fprintf('Mean |Resid BC| (log10): %.4f\n', mean_abs_resid_BC);
fprintf('Max |Resid BC| (log10): %.4f\n', max_abs_resid_BC);
fprintf('Mean |Resid EE| (log10): %.4f\n', mean_abs_resid_EE);
fprintf('Max |Resid EE| (log10): %.4f\n', max_abs_resid_EE);
