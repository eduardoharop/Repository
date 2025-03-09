clc;
clear all;
close all;

%% Initialization 

% parameters
alpha = 0.3;
beta  = 0.99;
delta = 1;
g = 0.57;

% initializating tol, dif, and iteration
tol      = 1e-6;
max_iter = 10000;
dif      = 10;
iter     = 0;

% labor function
function n = find_labor(c_t, k_t,g,alpha)
  fun = @(n) c_t - (g * (1-n) * (1-alpha) * k_t^(alpha) * n^(-alpha))/(1-g);
  n = fsolve(fun, 0.5);  
end

% next-labor function
function n_next = labor_next(n_t, k_t, k_next,beta,alpha,delta)
  fun2 = @(n_next) (k_next^(alpha)*n_next^(-alpha)*(1-n_next)) - (1-n_t)*beta*k_t^(alpha)*n_t^(-alpha)*(alpha*k_next^(alpha-1)*n_next^(1-alpha)+(1-delta));
  n_next = fsolve(fun2, 0.5);  
end


% steady states
A = (1-beta.*(1-delta)-alpha.*delta.*beta)/(1-beta.*(1-delta)) .* (1-g)/(g.*(1-alpha));
n_star = 1./(A+1);
k_l = (alpha/((1/beta)-(1-delta)))^(1/(1-alpha));
k_star = k_l*n_star;
c_star = (k_l^(alpha)-delta.*k_l)*n_star;
y_star = k_l^(alpha).*n_star;

% initializing variables and time horizon
k0   = k_star/3; % Assume k0 is given
c_L  = 0;        % Lowest value of C
c_H  = k0^alpha; % Highest value of C
MaxT = 200;      % Max T is 200


%% 

while dif > tol && iter < max_iter
    k_t = k0;
    c_t = (c_L+c_H)/2;
    n_t = find_labor(c_t,k_t,g,alpha);
    y_t = k_t^alpha * n_t^(1-alpha); 
    % Forward pass
    Y      = zeros(MaxT+1,1);
    C      = zeros(MaxT+1,1);
    N      = zeros(MaxT+1,1);
    K      = zeros(MaxT+1,1);
    K_next = zeros(MaxT+1,1);
    T      = 1;
    while (T <= MaxT) && (dif > tol) && (k_t > 0)
        k_next      = k_t^(alpha)*n_t^(1-alpha) + (1-delta)*k_t - c_t; % Capital accumulation
        n_next      = labor_next(n_t,k_t,k_next,beta,alpha,delta); % Labor Euler Equation
        c_next      = c_t*beta*(alpha*k_next^(alpha-1)*n_next^(1-alpha)+(1-delta)); % Consumption Euler Equation
        y_next      = y_t *(n_next*(1-n_t)*beta*(alpha*k_next^(alpha-1)*n_next^(1-alpha)+(1-delta)))* (1/(n_t*(1-n_next))); % Output Euler Equation
        C(T,1)      = c_t;
        K(T,1)      = k_t;
        N(T,1)      = n_t;
        Y(T,1)      = y_t;
        K_next(T,1) = k_next;
        dif         = abs((k_next-k_t)/k_t); % relative change in capital from one period to the next
        sign        = k_next - k_t;
        % updating c, k, n, y, T
        c_t = c_next; 
        k_t = k_next; 
        n_t = n_next;
        y_t = y_next;
        T = T+1;
    end
    C      = C(1:T-1,1);
    K      = K(1:T-1,1);
    N      = N(1:T-1,1); 
    Y      = Y(1:T-1,1);
    K_next = K_next(1:T-1,1);
     
    % Update Bounds
    if sign > 0  % k_t+1 > k_t 
         % Sign>0 => it means that capital has increased suggesting not
         % enough has been consumed. Therefore, the lower bound for consumption
         % (c_L) is adjusted upward.
         c_L=(c_L+c_H)/2;
    else         % k_t+1 < k_t 
         % Sign<0 => it means that capital has decreased suggesting we
         % consumed too much. Therefore, the upper bound for consumption
         % (c_H) is adjusted downward.
         c_H=(c_L+c_H)/2;
    end
    
    % update MaxT

    if (K_next(end,1)-k_star) > 1/3 * K_next(end,1) % checking whether the capital at the last time step K1(end,1) is
                                                    % significantly different from the steady-state capital k_star.
         MaxT = MaxT - 1;
         dif = Inf;
    elseif (k_star-K_next(end,1)) > 1/3 * K_next(end,1)
         MaxT = MaxT + 1;
         dif = Inf;
    else
         MaxT = MaxT;
    end
    
iter = iter + 1;
end
%% Plotting 

T=size(C,1);
time=1:1:T;
CS=ones(T,1)*c_star;
KS=ones(T,1)*k_star;
YS=ones(T,1)*y_star;
NS=ones(T,1)*n_star;

figure;

subplot(1,4,1)
plot(time,K_next,time,KS)
title('Capital')


subplot(1,4,2)
plot(time,C,time,CS)
title('Consumption')

subplot(1,4,3)
plot(time,N,time,NS)
title('Labor')

subplot(1,4,4)
plot(time,Y,time,YS)
title('Output')
