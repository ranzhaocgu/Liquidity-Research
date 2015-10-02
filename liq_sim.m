%% A No-arbitrage Model of Liquidity in Financial Markets invovling Brownian Sheets
% Simulation work
% Ran Zhao @ 2015-09-02

%% Define the variables
time_T = 10;   dt = 0.1;   n_time_step = time_T/dt + 1; 
state_S = 20;   ds = 0.1;   n_state_s = state_S/ds + 1; 

% Number of scenarios
N = 1000;

d_price = ds;

sigma_eta = 1;
eta_bar = 0.5;
a_eta = 0.3;
D = 1;

h = ones(n_state_s, n_time_step);
q = zeros(n_state_s, n_time_step);
Q = zeros(n_state_s, n_time_step);
E_dQ = zeros(n_state_s-1, n_time_step);
eta = zeros(n_state_s, n_time_step);
%% Define the functions
sigma_h = @(p) 1;
% b_h = @(p,s) 1./(2.^(p/2).*2.^(s/2));
b_h = eye(n_state_s, n_state_s) + 0.1 * randn(n_state_s, n_state_s);
for i = 1:n_state_s
    b_h(i,:) = b_h(i,:) / norm(b_h(i,:),2);
end

b_eta = @(s) 1./(2.^(s/2));
% f_p = @(p,t) exp(-(p-state_S/2)^2/(2*D) * t);
% f_p = @(p) -((p-state_S/2)/state_S)^2;
f_p = @(p) sqrt(p*(state_S-p));
%% Main Simulation work
for iter = 1:N;

if (mod(iter,10) == 0)
    sprintf('Iteration of %d with total iterations of %d', iter, N)
end
    
rand_norm_matrix = randn(n_state_s, n_time_step);    % s by t matrix with normal rands

for p_step = 1:n_state_s
    price = d_price * (p_step - 1);
%     eta_zeros = rand(1);
%     eta_zeros = 0.5;
    eta(p_step,1) = 0.5;
    for time = 2:n_time_step
        h(p_step, time) = h(p_step, time-1)+sigma_h(price) * sum(b_h(p_step,:) * rand_norm_matrix(:,time)*sqrt(ds*dt));
        eta(p_step,time) = eta(p_step, time-1) + a_eta * (eta_bar - eta(p_step,time))*dt + sqrt(eta(p_step,time) * (1-eta(p_step,time))) ... 
                     * sigma_eta * sum(b_eta(1:(n_state_s)) * rand_norm_matrix(:,time) *sqrt(ds*dt));
        
        if eta(p_step,time) > 1
            eta(p_step,time) = 1;
        end
        
        if eta(p_step,time) < 0
            eta(p_step,time) = 0;
        end
        
%         disp(eta(p_step,time))
        index = find(0:ds:state_S <= f_p(price),1,'last');
        q(p_step, time) = sum(h(1:index,time));
        
    end
end

end

for ii = 1:n_state_s
    for jj = 2:n_time_step
        Q(ii, jj) = sum(q(:,jj).*eta(:,jj)) - sum(q(1:ii, jj));
    end
end

dQ = Q(2:end,:) - Q(1:(end-1),:);
E_dQ = E_dQ + dQ;
E_dQ = E_dQ / N;

rand_x_ind = round(dt*(n_time_step-1));
rand_y_ind = round(ds*(state_S-1));
% Plot E[Q(p,t)-Q(p-1),t]
figure
subplot(1,3,1)
surf(2:size(E_dQ,2),1:size(E_dQ,1),E_dQ(:,2:end),'EdgeColor','none')
xlabel('time'); ylabel('price'); zlabel('E[Q(p,t)-Q(p-1),t]');
subplot(1,3,2)
plot(E_dQ(rand_x_ind,2:end));
xlabel('time'); ylabel('E[Q(p,t)-Q(p-1),t]');
subplot(1,3,3)
plot(E_dQ(:,rand_y_ind));
xlabel('price'); ylabel('E[Q(p,t)-Q(p-1),t]');

figure
subplot(1,3,1)
surf(1:size(h,2),1:size(h,1),h,'EdgeColor','none');
xlabel('price'); ylabel('time'); zlabel('h');
subplot(1,3,2)
plot(h(rand_x_ind,:));
xlabel('time'); ylabel('h');
subplot(1,3,3)
plot(h(:,rand_y_ind));
xlabel('price'); ylabel('h');

% subplot(1,3,3)
% plot(h(50,:))
% plot(q(50,:))
% plot(Q(50,:))
%hist(mean(dQ));
%plot(sort(mean(dQ)));
