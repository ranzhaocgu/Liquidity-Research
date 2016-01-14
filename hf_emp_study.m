%% Empirical study framework using high frequency data
% high frequency data contains milliseconds buy/sell shock orders

%% Data processing
company = 'AAPL'; date = '20110401';  % later functionalize

filename = strcat(company, '_', date, '.xlsx');
[num,str] = xlsread(filename);

% the name of each item of the data matrix
title = str(1,:);
str = str(2:end,:);

% clean the raw data
% delete all 'delete' and 'modified' orders
num = num(strcmp(str(:, strcmp(title, 'Stock_Selected_Type')), 'A'),:);
str = str(strcmp(str(:, strcmp(title, 'Stock_Selected_Type')), 'A'),:);

% delete all NA data -- price
num = num(~isnan(num(:,strcmp(title, 'Stock_Selected_Price'))),:);
str = str(~isnan(num(:,strcmp(title, 'Stock_Selected_Price'))),:);

% delete all NA data -- shares
num = num(~isnan(num(:,strcmp(title, 'Stock_Selected_Shares'))),:);
str = str(~isnan(num(:,strcmp(title, 'Stock_Selected_Shares'))),:);

% order specification 
%order_type = str(:, strcmp(title, 'Stock_Selected_Type'));
buy_or_sell = str(:, strcmp(title, 'Stock_Selected_BuySell'));

% define the time
second = num(:,strcmp(title, 'Stock_Selected_Seconds')); 
millisecond = num(:,strcmp(title, 'Stock_Selected_Milliseconds'));

% define the price/shares
price = num(:,strcmp(title, 'Stock_Selected_Price')); 
shares = num(:,strcmp(title, 'Stock_Selected_Shares')); 


%% Market Data Calibration
% time steps
time_step_minute = 15;
start_time = min(second); end_time = max(second);
total_time_steps = round((end_time - start_time) / (time_step_minute*60));

% initialize Q
Q_0 = zeros(1, total_time_steps);
Q_S = zeros(1, total_time_steps);
eta = zeros(1, total_time_steps);

% initialize q
training_set_end = 8 * (60 / time_step_minute);
min_price = quantile(price(1:training_set_end*time_step_minute*60), 0.01);
max_price = quantile(price(1:training_set_end*time_step_minute*60), 0.99);
price_step = (max_price-min_price)/100;
price_range = min_price:price_step:max_price;
q = zeros(length(price_range), total_time_steps);

% initialize Q
Q_profile = zeros(length(price_range), total_time_steps);

% define f(p)
%f_p = @(p) sqrt(p*(max_price-p));
f_p = zeros(length(price_range),1);

% initialize h(f(p),t)
h = zeros(length(price_range), total_time_steps);

% calibration of the real time data
for i = 1:total_time_steps  
    if i == 1  % include the starting second of the trading day
        real_time_seconds = start_time + [((i-1)*time_step_minute*60):(i*time_step_minute*60)];
    else
        real_time_seconds = start_time + [((i-1)*time_step_minute*60+1):(i*time_step_minute*60)];
    end
    
    % calculate excess demand Q
    Q_0(i) = Q(price(ismember(second, real_time_seconds)), ...
        shares(ismember(second, real_time_seconds)), ...
        buy_or_sell(ismember(second, real_time_seconds)), 0);
    
    Q_S(i) = Q(price(ismember(second, real_time_seconds)), ...
        shares(ismember(second, real_time_seconds)), ...
        buy_or_sell(ismember(second, real_time_seconds)), 'S');
    
    % calculate Q and q
    for j = 1:length(price_range)
        Q_profile(j,i) =  Q(price(ismember(second, real_time_seconds)), ...
            shares(ismember(second, real_time_seconds)), ...
            buy_or_sell(ismember(second, real_time_seconds)), price_range(j));
        if j > 1
            q(j,i) = -(Q_profile(j,i) - Q_profile((j-1),i)); 
            
            if q(j,i) == 0
                temp1 = 1;
            else
                temp1 = q(j,i);
            end

            if q(j-1,i) == 0
                temp2 = 1;
            else
                temp2 = q(j-1,i);
            end
            h(j,i) = log(temp1) - log(temp2);
        end
    end
    
    eta(i) = Q_0(i) / (Q_0(i) + Q_S(i));
    if isnan(eta(i)) 
        eta(i) = 0;
    end
    
    % simulate Q for the out-of-sample testing
    % 1. simulate the Brownian sheet
    
%     normal_rand_nums = randn(length(price_range), time_Rtep_minute*60);
end

% plot the Q (net demand) from 9:15 AM to 5 PM of the trading day
figure
surf(1:size(Q_profile,2), price_range,Q_profile,'EdgeColor','none')
clear title;
title('Actual Net Demand Q over Price and Time');
xlabel('time');
ylabel('price');
zlabel('net demand Q');

% variance-covariance matrix of h and eta, Q measure
var_matrix_h_eta = cov([h(2:end,:);eta]');
corr_matrix_h_eta = corr([h(2:end,:);eta]');

sigma_eta_square = var(eta);

% adjust the correlation matrix
corr_matrix_h_eta(isnan(corr_matrix_h_eta)) = 0;
corr_matrix = corr_matrix_h_eta;

simulate_time_steps = 32;
omega = 3;

eta_sim_init = 1;
eta_sim = zeros(simulate_time_steps,omega);
a_eta = 0.9;

h_sim = zeros(size(h,1)-1,simulate_time_steps,omega);
q_sim = zeros(size(h,1)-1,simulate_time_steps,omega);
Q_sim = zeros(size(h,1)-1,simulate_time_steps,omega);

% simulate the results into next pre-defined periods
for sim_sce = 1:omega
    normal_random_numbers = randn((size(h,1)-1)+1, simulate_time_steps);

    Brownian_sheets_sim = corr_matrix*normal_random_numbers;
    Brownian_sheets_h_sim = cumsum(Brownian_sheets_sim(1:end-1,1:end));
    Brownian_sheets_eta_sim = Brownian_sheets_sim(end,:);
%     Brownian_sheets_h_sim = Brownian_sheets_sim(1:end-1,1:end);

    % annualize the timestep
    dt = time_step_minute/60;
    eta_sim(1,sim_sce) = eta_sim_init + a_eta*(mean(eta) - eta_sim_init)* dt ...
                + sqrt(sigma_eta_square)*sqrt(eta_sim_init*(1-eta_sim_init))*Brownian_sheets_eta_sim(1);
            
    for i = 2:simulate_time_steps
        eta_sim(i,sim_sce) = eta_sim(i-1,sim_sce) + a_eta*(mean(eta) - eta_sim(i-1,sim_sce))* dt...
            + sqrt(sigma_eta_square)*sqrt(eta_sim_init*(1-eta_sim_init))*Brownian_sheets_eta_sim(i)*sqrt(dt) ...
            * corr_matrix_h_eta(end-1,end);
        for j = 2:size(h_sim,1)
            h_sim(j,i,sim_sce) = h_sim(j-1,i,sim_sce) + sqrt(var_matrix_h_eta(j,j))*corr_matrix(j,1:end-1)*Brownian_sheets_h_sim(:,i);
        end
    end
    
    % adjustment of h_sim
    h_sim(:,:,sim_sce) = h_sim(:,:,sim_sce)/1000;
%      h_sim(h_sim <= 0.7*max(max(h_sim)) & h_sim >= 0.7*min(min(h_sim))) = 0;
% 
%     h_sim(1:round(0.4*size(h_sim,1)),:,sim_sce) = 0;
%     h_sim(round(0.6*size(h_sim,1)):end,:,sim_sce) = 0;


    for i = 1:simulate_time_steps
        for j = 1:size(h_sim,1)
            q_sim(j,i,sim_sce) = exp(sum(h_sim(1:j,i,sim_sce)));
        end
    end
    
    % simulation of Q
    for i = 1:simulate_time_steps
        for j = 1:size(h_sim,1)
            Q_sim(j,i,sim_sce) = sum(q_sim(:,i,sim_sce))*eta_sim(i,sim_sce) - sum(q_sim(1:j,i,sim_sce));
        end
    end
end

figure
surf(1:simulate_time_steps,price_range(2:end),Q_sim(:,:,1),'EdgeColor','none');
clear title;
title('Simulated Net Demand Q over Price and Time');
xlabel('time');
ylabel('price');
zlabel('net demand Q');


%% Market price of risk equation

atm_index = zeros(simulate_time_steps, omega);
atm_price = zeros(simulate_time_steps, omega);
Sigma = zeros(size(h_sim,1),size(h_sim,1),simulate_time_steps, omega);
dQ_sim = (Q_sim(:,2:end,:) - Q_sim(:,1:end-1,:))./Q_sim(:,1:end-1,:);

% the calculation of SIGMA 
A = zeros(size(h_sim,1), size(h_sim,1), simulate_time_steps, omega);
B = zeros(size(h_sim,1), size(h_sim,1), simulate_time_steps, omega);
C = zeros(size(h_sim,1), size(h_sim,1), simulate_time_steps, omega);

for each_sce = 1:omega % outer loop for the scenarios
    for xxx = 1:simulate_time_steps
        atm_index(xxx,each_sce) = find(Q_sim(:, xxx) <= 0, 1);
        atm_price(xxx,each_sce) = price_range(atm_index(xxx,each_sce));
    end

    % calculate sigma and b's
    corr_dQ_sim = corr(dQ_sim(:,:,each_sce));

    % calculation the sigma matrix
    % Sigma(pi, s, t, omega)
    for i = 1:simulate_time_steps  % t loop
        dh_t = (h_sim(2:end,i,each_sce) - h_sim(1:end-1,i,each_sce))./h_sim(1:end-1,i,each_sce);
        dh_t(1,:) = 0; dh_t(:,1) = 0;
        sigma_h_x_b_h_x_s = corr(dh_t');
        sigma_h_x_b_h_x_s(isnan(sigma_h_x_b_h_x_s)) = 0;
        
        for j = 1:(size(h_sim,1)-1)    % pi loop
            for z = 1:(size(h_sim,1)-1)  % s loop
                temp = h_sim(2:end,i,each_sce) * sigma_h_x_b_h_x_s(:,z)';
                A(j,z,i,each_sce) = sum(exp(sum(h_sim(:,i,each_sce)))*temp(:,z))*eta_sim(i,each_sce);

                B(j,z,i,each_sce) = (Q_sim(1,i,each_sce) + sum(q_sim(:,i,each_sce))*...
                    corr_matrix_h_eta(end,end)*sqrt(eta_sim(i,each_sce)*(1-eta_sim(i,each_sce)))) ...
                    * corr_matrix_h_eta(end-1,end);
                
                C(j,z,i,each_sce) = sum(exp(sum(h_sim(:,i,each_sce)))*temp(1:j,z));
            end
        end
    end
    
    Sigma(:,:,:,each_sce) = A(:,:,:,each_sce) + B(:,:,:,each_sce) + C(:,:,:,each_sce);
end

C_pi_t = zeros(size(h_sim,1),simulate_time_steps, omega);
B_pi_t = zeros(size(h_sim,1),simulate_time_steps, omega);
 
for each_sce = 1:omega 
    for t = 1:simulate_time_steps
        C_pi_t(2:end,t,each_sce) = -sum((Sigma(2:end,:,t,each_sce) - ...
            Sigma(1:end-1,:,t,each_sce))./ Sigma(1:end-1,:,t,each_sce),2);
        B_pi_t(2:end,t,each_sce) = C_pi_t(2:end,t,each_sce) - 0.5;
    end
end



% for each_sce = 1:omega 
%     for xxx = 1:simulate_time_steps
%         
%     end
% end