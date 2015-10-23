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
time_step_minute = 5;
start_time = min(second); end_time = max(second);
total_time_steps = (end_time - start_time) / (time_step_minute*60);

% initialize Q
Q_0 = zeros(1, total_time_steps);
Q_S = zeros(1, total_time_steps);
eta = zeros(1, total_time_steps);

% initialize q
training_set_end = 84;
price_step = 1;
% min_price = round(min(price(1:training_set_end*time_step_minute*60)));
% max_price = round(max(price(1:training_set_end*time_step_minute*60)));
min_price = 340;
max_price = 360;
price_range = min_price:0.05:max_price;
q = zeros(length(price_range), total_time_steps);

% initialize Q
Q_profile = zeros(length(price_range), total_time_steps);

% define f(p)
%f_p = @(p) sqrt(p*(max_price-p));
f_p = zeros(length(price_range),1);

% for i = 1:total_time_steps
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
            q(j,i) = Q_profile(j,i) - Q_profile((j-1),i); 
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

% f_p = mean(q(:,1:training_set_end),2);
% x = [min_price, 0.5*(min_price+max_price), max_price]';
% y = [min_price, max_price, min_price]';
% fit_f_p = fit(x,y,'poly2');
% coeff_f_p = coeffvalues(fit_f_p);
% coeff_const = coeff_f_p(3);

surf(63:156, price_range,Q_profile(:,63:156),'EdgeColor','none')

% initialize h(f(p),t)
h = zeros(length(price_range), total_time_steps);

% change h to be 
%   q(p,t) = exp(sum h(x,t))
%   h(p,t) = ln(q(p,t)) - ln(q(p-1),t)

for i = 1:total_time_steps
    for j = 2:length(price_range)
        if q(j,i) == 0
            temp1 = 0;
        else
            temp1 = q(j,i);
        end
        
        if q(j-1,i) == 0
            temp2 = 0;
        else
            temp2 = q(j,i);
        end
        
        h(j,i) = log(temp1) - log(temp2);
    end
end

% for i = 1:total_time_steps
%     for j = 1:length(price_range)
%         coeff_f_p(3) = coeff_const - price_range(j);
%         inverse_f_p = roots(coeff_f_p);
%         f_left = min(inverse_f_p);
%         f_right = max(inverse_f_p);
%         
%         tmp_left = abs(price_range - f_left);
%         [~, idx_left] = min(tmp_left);
%         
%         tmp_right = abs(price_range - f_right);
%         [~, idx_right] = min(tmp_right);
%         % calculate h
%         try
%             h(j,i) = 0.5*((q(idx_left,i) - q((idx_left-1),i))*(2*coeff_f_p(1)*price_range(idx_left) + coeff_f_p(2))) + ...
%                  0.5*((q(idx_right,i) - q((idx_right-1),i))*(2*coeff_f_p(1)*price_range(idx_right) + coeff_f_p(2)));
%         catch
%             h(j,i) = 0;
%         end
%      end
% end

% scale the h to have around 1 h's
% h = 1 + 0.2 * h/(max(max(h)) - min(min(h)));

var_matrix_h_eta = zeros(size(h,1), size(h,1));
corr_matrix_h_eta = zeros(size(h,1), size(h,1));

% calculate the variance-covariance matrix
for xx1 = 1:(size(h,1)-1)
    for xx2 = 1:(size(h,1)-1)
        var_matrix_h_eta(xx1, xx2) = (1/(total_time_steps*time_step_minute/60)) ...
            * sum(log(h(xx1,2:end)./h(xx1,1:(end-1))).* ...
            log(h(xx1,2:end)./h(xx2,1:(end-1)))) ...
            / (time_step_minute/60);
    end
end

eta = 0.05 + 0.90*eta;
sigma_eta_square = (1/(total_time_steps*time_step_minute/60)) * ...
    sum(((eta(2:end)-eta(1:end-1)).^2)./(eta(1:end-1).*(1-eta(1:end-1))));

for i = 1:size(h,1)
    var_matrix_h_eta(i,end) = (1/(length(h)-1)) * ...
        sum(log(h(xx1,2:end)./h(xx1,1:(end-1))).* ...
        ((eta(2:end)-eta(1:end-1)).^2)./(eta(1:end-1).*(1-eta(1:end-1))));
end

var_matrix_h_eta(end,1:end-1) = var_matrix_h_eta(1:end-1,end);
var_matrix_h_eta(end,end) = sqrt(sigma_eta_square);

% floor the varianc matrix
% var_matrix_h_eta(var_matrix_h_eta <0) = 0;

% calculate the correlation matrix
for yy1 = 1:(size(h,1)-1)
    for yy2 = 1:(size(h,1)-1)
        corr_matrix_h_eta(yy1, yy2) = var_matrix_h_eta(yy1,yy2) / ...
            (sqrt(var_matrix_h_eta(yy1,yy1)) * sqrt(var_matrix_h_eta(yy2,yy2)));
    end
end

for j = 1:(size(h,1)-1)
    corr_matrix_h_eta(j,end) = var_matrix_h_eta(j,end) / (sqrt(var_matrix_h_eta(j,j))*sqrt(sigma_eta_square));
end

% clean the correlation matrix 
corr_matrix_h_eta(isnan(corr_matrix_h_eta)) = 0; 
corr_matrix_h_eta(isinf(corr_matrix_h_eta)) = 0; 
corr_matrix_h_eta = corr_matrix_h_eta/max(max(corr_matrix_h_eta));

for j = 1:size(h,1)
    corr_matrix_h_eta(j,j) = 1;
end

% simulate Q
B_h = chol(corr_matrix_h_eta(1:end-1,1:end-1));
corr_matrix = chol(corr_matrix_h_eta);

simulate_time_steps = 60;
normal_random_numbers = randn((size(h,1)-1)+1, simulate_time_steps);

Brownian_sheets_sim = corr_matrix*normal_random_numbers;
Brownian_sheets_h_sim = cumsum(Brownian_sheets_sim(1:end-1,1:end-1));
Brownian_sheets_eta_sim = Brownian_sheets_sim(1:end,end);

eta_sim_init = 0.5;
eta_sim = zeros(1,simulate_time_steps);
a_eta = 0.5;

for i = 1:simulate_time_steps
    if i == 1
        eta_sim(i) = eta_sim_init + a_eta*(mean(eta) - eta_sim_init)*(time_step_minute/60)...
            + sqrt(sigma_eta_square)*sqrt(eta_sim_init*(1-eta_sim_init))*Brownian_sheets_eta_sim(i);
    else
        eta_sim(i) = eta_sim(i-1) + a_eta*(mean(eta) - eta_sim(i-1))*(time_step_minute/60)...
            + sqrt(sigma_eta_square)*sqrt(eta_sim_init*(1-eta_sim_init))*Brownian_sheets_eta_sim(i)*sqrt(time_step_minute/60) ...
            * corr_matrix_h_eta(end-1,end);
    end
    
    
end



% figure
% plot(eta)
% %title('Training set eta_t');
% xlabel('t'); ylabel('eta');
% 
% figure
% subplot(1,3,1)
% plot(q(:,round(training_set_end/3)));
% xlabel('p'); ylabel('q at 1/3 training set');
% subplot(1,3,2)
% plot(q(:,round(training_set_end*2/3)));
% xlabel('p'); ylabel('q at 2/3 training set');
% subplot(1,3,3)
% plot(q(:,training_set_end));
% xlabel('p'); ylabel('q at training set end');

% out_of_sample_set = training_set_end + 1;
% for i = out_of_sample_set  % just find a interim time point 
%     real_time_seconds = start_time + [((i-1)*time_step_minute*60+1):(i*time_step_minute*60)];
% 
%     % calculate excess demand Q 
%     Q_0(i) = Q(price(ismember(second, real_time_seconds)), ...
%         shares(ismember(second, real_time_seconds)), ...
%         buy_or_sell(ismember(second, real_time_seconds)), 0);
%     
%     try
%         Q_S(i) = Q(price(ismember(second, real_time_seconds)), ...
%             shares(ismember(second, real_time_seconds)), ...
%             buy_or_sell(ismember(second, real_time_seconds)), 'S');
%     catch
%         Q_S(i) = 0;
%     end
%     
%     % calculate excess demand Q profile
%     for xx = 1:length(price_range)
%         Q_profile(xx) = Q(price(ismember(second, real_time_seconds)), ...
%             shares(ismember(second, real_time_seconds)), ...
%             buy_or_sell(ismember(second, real_time_seconds)), price_range(xx));
%     end
%     
%     % calculate q
%     for j = 2:size(q,1)
%         temp1 = Q(price(ismember(second, real_time_seconds)), ...
%             shares(ismember(second, real_time_seconds)), ...
%             buy_or_sell(ismember(second, real_time_seconds)), price_range(j-1));
%         temp2 = Q(price(ismember(second, real_time_seconds)), ...
%             shares(ismember(second, real_time_seconds)), ...
%             buy_or_sell(ismember(second, real_time_seconds)), price_range(j));
%         q(j,i) = temp2 - temp1; 
%     end
%     
%     eta(i) = Q_0(i) / (Q_0(i) + Q_S(i));
%     if isnan(eta(i)) 
%         eta(i) = 0;
%     end
% end
% 
