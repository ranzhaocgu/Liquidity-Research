%% Empirical study framework using high frequency data
% high frequency data contains milliseconds buy/sell shock orders

%% Data processing
company = 'GE'; date = '20110401';  % later functionalize

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
min_price = 10;
max_price = 30;
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

f_p = mean(q(:,1:training_set_end),2);
x = [min_price, 0.5*(min_price+max_price), max_price]';
y = [min_price, max_price, min_price]';
fit_f_p = fit(x,y,'poly2');
coeff_f_p = coeffvalues(fit_f_p);
coeff_const = coeff_f_p(3);

surf(63:156, price_range,Q_profile(:,63:156),'EdgeColor','none')

% initialize h(f(p),t)
h = zeros(length(price_range), total_time_steps);

for i = 1:total_time_steps
    for j = 1:length(price_range)
        coeff_f_p(3) = coeff_const - price_range(j);
        inverse_f_p = roots(coeff_f_p);
        f_left = min(inverse_f_p);
        f_right = max(inverse_f_p);
        
        tmp_left = abs(price_range - f_left);
        [~, idx_left] = min(tmp_left);
        
        tmp_right = abs(price_range - f_right);
        [~, idx_right] = min(tmp_right);
        % calculate h
        try
            h(j,i) = 0.5*((q(idx_left,i) - q((idx_left-1),i))*(2*coeff_f_p(1)*price_range(idx_left) + coeff_f_p(2)*price_range(idx_left))) + ...
                 0.5*((q(idx_right,i) - q((idx_right-1),i))*(2*coeff_f_p(1)*price_range(idx_right) + coeff_f_p(2)*price_range(idx_right)));
        catch
            h(j,i) = 0;
        end
     end
end

dim_mat_h = size(h,2)-1;

var_matrix_h = zeros(dim_mat_h, dim_mat_h);
corr_matrix_h = zeros(dim_mat_h, dim_mat_h);

% calculate the variance-covariance matrix
for xx1 = 1:dim_mat_h
    for xx2 = 1:dim_mat_h
        var_matrix_h(xx1, xx2) = (1/(total_time_steps*time_step_minute/60)) ...
            * sum(log(h(xx1,2:end)./h(xx1,1:(end-1)))*log(h(xx1,2:end)./h(xx2,1:(end-1)))) ...
            / (time_step_minute/60);
    end
end

% calculate the correlation matrix
for yy1 = 1:dim_mat_h
    for yy2 = 1:dim_mat_h
        corr_matrix_h(yy1, yy2) = var_matrix_h(yy1,yy2)^2 / ...
            (var_matrix_h(yy1,yy1) * var_matrix_h(yy2,yy2));
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
