function [Q_output] = Q(price, shares, order, p)
%Q: the excess demand of the stock

    data_matrix = [price, shares];
    buy_orders_data = data_matrix(strcmp(order, 'B'),:);
    buy_orders_data = sortrows(buy_orders_data, 1);
    
    sell_orders_data = data_matrix(strcmp(order, 'S'),:);
    sell_orders_data = sortrows(sell_orders_data, 1);
    
    if p == 0
        Q_output = sum(buy_orders_data(:,2));
    elseif strcmp(p, 'S')
%         i = 1;
%         total_demand = 0; 
%         while i <= length(buy_orders_data)
%             total_demand = total_demand + buy_orders_data(i,2);
%             total_supply = ...
%                 sum(sell_orders_data(sell_orders_data(:,1) <= buy_orders_data(i,2),2));
%             Q_output = total_demand;
%             if total_supply > total_demand
%                 break
%             end
%             i = i + 1;
%         end
        Q_output = sum(sell_orders_data(:,2));
    else
        total_demand = sum(buy_orders_data(buy_orders_data(:,1) >= p, 2));
        total_supply = sum(sell_orders_data(sell_orders_data(:,1) <= p, 2));
        Q_output = total_demand - total_supply;
    end
end

