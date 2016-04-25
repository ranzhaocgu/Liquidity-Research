function epsil = callPrice(Price, Strike, Rate, Time, Value, Vol)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    [~, p_p] = blsprice(Price, Strike, Rate, Time, Vol);
    epsil = (Value - p_p)^2;
end

