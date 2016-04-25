function epsil = putPrice(Price, Strike, Rate, Time, Value, Vol)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    [c_p, ~] = blsprice(Price, Strike, Rate, Time, Vol);
    epsil = (Value - c_p)^2;
end

