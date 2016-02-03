function s_v = smooth_vector(v)

n = length(v);
non_zero_index = find(v ~= 0);

if isempty(non_zero_index)
    s_v = v;
elseif length(non_zero_index) == 1
    s_v = v(non_zero_index) / n * ones(size(v));
else
    s_v = v;
    for i = 1:length(non_zero_index)
        if i == 1
            s_v(1:non_zero_index(i)) = v(non_zero_index(i)) / length(1:non_zero_index(i)) * ones(1,length(1:non_zero_index(i)));
        elseif i == length(non_zero_index)
            s_v(non_zero_index(i):end) = v(non_zero_index(i)) / length(non_zero_index(i):length(v)) ...
                * ones(1,length(non_zero_index(i):length(v)));
            s_v(non_zero_index(i-1):non_zero_index(i)) = (v(non_zero_index(i)) + v(non_zero_index(i-1))) / 2 ...
                * ones(1,length(non_zero_index(i-1):non_zero_index(i)));
        else
            s_v(non_zero_index(i-1):non_zero_index(i)) = (v(non_zero_index(i)) + v(non_zero_index(i-1))) / 2 ...
                * ones(1,length(non_zero_index(i-1):non_zero_index(i)));
        end
    end
end

s_v(non_zero_index) = v(non_zero_index);

end

