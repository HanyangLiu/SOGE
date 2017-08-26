function [ IDX ] = genLabel( Y, para )
%genLabel Generate the data with labels
%   Detailed explanation goes here
[n, c] = size(Y);
IDX = [];
for kk = 1:c
    y_k = Y(1:n, kk);
    idx = find(y_k);
    order = randperm(numel(idx));
    id_k = idx(order(1:para.p));
    IDX = [IDX; id_k];
end

end

