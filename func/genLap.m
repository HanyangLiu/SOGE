function genLap( X, gt, para )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
globals;
display('Generating Laplacian Matrix...');
c = numel(unique(gt));
A0 = constructW_PKN(X, para.nn, 0);
[~, S, evs, cs] = CLR(A0, c);
S = (S+S')/2;
D = diag(sum(S));
L = D-S;
save ([tmp_dir 'Laplassian.mat'], 'L');

end

