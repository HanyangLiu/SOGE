function [ W_final, obj, prop ] = SOGE( X, T, L, U, para )
%SOGE Recursively projecting the data
% X: each colomn is a data point
% L: Laplacian matrix
% T: n*c matrix, class indicator matrix. Tij=1 if xi is labeled as j, Tij=0 otherwise
% para: parameters
%       para.alpha   trade-off parameter alpha
%       para.uu        weight of the diagonal matrix U
%       para.nn        block size of affinity matrix
%       para.p          the number of data points with labels per class
%       para.K          the number of recursive loops

% W_final: the final projection matrix

[~,n] = size(X);
H = eye(n) - 1/n*ones(n); X = X*H; % centering the data

W_final = [];
for Recu = 1: para.K
    [ W, ~, ~, obj, prop ] = Optim( X, T, L, U, para );
    X = X-W*W'*X;
    W_final = [W_final W];
end

end



function [ W, F, b, obj, prop ] = Optim( X, Y, L, U, para )
%Optim Optimization of the objective function
% b: bias vector
% F: soft label matrix

[d,n] = size(X);
r = para.alpha;
Hc = eye(n)-(1/n)*ones(n,n);
P = eye(n)/(L+U+r*Hc);

M = r^2*X*Hc*(1/r*eye(n)-P)*Hc*X';
N = -r*X*Hc*P*U*Y; 

lam = max(eig(M));
A = lam*eye(d)-M;
B = -N;

[W, obj] = GPI(A, B, 1);
F = P*(U*Y+r*X'*W);
b = 1/n*(F'*ones(n,1)-W'*X*ones(n,1));

prop = struct;
prop.GraEmb = trace(F'*L*F);
prop.SupVis = trace((F-Y)'*U*(F-Y));
prop.LinReg = norm(X'*W+ones(n,1)*b'-F,'fro');

end

function [X, obj] = GPI(A, B, r)
% max_{X'*X=I}  trace(X'*A*X) + 2*r*trace(X'*B)
% A must be positive semi-definite

NITER = 500;
[n,m] = size(B);
X = orth(rand(n,m));

for iter = 1:NITER
    M = A*X + r*B;
    [U,~,V] = svd(M,'econ');
    X = U*V';
    
    obj(iter,1) = trace(X'*A*X) + 2*r*trace(X'*B);
    display(iter);
end

end
