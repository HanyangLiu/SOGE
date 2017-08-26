% testing with k-means

clear all;
close all;

globals;

%% Preparation
% load data
data_dir = [base_dir 'Data/'];
load([data_dir data]);
X = Data;
Y = TransformL(gt, class);

% initialization
display('Initializing...');
[X0, ~, ~] = pcaInit(X, infRes);
[d, n] = size(X0);
c = size(Y, 2);

% Generate Laplacian matrix L and diagonal matrix U
IDX = genLabel( Y, para );
[ L, U ] = genLU( X0, Y, para, IDX );

%% Learn the projection matrix
[ W_final, obj, prop ] = RecSSGE( X0, Y, L, U, para );
X_final = W_final'*X0;

%% Evaluation
display('Evaluating...');
[~, gt] = max(Y, [], 2);
Time = 10;
result = zeros(Time, 3); result0 = zeros(Time, 3);
for i = 1:Time
    la = kmeans(X_final', c); la0 = kmeans(X0', c);
    result(i, 1:end) = ClusteringMeasure(gt, la); result0(i, 1:end) = ClusteringMeasure(gt, la0);
    Result = mean(result); Result0 = mean(result0);
end

figure('name', 'Convergence Curve');
plot(obj);
display(prop);