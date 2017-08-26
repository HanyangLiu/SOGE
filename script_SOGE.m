% test with kNN

clear all;
close all;

globals;

%% Preprocessing

ACC_un = [];
ACC_te = [];

    
% denoise and split data
data_dir = [base_dir 'Data/'];
load([data_dir data]);
[X, ~, ~] = pcaInit(Data, infRes);                                                       % dimension reduction (denoise)
[X_tr, gt_tr, X_te, gt_te] = splitData(X, gt, dataName, 0.5);                 % split data into training set and testing set
genLap( X_tr, gt_tr, para);
%     genLap2( X_tr );

load([base_dir 'Data/split/' dataName '_split.mat']);
load([tmp_dir 'Laplassian']);

Y_tr = TransformL(gt_tr, class);

% select data with labels among training set
IDX = genLabel( Y_tr, para );                             
X_la = X_tr(1:end, IDX); 
gt_la = gt_tr(IDX);                                                                             % generate data with labels     

X_un = X_tr; X_un(:, IDX) = [];
gt_un = gt_tr; gt_un(IDX) = [];                                                          % generate data without labels          

% Generatediagonal matrix U
diagU = zeros(numel(gt_tr), 1);
diagU(IDX) = 1;
U = sparse(para.uu*diag(diagU));
T_seen = sparse(U*Y_tr);
X_seen = X_tr;

%% Processing

% generate the projection matrix W for dimension reduction
[ W, obj, prop ] = SOGE( X_seen, T_seen, L, U, para );

% dimension reduction for labeled data, unlabeled data and test set data
X_laR = W'*X_la;
X_unR = W'*X_un;
X_teR = W'*X_te;

%% Testing with Classification

la_un = knnclassify(X_unR', X_laR', gt_la, 1);
acc_un = length(find(la_un == gt_un))/length(gt_un)*100;
ACC_un = [ACC_un, acc_un];

la_te = knnclassify(X_teR', X_laR', gt_la, 1);
acc_te = length(find(la_te == gt_te))/length(gt_te)*100;
ACC_te = [ACC_te, acc_te];


figure('name', 'Convergence Curve');
plot(obj(1:500),'LineWidth', 2);
display(prop);
set(gca,'LineWidth', 2);

meanACC_un = mean(ACC_un); stdACC_un = std(ACC_un);
meanACC_te = mean(ACC_te); stdACC_te = std(ACC_te);

fprintf(1,'ACC - unlabeled = %f (%f)\n', meanACC_un, stdACC_un);
fprintf(1,'ACC - testing = %f (%f)\n', meanACC_te, stdACC_te);





