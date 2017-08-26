%% Global Variables

% Directory setting
base_dir = '/Users/ericliu1991/Documents/MY RESEARCH/[project] Recursive Semi-Supervised Graph Embedding/[release] SOGE/';
tmp_dir = [base_dir 'tmp/'];

% Parameter setting
infRes = 0.95;    % the percentage of information reserved of the data during PCA dimension reduction
data = 'AT&T';
para = struct;
para.alpha = 10^0;
para.uu = 100;    % weight of the diagonal matrix U
para.nn = 10;    % block size of affinity matrix
para.p = 3;   % the number of data points with labels per class
para.K = 1;                % the number of recursive loops

save([tmp_dir 'param.mat']);