function  [X_tr, gt_tr, X_te, gt_te] = splitData(Data, gt, dataName, rho)
%splitData Summary of this function goes here
%   rho is the ratio of training data by all data

        globals;
        
%         [~, n] = size(Data);
%         n_tr = floor(rho*n);
%         ord = randperm(n);
%         X_tr = Data(:, ord(1:n_tr)); gt_tr = gt(ord(1:n_tr));
%         X_te = Data(:, ord(n_tr+1:end)); gt_te = gt(ord(n_tr+1:end));
%         save([base_dir 'Data/split/' dataName '_split.mat'], 'X_tr', 'gt_tr', 'X_te', 'gt_te');
        
        [~, n] = size(Data);
        c = numel(unique(gt));
        Y = TransformL(gt, c);
        Nn = sum(Y);
        IDtr = [];
        for kk = 1:c
            nn_tr = floor(rho*Nn(kk));
            id_k = find(gt==kk);
            order = randperm(numel(id_k));
            idtr_k = id_k(order(1:nn_tr));
            IDtr = [IDtr; idtr_k];
        end
        X_tr = Data(:, IDtr);
        gt_tr = gt(IDtr);
        X_te = Data; X_te(:, IDtr) = [];
        gt_te = gt; gt_te(IDtr) = [];
        save([base_dir 'Data/split/' dataName '_split.mat'], 'X_tr', 'gt_tr', 'X_te', 'gt_te');


end