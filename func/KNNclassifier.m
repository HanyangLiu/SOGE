function [ predict_label ] = KNNclassifier( train_data, train_label, test_data, k )
%UNTITLED10 Summary of this function goes here
%   Detailed explanation goes here

IDX = knnsearch(train_data, test_data, 'K', k);
LA = train_label(IDX);
predict_label = mode(LA, 2);

end

