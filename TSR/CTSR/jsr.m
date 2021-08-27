%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CTSR  JSR  ZSL  DEMO
% 
% Hyperparameter: alpha, beta, lambda
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clc, clear all,  close all

addpath('lib')

load('CTSR.mat');

Y     = label_matrix(train_labels_tsr')';
W    = (X_tr'*X_tr + 50 * eye(size(X_tr'*X_tr)))^(-1)*X_tr'*Y;
X_tr = X_tr * W;
X_te = X_te * W;

alpha = 20;
beta = .001;
lambda  = 1;

S_tr    = NormalizeFea(S_tr);
W       = JSR(X_tr', S_tr', alpha, beta, lambda)';

%%%%% Testing %%%%%
dist    =  1 - zscore(pdist2(X_te, (S_te_pro * W'), 'cosine')) ;
dist    = zscore(dist);
HITK    = 1;
Y_hit5  = zeros(size(dist,1),HITK);
for i   = 1:size(dist,1)
    [sort_dist_i, I] = sort(dist(i,:),'descend');
    Y_hit5(i,:) = te_cl_id(I(1:HITK));
end

n=0;
for i  = 1:size(dist,1)
    if ismember(test_labels_tsr(i),Y_hit5(i,:))
        n = n + 1;
    end
end
zsl_accuracy = n/size(dist,1);
sv_acc = zsl_accuracy*100;
fprintf('\n CTSR ZSL accuracy: %.4f%%\n', sv_acc);

            
            









