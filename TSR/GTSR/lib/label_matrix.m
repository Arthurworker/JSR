function [ S ] = label_matrix(label_vector)

    
    Y      = label_vector;
    labels = unique(Y);
    [~,indexes] = ismember(Y,labels);
    rows        = 1:length(Y); %// row indx
    S           = zeros(length(Y),length(unique(indexes))); %// A matrix full of zeros
    S(sub2ind(size(S),rows ,indexes)) = 1; %// Ones at the desired row/column combinations
    S = S';
end

