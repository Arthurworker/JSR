function [ W ] = JSRW(X, S, P, alpha, lambda)

    
    %A = (beta + 1) * S*S';
    %[row, col] = size(A);   			% 
    temp = eye(28560, 30);   		% temp
    temp2 = eye(51, 51);
    M = 2*lambda*temp2;      	% A = 2*lambda
    
    K = (alpha + 1) * X*X';		%	B = (alpha+1)XX'
    Q = 2 * S*X' - (2*(alpha + 1)*X'*X*P)'*temp; 					% C = 2SX'
    W = sylvester(M,K,Q);
end

