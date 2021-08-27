function [ V ] = JSRV(X, S, beta, lambda)
    
    %A = (beta + 1) * S*S';
  
    %[row, col] = size(A);   			
    temp = eye(40, 40);   		
    M = lambda*temp;      	% A = SS'+lambda
    
    K = (beta + 1) * S*S';		%	B = (alpha+1)XX'
    Q = X*S'; 						% C = 2SX'
    V = sylvester(M,K,Q);
end

