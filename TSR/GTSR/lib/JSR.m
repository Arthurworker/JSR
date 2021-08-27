function [ W ] = JSR(X, S, P , alpha, beta, lambda)
        
    [row, col] = size(P);
    P = lambda*eye(row,col); 
    A = (beta + 1) * S*S';  
    [row, col] = size(A);   			
    temp = eye(row, col);   		
    A = A + lambda*temp;      	    
    B = (alpha + 1) * X*X';		%	B = (alpha+1)XX'
    C = 2 * S*X'; 						% C = 2SX'
    W = sylvester(A,B,C);
end

