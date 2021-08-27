function [b, normF] = NormalizeRows(a, type)


% Default: L1
if nargin == 1
    type = 'L1';
end

switch type
    case 'L1'
        normF = sum(a,2); % Get sums
        normFUsed = normF;
        normFUsed(normFUsed == 0) = 1; % Prevent division by zero
        b = bsxfun(@rdivide, a, normFUsed); % Normalise
    case 'L2'
        normF = sqrt(sum(a .* a, 2)); % Get length
        normFUsed = normF;
        
        normFUsed(normFUsed == 0) = 1; % Prevent division by zero
        b = bsxfun(@rdivide, a, normFUsed);
    case 'Sqrt->L1'
        b = NormalizeRows(sqrt(a), 'L1');
        normF = [];
    case 'L1->Sqrt' % This is the same as ROOTSIFT
        b = sqrt(NormalizeRows(abs(a), 'L1'));
        normF = [];
    case 'Sqrt->L2'
        b = NormalizeRows(sqrt(a), 'L2');
        normF = [];
    case 'None'
        b = a;
        normF = [];
    otherwise 
        error('Wrong normalization type given');
end
