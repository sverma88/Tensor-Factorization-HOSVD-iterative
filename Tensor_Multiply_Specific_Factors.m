% Function to Multiply Tensor in all Modes excluding specified Modes

function [Ten_A]=Tensor_Multiply_Specific_Factors(Tensor_A,Exclude_Modes,Singular_Factors_A)

%Input
% Tensor_A              : Tensor 'A' of Mode 'N'
% Exclude_Modes         : row vector specific modes in which tensor has not
%                         to be multipled with the factors
% Singular_Factors_A    : Singular Factors for each Modes of Tensor_A
%
%
% Output
% Ten_A                 : Tensor 'A' multiplied by Specific modes Factors
%
% Author                : Sunny Verma (sunnyverma.iitd@gmail.com)
% Last_Update           : 13/07/2016

%


Ten_A=Tensor_A;
iter_A=ndims(Tensor_A);

for j=1:iter_A
    
    if(~any(j==Exclude_Modes))
        
        Ten_A=ttm(Ten_A,Singular_Factors_A{j,1}',j);
    end
end

end