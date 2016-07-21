% Function to Reconstruct Tensor, given its factors Matrices and core
% tensor

function [Reconstructed_Tensor]=Reconstruct_Tensor(Core_Tensor,Singular_Factors)

%Input
% Core_Tensor           : Core Tensor of the original Tensor obtained using coupled
%                         HOSVD
% Singular_Factors      : Singular Factors of the Tensor in cell array
% 
% Output
% Reconstructed_Tensor  : Tensor reconstructed using core tensor and
%                         singular factors
% 
% Author                : Sunny Verma (sunnyverma.iitd@gmail.com)
% Last_Update           : 13/07/2016

% % 

iter=ndims(Core_Tensor);

Reconstructed_Tensor=Core_Tensor;
for i=1:iter
    Reconstructed_Tensor= ttm(Reconstructed_Tensor,Singular_Factors{i,1},i);
end

end