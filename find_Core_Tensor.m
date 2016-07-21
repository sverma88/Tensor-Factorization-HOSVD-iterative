% Function to find core tensor utilizing Factors Matrices of each Mode of 
% Tensor and the Tensor itself 

function [Core_Tensor]=find_Core_Tensor(Tensor_A,Singular_Factors_A)

% Inputs
% Tensor_A             : Tensor 'A' of Mode 'n' 
% Singular_Factors_A   : Left Singular values obtained by using svd on each
%                        of tensor (cell array or N*1 : in order with modes)
% 
% Outputs
% Core_Tensor          : Core Tensor of Tensor A (computed according to
%                        HOSVD)
% 
% Author               : Sunny Verma (sunnyverma.iitd@gmail.com)
% Last_Update          : 13/07/2016

% % 


iter=ndims(Tensor_A);
Core_Tensor=Tensor_A;
for i=1:iter
    Core_Tensor= ttm(Core_Tensor,Singular_Factors_A{i,1}',i);
end

end