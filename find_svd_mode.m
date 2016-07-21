% Function to find left singular vectors of a matrix

function [Singular_factor]=find_svd_mode(Tensor_A,Mode,Rank)

% Inputs
% Tensor_A          : Tensor A of mode N
% Mode              : Specifying the matricization Mode of tensor A
% Rank              : Rank reduction in SVD
% 
% Outputs
% Singular_factor   : Singular factor of Tensor_A matricized in Mode 'I'
% 
% Author            : Sunny Verma (sunnyverma.iitd@gmail.com)
% Last_Update       : 11/07/2016

% % 

Matricize_Tensor=(tenmat(Tensor_A,Mode));

[U,~,~]=svds((Matricize_Tensor.data),Rank);

Singular_factor=(U);
    
end
    