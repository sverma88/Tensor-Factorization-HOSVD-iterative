
% Function to reconstruct uncoupled Modes of the Tensor

function [S_FactorsA]=Reconstruct_Uncoupled_Modes_Tensor(Tensor_A,Singular_Factors_A,Rank_A,Coupled_Modes_A,S_Factors_A)

%Input
% Tensor_A              : Tensor 'A' of Mode 'N'
% Rank                  : Rank reduction in SVD
% Coupled_Modes_A       : index to the coupled modes of Tensor_A
% S_Factors_A           : Singular Factors of the Coupled Modes
%                         reconstrcuted Earlier
% Singular_Factors_A    : Singular Factors of Tensor 'A' in cell array
%                         format
%
% Output
% SFactorsA             : Singular Factors of the UnCoupled Modes
% Author                : Sunny Verma (sunnyverma.iitd@gmail.com)
% Last_Update           : 06/07/2016

%


iter_A=ndims(Tensor_A);

if nargin < 5
    Coupled_Modes_A=[];
    S_Factors_A=cell(iter_A,1);
end

S_FactorsA=S_Factors_A;

parfor i=1:iter_A
    
    if(~any(i==Coupled_Modes_A))
        [Ten_A]=Tensor_Multiply_Specific_Factors(Tensor_A,i,Singular_Factors_A);
        Reconstrcued_Mode_I=tenmat(Ten_A,i);
        [U,~,~]=svds((Reconstrcued_Mode_I.data),Rank_A(1,i));
        S_FactorsA{i,1}=(U);
    end
    
end

end