%% Function to find iterative HOSVD of Single Tensor

function [Core_Tensor_A,Singular_Factors_A,Singular_FA]=Decompose_Tensor_Coupled_HOSVD_iteratively(Tensor_A,Rank_A,Error_Threshold,Max_iterations)

%Input
% Tensor_A              : Tensor 'A' of Mode 'N' 
% Rank_A                : Rank reduction in SVD
% Error_Threshold       : Allowable error tolerance limit
% Max_iterations        : Maximum allowable iterations
%
%
% Output
% Core_Tensor_A         : Core Tensor of Tensor 'A' obtained using coupled
%                         iterative HOSVD
% Singular_Factors_A    : Singular Factors of Tensor 'A' in cell array
%                         format
% Singular_FA           : Singular Factors of Tensor A using Coupled HOSVD 
%                         without iteration
%
% Author                : Sunny Verma (sunnyverma.iitd@gmail.com)
% Last_Update           : 11/07/2016

%%

% Decomposing tensor HOSVD first time only independent modes are decomposed

% Tensor A
[Singular_Factors_A]=Decompose_Tensor_HOSVD_Cou_UnCo(Tensor_A,Rank_A);

% Computing the core tensors

[Core_Tensor_A]=find_Core_Tensor(Tensor_A,Singular_Factors_A);

% Reconstrcut Tensor

[Reconstructed_Tensor_A]=Reconstruct_Tensor(Core_Tensor_A,Singular_Factors_A);
%%

% initiallizing things before moving to iterative error reduction

Error=norm(Tensor_A)-norm(Reconstructed_Tensor_A)
Error_Prev=Error;

iter_number=1;
Singular_FA=Singular_Factors_A;

%%

while((Error > Error_Threshold) && (iter_number <= Max_iterations))
    
% Computing the Factors of Coupled Modes
    
      
    [S_FactorsA]=Reconstruct_Uncoupled_Modes_Tensor(Tensor_A,Singular_Factors_A,Rank_A);
  
%     Reconstructing Tensors based on iterative Calculations

    [Core_TrA]=find_Core_Tensor(Tensor_A,S_FactorsA);
    [RTA]=Reconstruct_Tensor(Core_TrA,S_FactorsA);
       
%     New Error
    
    Error_New=norm(Tensor_A)-norm(RTA);
    Error=abs(Error_Prev-Error_New)
    Error_Prev=Error_New;
    
    iter_number=iter_number+1
    Core_Tensor_A=Core_TrA;
    
    Singular_Factors_A=S_FactorsA;
    
end


end

