% Main Script to execute iterative tensor decomposition 
clc
rng('default');
index=randperm(41,8);
All_index=[1:41];
All_index(ismember(All_index,index))=[];

Tensor_1_Train=Tensor_Data(1,1:10,index,:,:);
Tensor_1_Test=Tensor_Data(1,1:10,All_index,:,:);
Rank_1=[5,30,5,5];
Max_iterations=50;
Error_Threshold=10^-5;

% Decomposing Tensor (finding low rank Projection)
[Core_Tensor_1,Singular_Factors_Tensor1,Original_Singular_Factors_Tensor1]=Decompose_Tensor_Coupled_HOSVD_iteratively(Tensor_1_Train,Rank_1,Error_Threshold,Max_iterations);

% Feature Extraction in Low rank
Row_Projection=Singular_Factors_Tensor1{3,1};
Col_Projection=Singular_Factors_Tensor1{4,1};
[Projected_Images_Train_Tensor1]=Project_Images(Tensor_1_Train,Row_Projection,Col_Projection);
[Projected_Images_Test_Tensor1]=Project_Images(Tensor_1_Test,Row_Projection,Col_Projection);

save('20_ETH80_Tensor1_Rank_5_30_5_5.mat','Singular_Factors_Tensor1','Original_Singular_Factors_Tensor1','Projected_Images_Train_Tensor1','Projected_Images_Test_Tensor1','Tensor_1_Test');