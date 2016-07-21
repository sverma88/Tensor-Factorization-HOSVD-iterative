% Function to project training images on lower dimensions

function [Projected_Images]=Project_Images(Tensor_A,Row_Projection,Col_Projection)

% Inputs
% Tensor_A             : Tensor 'A' of Mode 'n'
% Row_Projection       : Low rank Singular Factors of rows computed from
%                        Tensor A using Iterative HOSVD
% Col_Projection       : Low rank Singular Factors of columns computed from
%                        Tensor A using Iterative HOSVD
%
% Outputs
% Projected_Images     : Projection of images on Singualr factors (reducing
%                        dimensions)
%
%
% Author               : Sunny Verma (sunnyverma.iitd@gmail.com)
% Last_Update          : 14/07/2016

% %


Size_A=size(Tensor_A);
Number_of_Instances=Size_A(1,1);
Number_Images_Each_Instance=Size_A(1,2);
Pixels_Rows=size(Row_Projection,2);
Pixels_Cols=size(Col_Projection,2);
Projected_Images_Rows=Number_of_Instances*Number_Images_Each_Instance;
Projected_Images_Cols=Pixels_Rows*Pixels_Cols;
Projected_Images=zeros(Projected_Images_Rows,Projected_Images_Cols);
Gpu_Row_Projection=gpuArray(Row_Projection);
Gpu_Col_Projection=gpuArray(Col_Projection);


for i=1:Number_of_Instances
    Auxillary_P_Images=gpuArray(zeros(Number_Images_Each_Instance,Projected_Images_Cols));
    for j=1:Number_Images_Each_Instance
        Auxillary_P_Images(j,:)=reshape(Gpu_Row_Projection'*(gpuArray(Tensor_A(i,j,:,:).data))*Gpu_Col_Projection,1,Projected_Images_Cols);
    end
    start_index=((i-1)*Number_Images_Each_Instance) + 1;
    end_index=(i*Number_Images_Each_Instance);
    Projected_Images(start_index:end_index,:)=gather(Auxillary_P_Images);
end

end