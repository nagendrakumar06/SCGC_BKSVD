% =========================================================================
% An example code for the algorithm proposed in
%
%   Nagendra Kumar and Rohit Sinha,
%   "Improved Structured Dictionary Learning via Correlation and Class Based Block Formation", 
%   IEEE Transactions on Signal Processing, Vol. 66, no. 19, pp. 5082-5095, Oct. 2018.
%
% Author: Nagendra Kumar (k.nagendra@iitg.ac.in, nagendrakumar06@gmail.com)
% Date: 20-10-2018
% =========================================================================
clc;
close all;
clear all;
fprintf('START---Program...\n');
addpath(['./Additional_codes/']);
addpath('./toolkits/ompbox10/'); % add sparse coding algorithem OMP
addpath('./toolkits/ksvdbox13/');   % add K-SVD box
%%%----------------------------------
step_3=1; %% Learning the KSVD dictionary
step_4=1; %% Learning the KSVD dictionary
step_5=1; %% Do the batch sparse coding over full dictionary and use CDS for scoring
step_6=1; %% Do the sparse coding over each of class specific sub-dictionaries and use reconstruction error for scoring
%%----------------------------------------
paraDict.mcps=30;  %% Number of atoms for each class-specific sub-dictionary.

paraBKSVD.SCT ='bompm'; %%  'bomp_m'--'bomp', 'bompm', 'bompa'
paraBKSVD.alpha = 4; 
paraBKSVD.mcps  = paraDict.mcps;
paraBKSVD.blockSparsity=5;  %% Block sparsity for learing the block-dictionary.
paraBKSVD.block_size   =4;  %% Block size in block-structure.
paraBKSVD.bksvd_iter   =10; %%--10

parsTest.sparse_method='bompm'; %'bompm, omp
parsTest.sparsity_CD=30;  %% Sparsity for GLOBAL (Over FULL Dictionary) SPARSE CODING
parsTest.sparsity_LSC=25;  %% Sparsity for LOCAL SPARSE CODING
%%==========================================================
if (step_3==1)
   paraDict.T            =4; %%Sparsity of each example
   paraDict.iterInit     =25;%25;%
   paraDict.dict_learn   ='omp';
   Fun_MCPP_KSVD_dict(paraDict)
end
%%-------------------------------------------------------
if (step_4==1)
    Fun_DBKSVD_dict(paraBKSVD);
end
%%-------------------------------------------
if (step_5==1)
   Fun_ClassficationUsing_GSC_DBKSVD(parsTest, paraBKSVD);
end
%%%------------------------------------------
if (step_6==1)
   Fun_ClassficationUsing_LSC_RE_DBKSVD(parsTest, paraBKSVD);
end
%%%------------------------------------------
fclose all;
fprintf('END---Program...\n');