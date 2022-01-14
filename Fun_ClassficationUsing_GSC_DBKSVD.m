function []=Fun_ClassficationUsing_GSC_DBKSVD(params, paraBKSVD)
%%================================================================
mcps         =paraBKSVD.mcps; %%-'6_3', '9_3', '6', '9'
sparse_method=params.sparse_method; %%bompm %%"omp" 
alpha        =paraBKSVD.alpha;
blockSparsity=paraBKSVD.blockSparsity;
block_size   =paraBKSVD.block_size;
%%-----------------------------------
sparsity     =params.sparsity_CD;
%%==============================================================
dict_name=['Scene15_CGC_DBKSVD_'  num2str(alpha*1000) '_bompm_s' num2str(blockSparsity) '_mcps_' num2str(mcps) '_' num2str(block_size)];
dictDir     =['../dictionary/'];
dictFile    =[dictDir dict_name '.mat'];
dsc         =load(dictFile)
D=dsc.D;
d=dsc.d;
% % D_PersonId=dsc.D_ClassId;
%%============================
data_path =['../data/'];
%%=======================================================================
%%---------START SPARSE CODING-------------------------------------------
fprintf('%s is using  for sparse coding....\n',sparse_method);
%%=======================================================================
data_name=['Train_Scene15_data_type_100_1_PCA_3000'];
data_file=[data_path data_name '.mat'];
sdf=load(data_file);
SV=normc(sdf.SV);
paraTrn.utt_id=sdf.image_id;
paraTrn.class_id=sdf.person_id;
%%---------------------------
% % [SV, paraTrn.class_id]=create_dictionary(SV, sdf.person_id); %% This steps could be redundant for certain task. Please check once the code.
%%---------------------------------------------------
if(strcmp(sparse_method,'omp')==1)
    X = omp(D'*SV, D'*D, sparsity);  %  GAMMA = OMP(DtX,G,T) 
elseif(strcmp(sparse_method,'bompm')==1)
     [X]=BOMP_c_fast_max_corr_V2(SV, D, sparsity,d);  %  GAMMA = OMP(DtX,G,T) 
else
    error('Wrong Sparse Coding options...\n');
end
paraTrn.X=full(X);
clear X spk_logical  SV;
%%=============================================================================
data_name=['Test_Scene15_data_type_100_1_PCA_3000'];
data_file  =[data_path data_name '.mat'];
sdf        =load(data_file);
SV         =normc(sdf.SV);

paraTst.utt_id=sdf.image_id;
paraTst.class_id=sdf.person_id;
clear sdf;
%%---------------------------------------------------
if(strcmp(sparse_method,'omp')==1)
    X = omp(D'*SV, D'*D, sparsity);  %  GAMMA = OMP(DtX,G,T) 
elseif(strcmp(sparse_method,'bompm')==1)
     [X]=BOMP_c_fast_max_corr_V2(SV, D, sparsity,d);  %  GAMMA = OMP(DtX,G,T)  
else
    error('Wrong Sparse Coding options...\n');
end
paraTst.X=full(X);
%%=============================================================================
%%---Compute the scores----------------------
totTrnUtt=size(paraTrn.X,2);
totTstUtt=size(paraTst.X,2);

scores_mat=zeros(totTrnUtt,totTstUtt);
for currLP1=1:totTrnUtt
    for currLP2=1:totTstUtt
        scores_mat(currLP1,currLP2)=sum((paraTrn.X(:,currLP1).*paraTst.X(:,currLP2)))/sqrt(sum((paraTrn.X(:,currLP1).*paraTrn.X(:,currLP1))) * sum((paraTst.X(:,currLP2).*paraTst.X(:,currLP2)))); 
    end
end
paraR.scores_mat=scores_mat;
%%==============================================================
% % [scoreSort, scoreSPos]=sort(scores_mat,1,'descend');
[~, scoreSPos]=sort(scores_mat,1,'descend');

numTruePredict=0;
numFalsePredict=0;

for currLP1=1:totTstUtt
    currPreditPerson=paraTrn.class_id{scoreSPos(1,currLP1)};
    currOraclePerson=paraTst.class_id{currLP1};
    
    if(strcmp(currPreditPerson,currOraclePerson)==1)
          numTruePredict=numTruePredict+1;
    else
          numFalsePredict=numFalsePredict+1; 
     end
end
recogAccur=(numTruePredict*100)/(numTruePredict+numFalsePredict);
fprintf('Recognition accuracy is %f percent\n',recogAccur);
%%================================================================
return;
