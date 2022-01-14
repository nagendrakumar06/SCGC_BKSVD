function []=Fun_DBKSVD_dict(params)
%%--------------------
bompT        =params.SCT; %%  'bomp_m'--'bomp', 'bompm', 'bompa'
%%-------------------------
alpha        =params.alpha; 
%%-----------------------------------------------------
blockSparsity=params.blockSparsity;
block_size   =params.block_size;
bksvd_iter   =params.bksvd_iter; %%--10
mcps         =params.mcps;
%%=================================================
devDataPath=['../data/'];
devDataName=['Train_Scene15_data_type_100_1_PCA_3000'];

devDataFile=[devDataPath devDataName '.mat'];
ddf        =load(devDataFile)
Data       =normc(ddf.SV);
% SV_size    =size(Data) %%only for temporary purpose
%%==========================================================
dictDir     =['../dictionary/'];
dict_name   =['Scene15_KSVD_omp_mcps_' num2str(mcps)];
d_con       =load([dictDir dict_name '.mat'])
%%========================================
k      =blockSparsity; %%ceil(dict_sparsity/block_size)   %%1;        %block sparsity
max_it =bksvd_iter; 
%%=====================================
D       =d_con.D;
spk_seq =d_con.d; 
D_spk_id=d_con.D_ClassId;

Data_person_id=ddf.person_id;

paraDict.D_spk_id  =D_spk_id';
paraSC.D_spk_id=D_spk_id;
paraDict.trn_utt_id=Data_person_id';
%----------------------------------
 paraSC.Data=Data; paraSC.alpha=alpha; paraSC.blockSparsity=blockSparsity;
 paraSC.Data_person_id=Data_person_id;
for i = 1:max_it
    fprintf('%d out of %d going on...\n',i,max_it);
    [d]=CorrWiseFixBlockSpkCluster(D,block_size,spk_seq);
    %%--------------------------------------------------------
     paraSC.D=D; paraSC.d=d;
     [X] = Fun_Do_Supervised_SC(paraSC);
    %%--------------------------------------------------------
    [C]=Find_used_block(X,d);   %Find_used_block(X2,group);
    C=logical(C);
    [~, D] = DBKSVD(X, D, Data, d, C,alpha,paraDict);  
end
D_ClassId=d_con.D_ClassId;
% return;
%%========================================
DAP=['i_' num2str(max_it)];
dict_name=['Scene15_CGC_DBKSVD_'  num2str(alpha*1000) '_' bompT '_s' num2str(blockSparsity) '_mcps_' num2str(mcps) '_' num2str(block_size)];
dictFile=[dictDir dict_name '.mat'];
save(dictFile,'D','d','D_ClassId','-v7.3');
return;




            