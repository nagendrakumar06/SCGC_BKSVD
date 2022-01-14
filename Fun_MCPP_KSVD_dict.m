function []=Fun_MCPP_KSVD_dict(params)
%%%=-----------------------------------------
para_mcps.T            =params.T; %%Sparsity of each example
para_mcps.iterInit     =params.iterInit;%25;%
para_mcps.dict_learn   =params.dict_learn;
para_mcps.mcps         =params.mcps;
%%--------------------------------------
devDataPath=['../data/'];
devDataName=['Train_Scene15_data_type_100_1_PCA_3000'];

devDataFile=[devDataPath devDataName '.mat'];
ddf        =load(devDataFile);
SV         =normc(ddf.SV);
%%-----------------------------
% SV_size    =size(SV) %%only for temporary purpose
%%---------------------------
[D, d,D_ClassId]=CreateMCPS_Dict(SV,ddf.person_id,para_mcps);
%%-------------------------------------------------------
dictDir =['../dictionary/']; mkdir(dictDir);
dict_name=['Scene15_KSVD_' para_mcps.dict_learn '_mcps_' num2str(para_mcps.mcps)];
dictFile=[dictDir dict_name '.mat'];
save(dictFile,'D','d','D_ClassId','-v7.3');

return;



