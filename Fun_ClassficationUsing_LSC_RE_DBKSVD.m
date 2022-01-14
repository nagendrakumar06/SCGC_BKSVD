function []=Fun_ClassficationUsing_LSC_RE_DBKSVD(params, paraBKSVD)
%%================================================================
mcps         =paraBKSVD.mcps; %%-'6_3', '9_3', '6', '9'
sparse_method=params.sparse_method; %%bompm %%"omp" 
alpha        =paraBKSVD.alpha;
blockSparsity=paraBKSVD.blockSparsity;
block_size   =paraBKSVD.block_size;
%%-----------------------------------
sparsity     =params.sparsity_LSC;
%%==============================================================
dict_name=['Scene15_CGC_DBKSVD_'  num2str(alpha*1000) '_bompm_s' num2str(blockSparsity) '_mcps_' num2str(mcps) '_' num2str(block_size)];
dictDir     =['../dictionary/'];
dictFile    =[dictDir dict_name '.mat'];
dsc         =load(dictFile)
D=dsc.D;
d=dsc.d;
D_ClassId=dsc.D_ClassId;
%%============================
data_path =['../data/'];
%%=======================================================================
data_name=['Test_Scene15_data_type_100_1_PCA_3000'];
data_file=[data_path data_name '.mat'];
sdf=load(data_file);
SV=normc(sdf.SV);
paraTrn.utt_id=sdf.image_id;
paraTrn.class_id=sdf.person_id;
%%-----------------------------------------------------------------
[unqClassId,~, posClassId]=unique(paraTrn.class_id);
totCls=max(posClassId);

paraS.trn_cls_id=unqClassId;
paraS.tst_cls_id=paraTrn.class_id;

scores=zeros(totCls,size(SV,2));
%%---------START SPARSE CODING-------------------------------------------
fprintf('%s is using  for LOCAL SPARSE CODING....\n',sparse_method);
%%=======================================================================
for currLP2=1:totCls
    fprintf('Class Info RecogInfo--%d--%d\n',currLP2,totCls);
    currScrCls=unqClassId(currLP2);
    currSelAPos=strmatch(unqClassId(currLP2),D_ClassId)';
    Dc=D(:,currSelAPos);
    [~,~,dc]=unique(d(1,currSelAPos));
    [Xc_full]=BOMP_MCA_no_ML(SV, Dc, sparsity,dc);  %  GAMMA = OMP(DtX,G,T) 
    scores(currLP2,:)=sum((SV-Dc*Xc_full).^2);
end
paraS.scores=-scores;
%%-----------------------------------------------------------------------
totTestUtt=size(paraS.scores,2);
[scoreSort, scoreSPos]=sort(paraS.scores,1,'descend');

numTruePredict=0;
numFalsePredict=0;

for currLP1=1:totTestUtt
    currPreditPerson=paraS.trn_cls_id{scoreSPos(1,currLP1)};
    currOraclePerson=paraS.tst_cls_id{currLP1};
        
    if(strcmp(currPreditPerson,currOraclePerson)==1)
          numTruePredict=numTruePredict+1;
    else
          numFalsePredict=numFalsePredict+1; 
    end
end
recogAccur=(numTruePredict*100)/(numTruePredict+numFalsePredict);
fprintf('Recognition accuracy is %f percent\n',recogAccur);
return;