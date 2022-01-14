function [X, D] = DBKSVD(X, D, Y, d, C,alpha,paraDict)
%%-------------------------------
D_spk_id  =paraDict.D_spk_id;   %%--It stores the speakers information each atoms belong
trn_utt_id=paraDict.trn_utt_id; %%--It stores the speakers information for all the training data
[X_dis]   =FunMakeCrossClsCoefZERO(X,D_spk_id,trn_utt_id);
%%--------------------------
YY = D*X;
N = size(D,1);

for k = 1:max(d)
    wk = C(:,k)'~=0;
    SC_Conv_wk=find(wk~=0);
    swk = sum(wk);  %%--Total selected utterences
    %%--------------------------------
    currBlkPos=find(d==k);
    unqCBlkSID=unique(D_spk_id(1,currBlkPos));
    if((size(unqCBlkSID,1)~=1)&&(size(unqCBlkSID,2)~=1))
      error('Wrong speaker id. Please check code once..\n');  
    end
    currBlkClsUttPos=strmatch(unqCBlkSID,trn_utt_id);
    if(size(currBlkClsUttPos,2)==1)
        currBlkClsUttPos=currBlkClsUttPos';
    end
    %%-----------------------------
    [unionPos]=union(SC_Conv_wk,currBlkClsUttPos);
    
    if(size(unionPos,2)==1)
     unionPos=unionPos';
    end
    totInvolvedUtt=size(unionPos,2);
    
    binaryPosConvSC=ismember(unionPos,SC_Conv_wk);
    binaryPosClsSSC=ismember(unionPos,currBlkClsUttPos);
    
    absPosConvSC=find(binaryPosConvSC==1);
    absPosClsSSC=find(binaryPosClsSSC==1);
    
    %%-------------------------------------
    if ~totInvolvedUtt continue, end;  %If none of the training data using current block, do not update the block atoms
%     if ~swk continue, end;
    %%-------------------------------------------
    col = (d==k);  %%--Total number of atoms in current block. It is logical variable (0 or 1).
    len = sum(col);
    if(len<1)
        blockSeq=k
        len_block=len
        min_blk=min(d)
        max_blk=max(d)
        error('wrong block length');
    end
    %%-----------------------------------
    %%--------------------------------------
    YYk  = D(:,col)*X(col,wk);
    YYnk = YY(:,wk)-YYk;
    Erk  = Y(:,wk)-YYnk;
    %%--------------------------
    YY_dis  = D(:,currBlkPos)*X_dis(currBlkPos,currBlkClsUttPos);
    YYn_dis = YY(:,currBlkClsUttPos)-YY_dis;
    Er_dis  = Y(:,currBlkClsUttPos)-YYn_dis;
    %%-----------------------------
%     Er_comb=zeros(N,totInvolvedUtt);
    Er_global_conv=zeros(N,totInvolvedUtt);
    Er_global_dis =zeros(N,totInvolvedUtt);
    
    Er_global_conv(:,absPosConvSC)=Erk;
    Er_global_dis(:,absPosClsSSC)=Er_dis;
        
    Er_comb=((Er_global_conv)+(alpha*Er_global_dis))/(1+alpha);
    %%--------------------------
    [U,S,V] = pca(Er_comb,len);
     Dc = U;
     Xc = S*V';
     D(:,col) = Dc; %%- No change required
     X(col,unionPos) = Xc;
           
     YY_comb = D(:,col)*X(col,unionPos);
     
     YYn_comb=YY(:,unionPos)-YY_comb;   
     YY(:,unionPos) = YYn_comb + Dc*Xc;
end