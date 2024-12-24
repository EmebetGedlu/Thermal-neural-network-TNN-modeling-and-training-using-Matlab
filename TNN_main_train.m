
% Purpose: To model and train thermal neural network (TNN)
% This code has 6 main section 
% 1) UserInit.m: 
%The user can configure dataset, modeling and training related setups. 
% 2) data_prep.m: 
%The raw dataset is preprocessed for training perpuse 
% 3) createMiniBatch.m: 
%The pre-processed dataset is rearranged in mini-batches to ease the
%tarining process 
% 4) createNNParametrs.m: 
%The TNN parameters i.e. power loss NN, thermal conductance NN and
%inverted-thermal capacitance are defined and initialized here 
% 5) TNN_model.m:
%It is the TNN model wich forwards the input features and predicts the
%target


clc
clear all 
close all 
% rng (0,"twister")
%% Datast preparation
UserSetup=UserInit;                                                         %Configure dataset, model and training related setups
[DataSetup,XTrain,YTrain,HistWeight]=data_prep('Train');                     %Data processing 
[MiniBatchData]=createMiniBatch(DataSetup,XTrain,YTrain,HistWeight,UserSetup.MB);      %Mini-batch preparation
 clear XTrain YTrain

%% NN structure
[NNParametrs,LPTN]=createNNParametrs(DataSetup,UserSetup.LPTN);             %Creates trainable parameters 
% load("CustomModel_TNNParametrs.mat")  

%% Training  

% -----Learning curve plot
f = figure;
f.Position(3) = 2*f.Position(3);
% subplot(1,2,1)
C = colororder;
lineLossTrain = animatedline(Color=C(2,:),Marker="o");
lineLossVal = animatedline(Color=C(1,:),Marker="o");
ylim([0 inf])
xlabel("Iteration")
ylabel("Loss")
grid on

% Initialize the averageGrad and averageSqGrad parameters for the Adam solver.
averageGrad = [];
averageSqGrad = [];
start = tic;
vel=[];
% Mini-batch 


%% Train in itration 
MBTotal=size(MiniBatchData.inputftrs,2);                                    %Total Mini-batch
numMB=floor(MBTotal/UserSetup.Training.MBPerItration);                      %Number of full Mini-batches
% numMB=1;
iter=0;                                                                     %initial itration
Performance=zeros(UserSetup.Training.numEpoch ,1);                          %Performance recording matrix
for epoch = 1:UserSetup.Training.numEpoch                                   %For each epoch (entire training dataset is evaluated)    
    randSeqMB=randperm(MBTotal);                                            %Random mini-batch indexing    
%     subplot(1,2,2)
%     plot(randSeqMB)
%     hold on 
    EpochPerformance=zeros(numMB*UserSetup.MB.SubSeqLength/...
        UserSetup.Training.TBTTNumLength,1);
    updateCount=0;
    for iterMB=1:numMB                                                      %For each mini-batch in one epoch
        indexMB=false(size(randSeqMB));
        indexMB(randSeqMB((iterMB-1)*UserSetup.Training.MBPerItration+1:...
            iterMB*UserSetup.Training.MBPerItration))=true;                 %Random minibatch order

        y0=dlarray(MiniBatchData.y0(:,indexMB));                            %inital target value of the mini-batch 
        x=dlarray(MiniBatchData.inputftrs(:,indexMB,:));                    %input feature of the mini-batch 
        y=dlarray(MiniBatchData.targets(:,indexMB,:));                      %target feature of the mini-batch
        CostFnWeight=dlarray(MiniBatchData.CostFnWeight(:,indexMB,:));

        if UserSetup.LPTN.NumOfPredictDummyNode>0                           %if there is predictable node, initialize with mean of all targets 
            y0=[y0;repmat(max(y0),[UserSetup.LPTN.NumOfPredictDummyNode,1])];        
        end

        for TBTTNum=1:UserSetup.MB.SubSeqLength/UserSetup.Training.TBTTNumLength% For each truncated backpropagation through time (TBTT)            
            iter=iter+1;                                                    %Update itration
            TBTTNumIdx=(TBTTNum-1)*UserSetup.Training.TBTTNumLength+1:...
                TBTTNum*UserSetup.Training.TBTTNumLength;                   %TBTT index
            [loss,gradients,y0] = dlfeval(@modelLoss,y0,x(:,:,TBTTNumIdx),...
                NNParametrs,y(:,:,TBTTNumIdx),DataSetup,UserSetup,LPTN,CostFnWeight(:,:,TBTTNumIdx));    %Compute loss and gradient (See function below)  
            [NNParametrs,averageGrad,averageSqGrad] = adamupdate(...
                NNParametrs,gradients,averageGrad,averageSqGrad,iter,...
            UserSetup.Training.learnRate,UserSetup.Training.gradDecay,...
            UserSetup.Training.sqGradDecay,10e-20);                                %Adam updates the parameters 

            updateCount=updateCount+1;
            currentLoss = double(loss);                                     %Plot the TBTT loss   
            EpochPerformance(updateCount)=currentLoss;
            addpoints(lineLossTrain,iter,extractdata(currentLoss));
            D = duration(0,0,toc(start),Format="hh:mm:ss");
            title("Elapsed: " + string(D)+'NTC')
            drawnow
        end                         
    end   

Performance(epoch)=mean(EpochPerformance);
addpoints(lineLossVal,iter,mean(EpochPerformance));                          %Plot validation 
if mean(EpochPerformance)==min(Performance(Performance>0))
    BestNNParametrs=NNParametrs;
end
% BestNNParametrs=NNParametrs;
% %    legend('Training', 'Validation')
% if find(Performance==min(Performance(1:epoch)),1)<...
%     epoch-UserSetup.Training.ValdAttempt|toc(start)/3600>1.5                            %If performance is not improved for 15 attempts     
%     break
% end

end
SaveParameters(BestNNParametrs,Performance,UserSetup)                           %Save trained parameters and setups (See function below)


%% Loss and automatic differentiation 

function [loss,gradients,Y0_est] = modelLoss(Y0,inputftrs,NNParameters,...
    targets,DataSetup,UserSetup,LPTN,CostFnWeight)                      
    [Y,Y0_est] = forward(Y0,inputftrs,NNParameters,DataSetup,UserSetup,LPTN); %Estimate the target and set initial for the next TBTT(See function below)       
    mask=targets>0;
%     subplot(1,2,2)
    Yest=extractdata(Y.*mask);
    Ytar=extractdata(targets.*mask);
    t=cumsum(ones(1,size(Yest,3)));
%     hold off
%     plot(t,reshape(Yest(1,:,:),size(Yest,2),size(Yest,3)),'b');
%     hold on 
%     grid on 
%     plot(t,reshape(Yest(2,:,:),size(Yest,2),size(Yest,3)),'r');
%     plot(t,reshape(Ytar(1,:,:),size(Yest,2),size(Yest,3)),'g');
%     plot(t(1),extractdata(Y0(1,:)),'*')

    
    loss =l2loss(Y,targets,...
        DataFormat="CBT",Mask=mask,NormalizationFactor="mask-included");
   %.*DataSetup.norm.TargetWeight
        gradients = dlgradient(loss,NNParameters,'RetainData',true);

end

%% Estimate estimated target at each sampling time
function [Y,Y0] = forward(Y0,inputftrs,NNParameters,DataSetup,UserSetup,LPTN)
    
    Y_buff=dlarray(zeros(DataSetup.NumOfTargetFt+...                        %Estimated target matrix
        UserSetup.LPTN.NumOfPredictDummyNode,size(inputftrs,2),size(inputftrs,3)));
         
    for ii=1:size(inputftrs,3)                                              %For each sampling time 
        Y_buff(:,:,ii)=TNN_model(Y0,NNParameters,...
            inputftrs(:,:,ii),DataSetup,UserSetup,LPTN);                    %TNN prediction (See the function file)
        Y0=Y_buff(:,:,ii);
    end 
    Y=Y_buff(1:DataSetup.NumOfTargetFt,:,:);
end

function SaveParameters(NNParametrs,Performance,UserSetup)
    if UserSetup.LPTN.NN_Inputs==1
        Prefix1='BasicModel_5min_50_3_';
    else  
        Prefix1='CustomModel_5min_50_3_';
    end 
    if UserSetup.LPTN.NumOfConstantDummyNode>0
        Prefix2=strcat('CD',num2str(UserSetup.LPTN.NumOfConstantDummyNode));
    else 
        Prefix2='';
    end 

    if UserSetup.LPTN.NumOfPredictDummyNode>0
        Prefix3=strcat('PD',num2str(UserSetup.LPTN.NumOfPredictDummyNode));
    else 
        Prefix3='';
    end

    save(strcat(Prefix1,Prefix2,Prefix3,'UserSetup'),'UserSetup')
    save(strcat(Prefix1,Prefix2,Prefix3,'Performance'),'Performance')
    save(strcat(Prefix1,Prefix2,Prefix3,'NNParametrs'),'NNParametrs')

end 



