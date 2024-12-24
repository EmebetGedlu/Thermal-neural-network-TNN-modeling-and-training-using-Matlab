function [MiniBatchData] = createMiniBatch(DataSetup,XTrain,YTrain,HistWeight,UserSetup)

    [MBSetup.ProfileLength,MBSetup.ProfileNum,~]=...
        groupcounts(DataSetup.MeasNo);                                      %Compute number of profiles (experiments) and their length

    MiniBatchData=[];
    for ProfNum=1:numel(MBSetup.ProfileNum)                                                 
        ProfileData.InputFt=...
            XTrain(DataSetup.MeasNo==MBSetup.ProfileNum(ProfNum),:);        %Input feature of each profile
        ProfileData.TargetFt=...
            YTrain(DataSetup.MeasNo==MBSetup.ProfileNum(ProfNum),:);        %Target feature of each profile
        
        ProfileData.HistWeight=...
            HistWeight(DataSetup.MeasNo==MBSetup.ProfileNum(ProfNum),:);

        ProfileData.InputFt(:,1:2)=movmean(ProfileData.InputFt(:,1:2),20);

        NewMiniBatchData = Prof2MB(ProfileData, MBSetup,ProfNum,UserSetup); %Prepare mini-batch (See function below)
        clear ProfileData 
        if isempty(MiniBatchData)
            MiniBatchData=NewMiniBatchData;            
        else 
            MiniBatchData.targets=...
                cat(2,MiniBatchData.targets,NewMiniBatchData.targets);
            MiniBatchData.inputftrs=...
                cat(2,MiniBatchData.inputftrs,NewMiniBatchData.inputftrs);
            MiniBatchData.CostFnWeight=...
                cat(2,MiniBatchData.CostFnWeight,NewMiniBatchData.CostFnWeight);
            MiniBatchData.y0=...
                cat(2,MiniBatchData.y0,NewMiniBatchData.y0);
        end
        clear NewMiniBatchData
    end 
end


function MiniBatchData = Prof2MB(ProfileData,MBSetup,ProfIndex,UserSetup)   %Prepare mini-batch for ProfileData

    numTimesteps=MBSetup.ProfileLength(ProfIndex);                          %Profile length
    numTimesPerObs=UserSetup.SubSeqLength;                                  %User defined subsequence length
    
    XTrain=ProfileData.InputFt';                                            %NN time series is row vector
    YTrain=ProfileData.TargetFt';
    HistWeight=ProfileData.HistWeight';
    MBSize= floor(numTimesteps/(numTimesPerObs));                         %Number of full batches
    
    s=1:numTimesPerObs:numTimesteps;                                        %Mini-batch start points
    y0 = YTrain(:, s);                                                      %Initial target values
    targets = zeros([size(YTrain,1), MBSize, numTimesPerObs]);                %Training dataset in "Channel-Batch-Time (CBT)"format
    inputftrs= zeros([size(XTrain,1), MBSize, numTimesPerObs]);
    weight=zeros([size(YTrain,1), MBSize, numTimesPerObs]);
    
    for MBNum = 1:MBSize                                                    %Filling the full batches
        targets(:, MBNum , 1:numTimesPerObs) =...
            YTrain(:, s(MBNum ) + 1:(s(MBNum ) + numTimesPerObs));
        inputftrs(:, MBNum , 1:numTimesPerObs) =...                         %Input features are smoothed using moving average 
            movmean(XTrain(:, s(MBNum ) + 1:(s(MBNum ) + numTimesPerObs)),...
            UserSetup.MAHor,2);

        weight(:, MBNum , 1:numTimesPerObs) =...
            HistWeight(:, s(MBNum ) + 1:(s(MBNum ) + numTimesPerObs));
    end
    
    paddedBatchTarget=-1.*ones([size(YTrain,1), 1, numTimesPerObs]);
    paddedBatchInput=-1.*ones([size(XTrain,1), 1, numTimesPerObs]);
    paddedBatchWeight=-1.*ones([size(YTrain,1), 1, numTimesPerObs]);
    remain=numTimesteps-s(MBNum +1);  
    if remain>UserSetup.MinSubSeqLength     % Adding the last batch if it is long enough
        paddedBatchTarget(:,:,1:remain)=YTrain(:, s(MBNum +1) + 1:end);
%         reshape(YTrain(:, end-numTimesPerObs+1:end),...
%             [size(targets,1),1,size(targets,3)]);
        paddedBatchInput(:,:,1:remain)=XTrain(:, s(MBNum +1) + 1:end);
        paddedBatchWeight(:,:,1:remain)=HistWeight(:, s(MBNum +1) + 1:end);
%         y0End=YTrain(:, end-numTimesPerObs);
        targets=cat(2,targets,paddedBatchTarget);
        inputftrs=cat(2,inputftrs,paddedBatchInput);
        weight=cat(2,weight,paddedBatchWeight);
%         y0(:,end)=y0End;
    else 
        y0(:,end)=[];
    end 
    MiniBatchData.targets=targets;
    MiniBatchData.inputftrs=inputftrs;
%     MiniBatchData.CostFnWeight=targets./DataSetup.norm.TargetMax;
    MiniBatchData.CostFnWeight=weight;
    MiniBatchData.y0=y0;
end 