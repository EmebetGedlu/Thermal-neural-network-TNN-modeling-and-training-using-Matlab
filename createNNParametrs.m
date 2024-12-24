function [NNParameters,LPTN]=createNNParametrs(DataSetup,UserSetup)

    NNParameters=struct;
    LPTN.NumDummyTemp=UserSetup.NumOfConstantDummyNode+...                  %Number of predictable or constant dummy nodes 
    UserSetup.NumOfPredictDummyNode;

%% Power Loss Net   
    if UserSetup.NN_Inputs==1                                               %If all input features 
        DataInput=DataSetup.NumOfInputFt;
    else                                                                    %If customized power loss input features
        DataInput=sum(DataSetup.Idx.PowerLosInput);     
    end 
   
    NumInput=DataInput+DataSetup.NumOfTargetFt+...                          %Number of NN input 
        UserSetup.NumOfPredictDummyNode;
    NumOutput=DataSetup.NumOfTargetFt+UserSetup.NumOfPredictDummyNode;      %Number of NN output

    SubModelName='PowerLoss';                                               %NN name 

    NNParameters=NNStructure(NumInput,NumOutput,UserSetup.NumOfLayer,...    %Create NN weights and bias (See the function below)
        UserSetup.NumNeuronPerLayer,SubModelName,NNParameters);
    clear SubModelName

%% Thermal conductances

    if UserSetup.NN_Inputs==1                                               %If all input features 
        DataInput=DataSetup.NumOfInputFt;
    else                                                                    %If customized power loss input features
        DataInput=sum(DataSetup.Idx.ConducInput);
    end 
%------------Structuring thermal conductance 
    LPTN.NumbTemp=DataSetup.NumOfTargetFt+sum(DataSetup.Idx.AuxTemp)+...
        LPTN.NumDummyTemp;                                                  %Number of thermal nodes 
    LPTN.ThermalCond=ones(DataSetup.NumOfTargetFt+...
        UserSetup.NumOfPredictDummyNode,LPTN.NumbTemp);                     %Shape of conductance matrix 
    LPTN.NumOfThermalCond= sum(triu(LPTN.ThermalCond,1),"all");             %Number of unique thermal conductance

    triuIndex=zeros(1,LPTN.NumOfThermalCond);                               %Conductance matrix linear index 
    IdxCount=1;

    for RowCount=1:LPTN.NumbTemp-1                                          %Assign conductance matrix linear index 
        for ColCount=RowCount:LPTN.NumbTemp-1            
            triuIndex(IdxCount)=ColCount*LPTN.NumbTemp+RowCount;
            IdxCount=IdxCount+1;     
        end
    end 

    triuMatrix=zeros(LPTN.NumbTemp); 
    triuMatrix(triuIndex)=1:numel(triuIndex);
    LPTN.ThermalCond=triuMatrix+triuMatrix';
    LPTN.ThermalCond(DataSetup.NumOfTargetFt+...
      UserSetup.NumOfPredictDummyNode+1:end,:)=[];                          %conductance matrix with NN output number 
 
    [LPTN.SinkIdx, LPTN.SourceIdx, LPTN.ThermalCondVec]=...                 %Assign sink and source temeperature for each thermal conductance
        find(LPTN.ThermalCond);


    LPTN.ThermalCondSelector=zeros(DataSetup.NumOfTargetFt+...
        UserSetup.NumOfPredictDummyNode,sum(LPTN.ThermalCond~=0,"all"));    %This matrix will assign thermal conductance with temeperatures 
    for sink=1:DataSetup.NumOfTargetFt+UserSetup.NumOfPredictDummyNode
        LPTN.ThermalCondSelector(sink,LPTN.SinkIdx==sink)=1;
%         LPTN.ThermalCondSelector(sink,LPTN.SourceIdx>3)=0.1;
    end      

    NumInput=DataInput+UserSetup.NumOfPredictDummyNode;                         %Number of NN input 
        

    NumOutput=LPTN.NumOfThermalCond;
    SubModelName='ThermalCond';

    NNParameters=NNStructure(NumInput,NumOutput,UserSetup.NumOfLayer,...
    UserSetup.NumNeuronPerLayer,SubModelName,NNParameters);
    clear SubModelName


%% Thermal capacitance 
    
%     SubModelName='ThermalCap';
%     dynamicname=strcat('fc',num2str(1));
% 
%     NumInput=DataSetup.NumOfTargetFt+UserSetup.NumOfPredictDummyNode;
%     NumOutput=DataSetup.NumOfTargetFt+UserSetup.NumOfPredictDummyNode;
% 
%     NNParameters=NNStructure(NumInput,NumOutput,2,...
%     UserSetup.NumNeuronPerLayer,SubModelName,NNParameters);

    SubModelName='ThermalCap';
    dynamicname=strcat('fc',num2str(1));

    NNParameters.(SubModelName).(dynamicname).Weights = ...
        dlarray(-3.*ones(DataSetup.NumOfTargetFt+UserSetup.NumOfPredictDummyNode,1));

end 

%%
function NNParameters=NNStructure(NumInput,NumOutput,NumOfLayer,...
    NumNeuronPerLayer,SubModelName,NNParameters)
    for Layer=1:NumOfLayer                                                  %For every layer 
        if Layer==1                                                         %If the first layer
            sz = [NumNeuronPerLayer NumInput];
        elseif Layer==NumOfLayer                                            %If output layer
            sz = [NumOutput NumNeuronPerLayer];                             
        else                                                                %If intermediate layer
            sz = [NumNeuronPerLayer NumNeuronPerLayer];
        end 
        dynamicName=strcat('fc',num2str(Layer));
        NNParameters.(SubModelName).(dynamicName).Weights = ...
            initializeGlorot(sz, NumInput, NumOutput);                      %Initialize the weights (See the function below)
        NNParameters.(SubModelName).(dynamicName).Bias = ...
            initializeZeros([sz(1) 1]);                                     %Initialize the biases (See the function below)
        clear sz dynamicName
    end    
end 
%% NN parameters initialization 
function weights = initializeGlorot(sz,numOut,numIn,className)
    arguments
        sz
        numOut
        numIn
        className = 'single'
    end
    
    Z = 2*rand(sz,className) - 1;

    bound = sqrt(6 / (numIn + numOut));
    
    weights = bound * Z;
    weights = dlarray(weights);    
end

function parameter = initializeZeros(sz,className)    
    arguments
        sz
        className = 'single'
    end    
    parameter = rand(sz,className);
    parameter = dlarray(parameter);
end