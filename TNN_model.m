function [y] = TNN_model(y0,theta,inputftrs,DataSetup,UserSetup,LPTN)
    
    thetaFieldName=fieldnames(theta);                                       %NN field name 'Power Loss', 'Themal Conductance' or 'Thermal capacitance'


    for subModel=1:numel(thetaFieldName)                                    %For each submodel
        if strcmp(thetaFieldName{subModel},'PowerLoss')                     %If the sub-model is power loss
            if UserSetup.LPTN.NN_Inputs==1                                  %If all input features 
                NNInput=[y0; inputftrs];
            else                                                            %If customized inputs 
                NNInput=[y0; inputftrs(DataSetup.Idx.PowerLosInput,:)];
            end 
            PowerLos=forwardNN(theta.(thetaFieldName{subModel}),NNInput);   % Forward the input trough the NN layers (See function below)
            clear NNInput
        elseif strcmp(thetaFieldName{subModel},'ThermalCond')               %If the sub-model is thermal conductance
            if UserSetup.LPTN.NN_Inputs==1                                  %If all input features  
                NNInput=inputftrs;
            else                                                            %If the sub-model is power loss
                NNInput=inputftrs(DataSetup.Idx.ConducInput,:);
            end 
            ThermalCond=forwardNN(theta.(thetaFieldName{subModel}),NNInput); % Forward the input trough the NN layers (See function below)
            clear NNInput
        else
            TherCap=sigmoid(theta.(thetaFieldName{subModel}).fc1.Weights);      %Exponentially weighted thermal capacitance
        end 
        
    end    

    if UserSetup.LPTN.NumOfConstantDummyNode==1                             %If one constant dummy
        Temps=[[y0;(min(DataSetup.norm.TargetMin)/DataSetup.norm.TargetWeight).*...
            ones(1,size(y0,2))];...
            inputftrs(DataSetup.Idx.AuxTemp,:)];
    elseif UserSetup.LPTN.NumOfConstantDummyNode==2                         %If two constant dummy 
        Temps=[[y0;[(min(DataSetup.norm.TargetMin)/DataSetup.norm.TargetWeight);1].*...
            ones(2,size(y0,2))];inputftrs(DataSetup.Idx.AuxTemp,:)];
    else                                                                    %If no constant dummy 
        Temps=[y0;inputftrs(DataSetup.Idx.AuxTemp,:)];
    end            

    PowerThermal=(Temps(LPTN.SourceIdx,:)-Temps(LPTN.SinkIdx,:)).*...
        ThermalCond(LPTN.ThermalCondVec,:);                                 %Heat transfer due to temeperature gradient 
    
    y=TherCap.*((LPTN.ThermalCondSelector*PowerThermal)+PowerLos)+y0;         %Final LPTN equation
end

function NNOut=forwardNN(theta,NNInput)                                     % Forward the input trough the NN layers 
        subModelFieldName=fieldnames(theta);                                %Get layer names 
        
        for layer=1:numel(subModelFieldName)                                %For each layer 
            if layer==1                                                     %If the first layer 
                buff = sigmoid(theta.(subModelFieldName{layer}).Weights *...
                    NNInput +...
                    theta.(subModelFieldName{layer}).Bias);                 %Forward the input through the first layer
            elseif layer==numel(subModelFieldName)                          %If the last layer
                
                NNOut = sigmoid(theta.(subModelFieldName{layer}).Weights*buff + ...
                    theta.(subModelFieldName{layer}).Bias);                %Pass the NN output 
                
            else                                                            %if intermediate layer
                buff = sigmoid(theta.(subModelFieldName{layer}).Weights*buff +...
                    theta.(subModelFieldName{layer}).Bias);                 %Pass the previous layer output to the next layer
            end
            
        end       
end 