function [Setup,XTrain,YTrain,HistWeight]=data_prep(DatasetClass)
% DataSet=load('PM2315_3_TpMdlData.mat');                                     %Load the row dataset
ds = tabularTextDatastore('measures.csv'); %read data without copying in to memory 
T = readall(ds); %read all of the data in the data store 


Idx.Train=true(size(T,1),1);                                         %Training dataset index

Idx.Train(T.profile_id==60|T.profile_id==62|T.profile_id==74)=false;
Idx.Test=~Idx.Train;  

%Test dataset index

[Setup.InputName,Setup.Idx]=ListofInputFt(T.Properties.VariableNames);             %Input feature selection (See the function below)    

[Setup.TargetName,Setup.Idx]=ListofTargetFt(T.Properties.VariableNames,Setup.Idx); %Target feature selection (See the function below)



if strcmp(DatasetClass,'Train')                                                %Is the data preparation for training or test                                
    IdxDataSetType=Idx.Train;    
else 
    IdxDataSetType=Idx.Test;
end




%------------Dataset Extract
InputFt=table2array(T(IdxDataSetType,Setup.Idx.BaseFeat));                  %Extract selected input features from the raw dataset 
TargetFt_buff=table2array(T(IdxDataSetType,Setup.Idx.TargetFeat));               %Extract selected target features from the raw dataset 
TargetFt=[TargetFt_buff(:,1),max(TargetFt_buff(:,2:4),[],2)];
% TargetFt(:,1)=max(DataSet.data.Y(IdxDataSetType,1:16),[],2);
Setup.Idx.TargetFeat(:,8:end)=false;

%-----------Dataset metrices
Setup.norm.InputMax=max(abs(table2array(T(:,Setup.Idx.BaseFeat))));         %Input Absolute maximum
Setup.norm.TargetMax=max(abs(table2array(T(:,Setup.Idx.TargetFeat))));      %Target Absolute maximum
Setup.norm.TargetMin=min(abs(table2array(T(:,Setup.Idx.TargetFeat))));      %Target Absolute minimum
Setup.norm.InputMean=mean(table2array(T(:,Setup.Idx.BaseFeat)));            %Input mean
Setup.norm.TargetMean=mean(table2array(T(:,Setup.Idx.TargetFeat)));         %Target mean
Setup.norm.InputStd=std(table2array(T(:,Setup.Idx.BaseFeat)));              %Input standard deviation
Setup.norm.TargetWeight=200;



Setup.MeasNo=T.profile_id(IdxDataSetType,:);                         %Extract experiment number 
Setup.NumOfInputFt=size(InputFt,2);                                         %Number of input features 
Setup.NumOfTargetFt=size(TargetFt,2);                                       %Number of target features

[XTrain,YTrain]= data_normalise(Setup,InputFt,TargetFt);               %Normalise input and target features (See function below)

HistWeight=YTrain./Setup.norm.TargetMax;

end

%% Normalize the dataset
function [XTrain,YTrain]= data_normalise(DataSetup,InputFt,TargetFt)
XTrain(:,~DataSetup.Idx.AuxTemp)=InputFt(:,~DataSetup.Idx.AuxTemp)./DataSetup.norm.InputMax(:,~DataSetup.Idx.AuxTemp);                                     %Normalise the input by absolute maximum (different methods can be adopted here)
XTrain(:,DataSetup.Idx.AuxTemp)=InputFt(:,DataSetup.Idx.AuxTemp)./DataSetup.norm.TargetWeight;
YTrain=TargetFt./DataSetup.norm.TargetWeight;
end 
%% Choose inpit features 
function [Name,Idx]=ListofInputFt(XNames)                                   %List here all necessary input features 
    BaseFeat={  'ambient',... %rotor speed                                           
                'coolant',... %d-axis current
                'i_d',...%q-axis current  
                'i_q',... %d-axis voltage
                'motor_speed',...%q-axis voltage
                };
%                 'TpNtc1',...
    Idx.BaseFeat=false(1,numel(XNames));                                    %Indexing selected input features from all input features 
    for ii=1:numel(BaseFeat)
        Idx.BaseFeat(strcmp(XNames, BaseFeat{ii}))=true;                    
    end
    Name.BaseFeat=XNames(1,Idx.BaseFeat);

    %% dividing input features into auxiliarx temperature, power loss input and thermal conductance input
    
    Name.AuxTemp={  'ambient',...%Stator oil temperature               %List here auxiliary temperatures (real time measured ones)
                    'coolant',...%Stator oil temperature
                    };
    Idx.AuxTemp=false(1,numel(Name.BaseFeat));                              %Indexing auxiliary temperatures from the selected input features
    for ii=1:numel(Name.AuxTemp)
        Idx.AuxTemp(strcmp(Name.BaseFeat, Name.AuxTemp{ii}))=true;
    end

    Name.PowerLosInput={    'i_d',...%q-axis current  
                            'i_q',... %d-axis voltage
                            'motor_speed',...%q-axis voltage
                            };

    Idx.PowerLosInput=false(1,numel(Name.BaseFeat));                        %Indexing customized power loss inputs from the selected input features
    for ii=1:numel(Name.PowerLosInput)
        Idx.PowerLosInput(strcmp(Name.BaseFeat, Name.PowerLosInput{ii}))=true;
    end

    
    Name.ConducInput={  'motor_speed',...%q-axis voltage                            %List of input variables for customized thermal conductance NN                                   
                        'ambient',...%Stator oil flow rate
                        'coolant',...%Stator oil temperature                        
                        };
    Idx.ConducInput=false(1,numel(Name.BaseFeat));                          %Indexing customized thermal conductance inputs from the selected input features
    for ii=1:numel(Name.ConducInput)
        Idx.ConducInput(strcmp(Name.BaseFeat, Name.ConducInput{ii}))=true;
    end

end


function [Name,Idx]=ListofTargetFt(YNames,Idx)                              %List here target features 
    TargetFeat={  'pm',... %Selected rotor hotspot   
                  'stator_tooth',... %Selected stator winding hotspot
                  'stator_winding',... %Selected stator winding hotspot
                  'stator_yoke',... %Selected stator winding hotspot
                };
    Idx.TargetFeat=false(1,numel(YNames));                                  %Indexing selected target features from all target features 
    for ii=1:numel(TargetFeat)
        Idx.TargetFeat(strcmp(YNames, TargetFeat{ii}))=true;
    end
        Name.TargetFeat=YNames(1,Idx.TargetFeat);  
end
