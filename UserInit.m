function UserSetup=UserInit
%% Dataset related setups
UserSetup.Dataset.DatasetClass=1;                                           %1 for train                            
                                                                            %2 for test

%% Mini-Batch related setups
UserSetup.MB.SubSeqLength=30*60*2;                                          %Mini-batch subsequence length
UserSetup.MB.MAHor=1;                                                       %Moving average horizon length
UserSetup.MB.MinSubSeqLength=2*60*2;                                        %Mini-batch minimum subsequence length

%% LPTN structure related setups 
UserSetup.LPTN.NN_Inputs=1;                                                 %1 Use all InputFt as NN input
                                                                            %2 Use customed InputFt as NN input

%If predictable dummy nodes are chosen                                  
UserSetup.LPTN.NumOfConstantDummyNode=0;                                    %1 Number of dummy node
UserSetup.LPTN.NumOfPredictDummyNode=0;                                     %1 Number of dummy node
UserSetup.LPTN.NumOfLayer=2;                                                %NN number of layers
UserSetup.LPTN.NumNeuronPerLayer=20;                                        %NN number of neurons per layer except the output layer


%% Training related setup
% Specify options for Adam optimization.
UserSetup.Training.gradDecay = 0.5;                                         %Gradient decay rate 
UserSetup.Training.sqGradDecay = 0.5;                                     %Sequence gradient decay rate
UserSetup.Training.learnRate = 0.01;                                        %Learning rate

% itration and mini batch 
UserSetup.Training.numEpoch = 120;                                          %Number of eposhes 
UserSetup.Training.ValdAttempt = 70;                                        %Number of validation attempts for early stop
UserSetup.Training.MBPerItration=50;                                        %Number of mini-batches per itration
UserSetup.Training.TBTTNumLength=5*60*2;                                   %Truncated backpropagation through time length
end 