Purpose: To model and train thermal neural network (TNN)
This code has 6 main section 
1) UserInit.m:
The user can configure dataset, modeling and training related setups. 
2) data_prep.m: 
The raw dataset is preprocessed for training perpuse
3) createMiniBatch.m: 
The pre-processed dataset is rearranged in mini-batches to ease the tarining process 
4) createNNParametrs.m: 
The TNN parameters i.e. power loss NN, thermal conductance NN and inverted-thermal capacitance are defined and initialized here 
5) TNN_model.m:
It is the TNN model wich forwards the input features and predicts the target
