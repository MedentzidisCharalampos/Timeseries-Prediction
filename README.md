# Timeseries-Prediction
In this project we predict the new value in  a sequence of values. The data was artificially made from a combination of trend, seasonality and noise patterns. A Deep Neural Network architecture is chosen for that  task.

The data is divided into features and labels, the features is a number of values in the series and our label is the next value. The number of values is called the windows size, where we take the window of the data and predicting the next value.

The dataset will be created from the input data as follows:  
1. Use a window of data of specific size, which is shifted each time slice and is regulararized to have the same size each time slice  
2. Put the data into numpy lists  
3. Split the data into features and labels, the last time slice is the label and all the previous is the features  
4. The data is shuffled before training  
5. Batch the data into sets  

The dataset is splited into training and validation set at time step 1,000.

The architecture of the model is as follows:
1. A dense layer of 10-units that takes as input the window size and activation function Relu
2. A dense layer of 10-units  and activation function Relu
3. A dense layer of 1-unit

The model is compiles with mean squared error loss function and stochastic gradient descent optimizer with 8e-6 learning rate. 
The model is trained for 500 epochs. The mean squared error after training is 4.4847784 
