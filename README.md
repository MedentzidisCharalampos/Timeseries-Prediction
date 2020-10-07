# Timeseries-Prediction
In this project we predict the new value in  a sequence of values. The data was artificially made from a combination of trend, seasonality and noise patterns. A Deep Neural Network architecture is chosen for that  task.   

There are three implementations:   
1. Using dense layers  (time_series_pred_dnn.py)
2. Using Recurrent Neural Networks(RNNs)  (time_series_pred_rnn.py)
2. Using dense and long-short-term (lstm) layers  (time_series_pred_lstm.py)
3. Using dense, lstm and convolutional layers(cnn) layers  (time_series_pred_cnn_lstm.py)

The data is divided into features and labels, the features is a number of values in the series and our label is the next value. The number of values is called the windows size, where we take the window of the data and predicting the next value.

The dataset will be created from the input data as follows:  
1. Use a window of data of specific size, which is shifted each time slice and is regulararized to have the same size each time slice  
2. Put the data into numpy lists  
3. Split the data into features and labels, the last time slice is the label and all the previous is the features  
4. The data is shuffled before training  
5. Batch the data into sets  

The dataset is splited into training and validation set at time step 1,000.

1. Using dense layers:

    The architecture of the model is as follows:
    1. A dense layer of 10-units that takes as input the window size and activation function Relu
    2. A dense layer of 10-units  and activation function Relu
    3. A dense layer of 1-unit

    The model compiled with mean squared error loss function and stochastic gradient descent optimizer with 8e-6 learning rate. 
    The model is trained for 500 epochs. The mean absolute error after training is 4.4847784 


2. Using Recurrent Neural Networks(RNNs) 

This imlpementation uses Reccurent Neural Networks (RNN). A RNN is a neural network that contains recurrent layers. These are designed to sequentially process sequences of inputs.  The full input shape when using RNN is three dimensional. The first dimension is the batch size, the second is the timestamps and the thrird is the dimensionality of the inputs at each time step. In our dataset tha dimensionality of the inputs is 1.
The recurrent layer consists of one cell that is used repeatedly to compute the outputs. At each time step the memory cell takes the input value for that step and then caluculate the output for that step, and a state vector that is fed into the next time step. These steps will continue until we reach the end of the input dimension. This is what gives the name recurrent neual network, because the output recur due to the output of the cell, a one-step being fed back into itself to the next time step. In order the recurrent layer to output a sequence, we specify return sequences equal to true because we are going to stack two rnn layers together.

The architecture of the model is as follows:
1. A Lamba layer, expands the array by one dimension and input_shape = None which means that the model can take sequences of any length
2. A Recurrent Layer that has return_sequences = True, it will output a sequence which is fed to the next layer and input_shape batch_size x timestamps = 1.
3. A Recurrent Layer that will only output to the final step
4. A Dense Layer of 1-unit
5. A Lamba Layer scale the outputs by 100. The default activation function is tanH which has out puts in range [-1, 1] , since the timeseries values are in that order usually 40s, 50s e.t.c., then scalling up the outputs can help us with learning

The model compiled with Huber loss function  that's less sensitive to outliers as the data is a bit noisy and stochastic gradient descent optimizer with 5e-5 learning rate. 
The model is trained for 400 epochs. The mean absolute error after training is 6.4141674.


3. Using Long Short Term Memory Networks(LSTMs):

Long Short Term Memory Networks are the cell state that keep the state throughout the life of the training so that the state is passed from cell to cell, timestamp to timestamp and can be better maintained. This means that the data from earlier in the window can havea greater impact on the overall projection than in the case of RNNs. The statecan also be bidirectional so that state moves forwards and backwards. 


The architecture of the model is as follows:
1. A Lamba layer, expands the array by one dimension and input_shape = None which means that the model can take sequences of any length
2. A Long Short Term Memory Layer with 32 cells that is bidirectional and has return_sequences = True, it will output a sequence which is fed to the next layer
3. A Long Short Term Memory Layer with 32 cells that is bidirectional 
4. A Dense Layer of 1-unit will give the output prediction value
5. A Lamba Layer scale the outputs by 100. The default activation function is tanH which has out puts in range [-1, 1] , since the timeseries values are in that order usually 40s, 50s e.t.c., then scalling up the outputs can help us with learning

The model compiled with Huber loss function  that's less sensitive to outliers as the data is a bit noisy and stochastic gradient descent optimizer with 1e-6 learning rate. 
The model is trained for 100 epochs. The mean absolute error after training is 5.2872233.


