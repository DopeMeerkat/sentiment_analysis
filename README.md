# Sentiment Analysis Experiment Report
## Abstract
This code uses Tensorflow’s backend to perform sentiment analysis by using the corpora given to us. The are two implementations of the code, one using CNN and the other LSTM. The code trains by using `sina/sinanews.train`, then validates using `sina/sinanews.test`.

# Implementation

This programming experiment implements two models (CNN and RNN) and applies it to the emotion classification task. RNN is an LSTM type. The language of the code uses Python, which uses the Keras API and the deep learning framework TensorFlow.

The implementation flow of the two models is the same:

1. Import the training text data set sinanews.train
2. Create a dictionary of data from the Word2Vec model, starting with index 1
3. Convert Chinese characters to numeric IDs through a dictionary and form a vector
4. Define the model
5. Train, define the callback function, calculate the Precision Score, Recall Scrore and F1-Score after each epoch
6. Sketch the Accuracy Vs. Epoch curve for Training and Validation after the training
7. Import the test text dataset sinanews.test, perform sentiment analysis through the model
8. Finally, calculate the correlation coefficient and draw the result distribution map.
### CNN:
We first truncate each line from sinanews so that they’re all the same size, then we use the embedding layer to convert each word encoding into a word vector. We split the training data into a training set and a validation set, with a 4:1 ratio. We then build the model with Convolution1D kernels, with sigmoid as an activation function and loss using a logarithmic loss function. After the model is done fitting multiple times, we display the graph from pyplot, and then input the test entries from sinanews.test. Pyplot then plots the distribution of the sentiment for predicted vs found for the results.
### LSTM:
LSTM is implemented the same way as the CNN script with the exception of the model itself. We configure the model to be Bidirectional(LSTM), which takes care of the rest of the configurations for us.
## Experimental Results
Experimental results can be found in `results`
- In this experiment, the CNN model runs much faster than the LSTM model.   
- Both models can achieve a high accuracy for training dataset after some epochs, CNN is much faster though. 
- CNN can achieve an accuracy nearly 80% for validation dataset, for LSTM, it is only around 50%.
- The F1-Score for both is under 0.5, it means the model is not working so well yet.
