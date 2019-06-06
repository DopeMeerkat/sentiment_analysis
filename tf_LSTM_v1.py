# -*- coding: UTF-8 -*-
 
from io import open 
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from gensim import models
import pandas as pd
import jieba
import logging
import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Bidirectional,LSTM,Dense,Embedding,Dropout,Activation,Softmax 
from keras.layers import Conv2D, MaxPooling2D, Flatten, Convolution1D
from keras.utils import np_utils
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt 
import matplotlib as plotlib 
 

def parse(filename): 
    totals = []
    sentimentList = []
    wordList = []

    print('reading ' + filename)
    with open(filename, 'r', encoding="UTF-8") as f:
        for line in f:
            arr = line.split(' ', 9)
            total = arr[0].split('Total:')[1] 
            sentiments = [] 
            for i in range(1, 8):
                sentiments.append(float(arr[i].split(':')[1])) 
            sentiments.append(float(arr[8].split(':')[1].split('\t')[0]))   
 
            words = arr[9] 
            sentiments1 = sentiments.index(max(sentiments)) 
            totals.append(total)
            sentimentList.append(sentiments1)
            wordList.append(words)    

    return sentimentList, wordList
 

def train_word2vec(sentences,word_vec_dimension,save_path):
    sentences_seg = []
    sen_str = "\n".join(sentences)
    res = jieba.lcut(sen_str)
    seg_str = " ".join(res)
    sen_list = seg_str.split("\n")
    for i in sen_list:
        sentences_seg.append(i.split())
    print("starting to train word vector")  
    model = Word2Vec(sentences_seg,
                size=word_vec_dimension,  # dimension of word vector
                min_count=1,  # word frequency threshhold
                window=5)  # window size
    model.save(save_path)
    return model


def generate_id2wec(word2vec_model):
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
    w2id = {v: k + 1 for k, v in gensim_dict.items()}  # word index, start from 1
    w2vec = {word: model[word] for word in w2id.keys()}  # word vector
    n_vocabs = len(w2id) + 1
    embedding_weights = np.zeros((n_vocabs, 100))
    for w, index in w2id.items():  # Starting with a word with index 1, fill the matrix with a word vector
        embedding_weights[index, :] = w2vec[w]
    return w2id,embedding_weights
 
 
def text_to_array(w2index, senlist):  # Text to indexed number mode
    sentences_array = []
    for sen in senlist:
        new_sen = [ w2index.get(word,0) for word in sen]   # Word to index number
        sentences_array.append(new_sen)
    return np.array(sentences_array)
 
 
def prepare_data(w2id,sentences,labels,max_len=20):
    X_train, X_val, y_train, y_val = train_test_split(sentences,labels, test_size=0.2)
    X_train = text_to_array(w2id, X_train)
    X_val = text_to_array(w2id, X_val)
    X_train = pad_sequences(X_train, maxlen=max_len)
    X_val = pad_sequences(X_val, maxlen=max_len)
    return np.array(X_train), np_utils.to_categorical(y_train) ,np.array(X_val), np_utils.to_categorical(y_val)
  


class Sentiment:
    def __init__(self,w2id,embedding_weights,Embedding_dim,maxlen,labels_category):
        self.Embedding_dim = Embedding_dim
        self.embedding_weights = embedding_weights
        self.vocab = w2id
        self.labels_category = labels_category
        self.maxlen = maxlen
        self.model = self.build_model()
      
        
    def build_model(self):
        model = Sequential()
        #input dim(140,100)
        model.add(Embedding(output_dim = self.Embedding_dim,
                           input_dim=len(self.vocab)+1,
                           weights=[self.embedding_weights],
                           input_length=self.maxlen))
        model.add(Bidirectional(LSTM(50),merge_mode='concat'))
        model.add(Dropout(0.5))
        model.add(Dense(self.labels_category))
        model.add(Activation('softmax')) 

        model.compile(loss='categorical_crossentropy',
                     optimizer='adam', 
                     metrics=['accuracy'])
        model.summary()
        return model
    
 
    def train(self,X_train, y_train,X_test, y_test,n_epoch=5 ):
        
 

        history = self.model.fit(X_train, y_train, batch_size=32, epochs=n_epoch,
                     validation_data=(X_test, y_test), callbacks=[metrics])
        self.model.save('sentiment.h5')   


        history_dict = history.history
        history_dict.keys()

        #dict_keys(['loss', 'acc', 'val_loss', 'val_acc'])
 
        loss = history_dict['loss']
        acc = history_dict['acc']
        val_loss = history_dict['val_loss']
        val_acc = history_dict['val_acc']

        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy') 
        plt.legend(loc=7)

        plt.show()
        
 
    def predict(self,model_path,new_sen):
        model = self.model
        model.load_weights(model_path)
        new_sen_list = jieba.lcut(new_sen)

        while(' ' in new_sen_list) : 
          new_sen_list.remove(' ')  

        sen2id =[ self.vocab.get(word,0) for word in new_sen_list]
        sen_input = pad_sequences([sen2id], maxlen=self.maxlen)
        res = model.predict(sen_input)[0]
        return np.argmax(res)
 

class Metrics(Callback):

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
 
    def on_epoch_end(self, epoch, logs={}): 

        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]

        _val_f1 = f1_score(val_targ, val_predict, average='micro')
        _val_recall = recall_score(val_targ, val_predict, average='micro')
        _val_precision = precision_score(val_targ, val_predict, average='micro')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
   
        print("F1 Score : ",_val_f1)
        print("Precision Score : ",_val_precision)
        print("Recall Score : ",_val_recall)
  
        return 

if __name__ == '__main__':  

    word_vec_dimension = 100
    word_max_length = 500
    labels_category = 8

    sentimentList, wordList = parse('sina/sinanews.train')  
    sentimentList_test, wordList_test = parse('sina/sinanews.test')   
     
    model =  train_word2vec(wordList,word_vec_dimension,'word2vec.model')
    w2id,embedding_weights = generate_id2wec(model)
    x_train,y_train, x_val , y_val = prepare_data(w2id,wordList,sentimentList,word_max_length)  

    metrics = Metrics()

    senti = Sentiment(w2id,embedding_weights,word_vec_dimension,word_max_length,labels_category)
    
    senti.train(x_train, y_train, x_val, y_val, 30)  
 

    category_names = {0:"感动",1:"同情",2:"无聊",3:"愤怒",4:"搞笑",5:"难过",6:"新奇",7:"温馨"}
    
    index_array = []
    pre_array = []
    label_array = []
    fout = open("LSTM_out.txt", "w", encoding='utf-8')
    index = 0
    for sen_new in wordList_test:
 
      pre = senti.predict("./sentiment.h5",sen_new) 
      pre_array.append(pre)
      index_array.append(index)
      fout.write("{}\t{}\n".format(index, category_names.get(pre)))
      #print("'{}'的情感是:\n{}".format(index, category_names.get(pre)))
      #print("label:\n{}\n".format(category_names.get(sentimentList_test[index])))
      label_array.append(sentimentList_test[index])
      index = index + 1

    fout.close()

    #print(pre_array)
    #print(label_array)
    np.corrcoef(pre_array, label_array)
    plotlib.style.use('ggplot')

    plt.scatter(pre_array, label_array)
    plt.show()



 
    plt.plot(index_array, pre_array, 'bo', label='prediction')
    plt.plot(index_array, label_array, 'b', label='label')
    plt.title('Prediction vs Label')
    plt.xlabel('Epochs')
    plt.ylabel('Prediction') 
    plt.legend(loc=7)




