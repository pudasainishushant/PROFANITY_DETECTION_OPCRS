# Basic packages
import pandas as pd 
import numpy as np
import nltk
import re
import collections
# import matplotlib.pyplot as plt
from pathlib import Path

import pickle
# Packages for data preparation
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import LSTM
from keras.models import load_model
from keras import backend as K
from sklearn.preprocessing import LabelEncoder

# Packages for modeling
from keras import models
from keras import layers
from keras import regularizers

# lstm_model = model = load_model('output/model/model.h5')


class CharProfanity():
    def __init__(self):
        self.NB_WORDS = 10000  # Parameter indicating the number of words we'll put in the dictionary
        self.VAL_SIZE = 1000  # Size of the validation set
        self.NB_START_EPOCHS = 20  # Number of epochs we usually start to train with
        self.BATCH_SIZE = 512  # Size of the batches used in the mini-batch gradient descent
        self.MAX_LEN = 24  # Maximum number of words in a sequence
        self.GLOVE_DIM = 50  # Number of dimensions of the GloVe word embeddings
        self.lstm_out = 20
        self.profanity_model = load_model('output/model/model.h5')
    
    def deep_model(self,model, X_train, y_train, X_valid, y_valid):
        '''
        Function to train a multi-class model. The number of epochs and 
        batch_size are set by the constants at the top of the
        notebook. 
        
        Parameters:
        model : model with the chosen architecture
        X_train : training features
        y_train : training target
        X_valid : validation features
        Y_valid : validation target
        
        Output:
        model training history
        '''
        model.compile(optimizer='rmsprop'
                  , loss='categorical_crossentropy'
                  , metrics=['accuracy'])
        model.fit(X_train
                    , y_train
                    , epochs=self.NB_START_EPOCHS
                    , batch_size=self.BATCH_SIZE
                    , validation_data=(X_valid, y_valid)
                    , verbose=1)
        model.save("./output/model/model.h5")
    
    
    def eval_metric(history, metric_name):
        '''
        Function to evaluate a trained model on a chosen metric. 
        Training and validation metric are plotted in a
        line chart for each epoch.
        Parameters:
        history : model training history
        metric_name : loss or accuracy
        Output:
        line chart with epochs of x-axis and metric on
        y-axis
        '''
        metric = history.history[metric_name]
        val_metric = history.history['val_' + metric_name]
        
        e = range(1, self.NB_START_EPOCHS + 1)
        
        
    def test_model(self,model, X_train, y_train, X_test, y_test, epoch_stop):
        '''
        Function to test the model on new data after training it
        on the full training data with the optimal number of epochs.
        Parameters:
        model : trained model
        X_train : training features
        y_train : training target
        X_test : test features
        y_test : test target
        epochs : optimal number of epochs
        Output:
        test accuracy and test loss
        '''
        model.fit(X_train
              , y_train
              , epochs=epoch_stop
              , batch_size=self.BATCH_SIZE
              , verbose=0)
        results = model.evaluate(X_test, y_test)
    
        return results
        
    def remove_stopwords(self,input_text):
        '''
        Function to remove English stopwords from a Pandas Series.
        Parameters:
        input_text : text to clean
        Output:
        cleaned Pandas Series 
        '''
        stopwords_list = stopwords.words('english')
        # Some words which might indicate a certain sentiment are kept via a whitelist
        whitelist = ["n't", "not", "no"]
        words = input_text.split() 
        clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1] 
        return " ".join(clean_words) 
        
    def remove_mentions(self,input_text):
        '''
        Function to remove mentions, preceded by @, in a Pandas Series
        Parameters:
        input_text : text to clean
        Output:
        cleaned Pandas Series 
        '''
        return re.sub(r'@\w+', '', input_text)

    def process_dataset(self):
        input_path = './input'
        df = pd.read_csv('input/train.csv')
        df = df.reindex(np.random.permutation(df.index))  
        df = df[['comment_text', 'toxic']]
        df.text = df.comment_text.apply(self.remove_stopwords).apply(self.remove_mentions)
        X_train, X_test, y_train, y_test = train_test_split(df.comment_text, df.toxic, test_size=0.1, random_state=37)
        print('# Train data samples:', X_train.shape[0])
        print('# Test data samples:', X_test.shape[0])
        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]
        tk = Tokenizer(num_words=self.NB_WORDS,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            char_level=True,
            lower=True,
            split=" ")
        tk.fit_on_texts(X_train)
        with open('./output/model/tk.pickle', 'wb') as handle:
            pickle.dump(tk, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # X_train,X_test,y_train,y_test = self.split_dataset()
        X_train_seq = tk.texts_to_sequences(X_train)
        X_test_seq = tk.texts_to_sequences(X_test)
        
        seq_lengths = X_train.apply(lambda x: len(x.split(' ')))
        seq_lengths.describe()

        X_train_seq_trunc = pad_sequences(X_train_seq, maxlen=self.MAX_LEN)
        X_test_seq_trunc = pad_sequences(X_test_seq, maxlen=self.MAX_LEN)
        
        le = LabelEncoder()
        y_train_le = le.fit_transform(y_train)
        y_test_le = le.transform(y_test)
        y_train_oh = to_categorical(y_train_le)
        y_test_oh = to_categorical(y_test_le)
        
        X_train_emb, X_valid_emb, y_train_emb, y_valid_emb = train_test_split(X_train_seq_trunc, y_train_oh, test_size=0.1, random_state=37)

        assert X_valid_emb.shape[0] == y_valid_emb.shape[0]
        assert X_train_emb.shape[0] == y_train_emb.shape[0]

        print('Shape of validation set:',X_valid_emb.shape)

        return X_train_emb, X_valid_emb, y_train_emb, y_valid_emb,X_train_seq_trunc,X_test_seq_trunc,y_train_oh,y_test_oh


    def load_pre_trained_emb(self):
        glove_file = 'glove.twitter.27B.50d.txt'
        glove_dir = 'glove/'
        input_path = './input'
        emb_dict = {}
        glove = open(input_path / glove_dir / glove_file)
        for line in glove:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            emb_dict[word] = vector
        glove.close()
        return emb_dict


    def fit_emb(self):
        
        emb_matrix = np.zeros((self.NB_WORDS, self.GLOVE_DIM))
        tk = self.tokenize_data()
        emb_dict = self.load_pre_train_emb()
        for w, i in tk.word_index.items():
            # The word_index contains a token for all words of the training data so we need to limit that
            if i < self.NB_WORDS:
                vect = emb_dict.get(w)
                # Check if the word from the training data occurs in the GloVe word embeddings
                # Otherwise the vector is kept with only zeros
                if vect is not None:
                    emb_matrix[i] = vect
            else:
                break
        return vect



    def lstm_model(self):
        emb_model2 = models.Sequential()
        emb_model2.add(layers.Embedding(self.NB_WORDS, self.GLOVE_DIM, input_length=self.MAX_LEN))
        emb_model2.add(LSTM(self.lstm_out, dropout=0.2, recurrent_dropout=0.2))
        emb_model2.add(layers.Dense(2, activation='softmax'))
        emb_model2.summary()
        return emb_model2


    def train_model(self):
        X_train_emb, X_valid_emb, y_train_emb, y_valid_emb,X_train_seq_trunc,X_test_seq_trunc,y_train_oh,y_test_oh = self.process_dataset()
        emb_model2 = self.lstm_model()
        emb_history2 = self.deep_model(emb_model2, X_train_emb, y_train_emb, X_valid_emb, y_valid_emb)
        emb_results2 = self.test_model(emb_model2, X_train_seq_trunc, y_train_oh, X_test_seq_trunc, y_test_oh, 3)
        print('/n')
        print('Test accuracy of word embedding model 2: {0:.2f}%'.format(emb_results2[1]*100))
        return emb_history2,emb_results2

    def load_model(self):
        model = load_model('output/model/model.h5')
        return model

    
    
    def check_words(self,sentence):
        # filepath = os.getcwd()+'/Data/word_list.txt'
        with open('./input/rough_words.txt','r') as word_file:
            words = [line for line in word_file]
        stripped_words = [s.rstrip().lower() for s in words]
        sentence_tokens = nltk.word_tokenize(sentence.lower())
        for tokens in sentence_tokens:
            if tokens in stripped_words:
                return True

    # # @staticmethod
    # def text_subset(self,text):
    #     n = len(text)  
    #     #For holding all the formed substrings  
    #     arr = []
    #     #This loop maintains the starting character  
    #     for i in range(0,n):  
    #         #This loop will add a character to start character one by one till the end is reached  
    #         for j in range(i,n):  
    #             arr.append(text[i:(j+1)])
    #     return arr


    def predict(self,request):
        with open('./output/model/tk.pickle', 'rb') as handle:
            tk = pickle.load(handle)
        for d in request['data']:
            # print(d['headValue'])
            api_data = d['headValue']
            print(api_data)
            nepali_rough = self.check_words(api_data)
            # tokenize_data = nltk.word_tokenize(api_data.lower())
            check_data = [api_data]
            if nepali_rough == True:
                d['status'] = "False"
                d['message'] = "Text is profane"
            else:
                pred = tk.texts_to_sequences(check_data)
                pred = pad_sequences(pred, maxlen=24, dtype='int32', value=0)
                # model = load_model('output/model/model.h5')
                profanity = self.profanity_model.predict(pred,batch_size=1,verbose = 2)[0]
                # K.clear_session()
                if(np.argmax(profanity) == 0):
                    d['status'] = 'True'
                    
                elif (np.argmax(profanity) == 1):
                    d['status'] = "False"
                    d['message'] = "Text is profane"
            
            
        return request
    


    
    # def predict(self,request):
    #     name_checked_words = []
    #     profanity_checked_words = []
    #     with open('./output/model/tk.pickle', 'rb') as handle:
    #         tk = pickle.load(handle)
    #     for d in request['data']:
    #         # print(d['headValue'])
    #         api_data = d['headValue']
    #         # print("HElllllllllllllo",api_data)
    #         test_data = self.text_subset(api_data)
    #         # print(test_data)
    #         for i in test_data:
    #             # print(i)
                
    #             nepali_rough = self.check_words(i)
    #             name_checked_words.append(nepali_rough)
    #             # print(checked_words)
    #             # tokenize_data = nltk.word_tokenize(api_data.lower())
    #             check_data = [i]
    #             # print(check_data)
    #             # if True in name_checked_words:
    #             #     d['status'] = "False"
    #             #     d['message'] = "Text is profane"

    #             # else:
    #             pred = tk.texts_to_sequences(check_data)
    #             pred = pad_sequences(pred, maxlen=24, dtype='int32', value=0)
    #             # print(pred)
    #             # model = load_model('output/model/model.h5')
    #             profanity = self.profanity_model.predict(pred,batch_size=1,verbose = 2)[0]
    #             # K.clear_session()
    #         # name_checked_words.append(nepali_rough)
    #             if(np.argmax(profanity) == 0):
    #                 profanity_checked_words.append(True)
    #                 # d['status'] = 'True'
    #             elif (np.argmax(profanity) == 1):
    #                 profanity_checked_words.append(False)
            
        
    #         if True in name_checked_words:
    #             d['status'] = "False"
    #             d['message'] = "Text is profane"
    #         else:
    #             if False in profanity_checked_words:
    #                 d['status'] = "False"
    #                 d['message'] = "Text is profane"
    #             else:
    #                 d['status'] = 'True'
    #         print(profanity_checked_words)
    #         print(name_checked_words)
            
    #         name_checked_words[:] = []
    #         profanity_checked_words[:] = []

        
    #                 # elif False in profanity 
    #         #             # d['status'] = "False"
    #         #             # d['message'] = "Text is profane"

                    
            
            
    #     return request

        

request ={
    "data":
    [
        {
            "headName":"FirstName",
            "headValue":"Nitesh",
            "status":"True",
            "message":""
        },
        {
            "headName":"MiddleName",
            "headValue":"",
            "status":"True",
            "message":""
        },
        {
            "headName":"LastName",
            "headValue":"suck my dick",
            "status":"True",
            "message":""
        },
        {
            "headName":"Discription",
            "headValue":"fuck you boy",
            "status":"True",
            "message":""
        }
    ]
}


# from name_check import CheckName

# check_name(request)

# str = "hellofuckerkchaterochalasalegadhakukurbautalaikhakopachenayeramechorkukurbhatesaletalaihandinchu"
# profanity = CharProfanity()
# profanity.train_model()
# profanity.train_model()
# print(profanity.subset(str))
# model = profanity.load_model()
# print(model)
# print(profanity.predict(request))
# ct(request))
# a = model.check_words("hi hello how are you  khate pussy")
# if a==True:
#     print("True")
# else:
#     print("False")
# print(profanity.predict(request))