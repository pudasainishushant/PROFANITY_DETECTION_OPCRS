#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


name_df = pd.read_csv("name_data.csv",sep="\t")


# In[3]:


name_df.shape


# In[4]:


profane_df = pd.read_csv("profane_words.csv",sep="\t")


# In[5]:


profane_df.shape


# In[6]:


final_df = name_df.append(profane_df, ignore_index=True)


# In[7]:


final_df=final_df.sample(frac=1).reset_index(drop=True)


# In[8]:


final_df.isna().sum()


# In[9]:


final_df.head(n=10)


# In[10]:


for l in final_df["TEXT"]:
    if type(l) != str:
        print("bad")
        print(l)


# In[11]:


final_df.loc[:,"TEXT"] = final_df.TEXT.apply(lambda x : str(x))


# Data splitting

# In[13]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
vocabulary_size = 20000
tokenizer = Tokenizer(char_level=False, oov_token='UNK',num_words = vocabulary_size)
tokenizer.fit_on_texts(final_df["TEXT"])
sequences = tokenizer.texts_to_sequences(final_df['TEXT'])
data = pad_sequences(sequences,maxlen = 10)


# In[14]:


tokenizer.index_word


# In[15]:


final_df["LABEL"] = final_df["LABEL"].replace(["NAME" , "NOTNAME"] , [1 , 0])


# In[16]:


labels = final_df["LABEL"]


# In[17]:


from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Dense,  LSTM, Conv1D, MaxPooling1D, Dropout, Activation
import numpy as np
from keras.layers import BatchNormalization


# In[31]:


def create_model():
    model2 = Sequential()
    model2.add(Embedding(20000, 50, input_length=10))
    model2.add(LSTM(50, dropout=0.2, recurrent_dropout=0.2))
    model2.add(BatchNormalization())
    model2.add(Dropout(0.4))
    model2.add(Dense(50, activation="tanh"))
    model2.add(BatchNormalization())
    model2.add(Dropout(0.4))
    model2.add(Dense(50, activation="tanh"))
    model2.add(BatchNormalization())
    model2.add(Dropout(0.4))
    model2.add(Dense(1, activation="sigmoid"))
    model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model2


# In[32]:


from keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn = create_model)


# In[ ]:


from sklearn.model_selection import GridSearchCV
param_grid = dict(epochs=[10,20,30])
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(data, np.array(labels))


# In[ ]:




