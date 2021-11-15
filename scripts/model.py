import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Dense,  LSTM, Conv1D, MaxPooling1D, Dropout, Activation
import numpy as np
from keras.layers import BatchNormalization
import pickle


vocabulary_size = 20000

class ModelName:
	def __init__(self,name_path,profane_path):
		self.name_df = pd.read_csv(name_path,sep="\t")
		self.profane_df = pd.read_csv(profane_path,sep="\t")
		

	def train_model(self):
		final_df = self.name_df.append(self.profane_df, ignore_index=True)
		final_df=final_df.sample(frac=1).reset_index(drop=True)
		final_df.loc[:,"TEXT"] = final_df.TEXT.apply(lambda x : str(x))
		final_df["LABEL"] = final_df["LABEL"].replace(["NAME" , "NOTNAME"] , [1 , 0])
		labels = final_df["LABEL"]
		

		tokenizer = Tokenizer(char_level=False, oov_token='UNK',num_words = vocabulary_size)
		
		tokenizer.fit_on_texts(final_df["TEXT"])
		with open('serializer/tokenizer.pickle', 'wb') as handle:
			pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

		sequences = tokenizer.texts_to_sequences(final_df['TEXT'])
		data = pad_sequences(sequences,maxlen = 10)
		model2 = Sequential()
		model2.add(Embedding(20000,50,input_length=10))
		model2.add(LSTM(50,dropout=0.2, recurrent_dropout=0.2))
		model2.add(BatchNormalization())
		model2.add(Dropout(0.4))
		model2.add(Dense(50,activation="tanh"))
		model2.add(BatchNormalization())
		model2.add(Dropout(0.4))
		model2.add(Dense(50,activation="tanh"))
		model2.add(BatchNormalization())
		model2.add(Dropout(0.4))
		model2.add(Dense(1,activation="sigmoid"))
		model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		model2.fit(data,np.array(labels),validation_split=0.4, epochs=5)
		model_json = model2.to_json()
		with open("serializer/model.json", "w") as json_file:
			json_file.write(model_json)
		model2.save_weights("serializer/model.h5")
		#evaluate_model
		# evaluate the model
		scores = model2.evaluate(final_test_data, y_test, verbose=0)
		print("%s: %.2f%%" % (model2.metrics_names[1], scores[1]*100))
		#splitting training and testing data
		from sklearn.model_selection import train_test_split
		X_train, X_test, y_train, y_test = train_test_split(final_df["TEXT"], final_df["LABEL"], test_size=.33, random_state=17)
		sequences_for_test = tokenizer.texts_to_sequences(X_test)
		final_test_data = pad_sequences(sequences_for_test,maxlen = 10)
		y_pred = model2.predict(final_test_data)
		new_y_pred = []
		for i in y_pred:
		    if i>0.5:
		        new_y_pred.append("NAME")
		    else:
		        new_y_pred.append("NOTNAME")
		new_y_test = y_test.tolist() 
		print(confusion_matrix(y_test, new_y_pred))