import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Dense,  LSTM,Dropout, Activation
from keras.layers import BatchNormalization

import tensorflow as tf
import pickle
from keras.models import load_model
from keras import backend as K
from sklearn.model_selection import train_test_split


class CheckName():
	def __init__(self):
		self.name_identifier_model = load_model('serializer/name_not_name_identifier_bilstm.h5')
		with open('serializer/name_not_name_tokenizer.pickle', 'rb') as handle:
			self.tokenizer = pickle.load(handle)

	def train_model(self,name_path,notname_path):
		# read existing name list in excel as a pandas dataframe
		name_df = pd.read_excel(name_path, delimiter="\t")

		# read existing not names dataset in excel file as a pandas dataframe
		notname_df = pd.read_excel(notname_path, delimiter="\t")
		# name_df, and notname_df has different column names, so concatination will cause different dataframe
		notname_df.rename(columns={'TEXT': 'Full Name', 'LABEL': 'Label'}, inplace=True)

		name_notname_mixed_dataset = pd.concat([name_df,notname_df])
		vocabulary_size = 60
		tokenizer = Tokenizer(char_level=True, oov_token='UNK', num_words=vocabulary_size)
		tokenizer.fit_on_texts(name_notname_mixed_dataset["Full Name"].astype(str))

		sequences = tokenizer.texts_to_sequences(name_notname_mixed_dataset['Full Name'].astype(str))
		data = pad_sequences(sequences, maxlen=20)

		X_train, X_test, y_train, y_test = train_test_split(data, name_notname_mixed_dataset["Label"],
															test_size=0.33, random_state=17)

		model = Sequential()
		model.add(Embedding(136, 50, input_length=20))
		model.add(LSTM(50, dropout=0.2, recurrent_dropout=0.2))
		model.add(BatchNormalization())
		model.add(Dropout(0.4))
		model.add(Dense(50, activation="relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.4))
		model.add(Dense(50, activation="relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.4))
		model.add(Dense(1, activation="sigmoid"))
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		model.fit(X_train, np.array(y_train), validation_split=0.4, epochs=5)

		# saving trained model
		with open('serializer/name_not_name_identifier.pickle', 'wb') as handle:
			pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

		# saving tokenizer
		with open('serializer/name_not_name_tokenizer.pickle', 'wb') as handle:
			pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

	def testing_name_notname(self, input_text):
		# K.clear_session()
		test_sequences = self.tokenizer.texts_to_sequences([input_text])
		test_padded_data = pad_sequences(np.array(test_sequences), maxlen=20,
										 padding='pre', truncating='pre')
		 
		prediction = self.name_identifier_model.predict(test_padded_data)
		result = "True" if prediction > 0.6 else "False"
		# K.clear_session()
		return result 
		# K.clear_session()

	def check_name(self,data):
		for d in data['data']:
			if 'isName' in d:
				name = self.testing_name_notname(d['headValue'])
				if name=="False":
					d['isName'] = "False"
					d['name_msg'] = "Please enter the correct name"
		return data
	

data = {
	"data":
	[
        {
            "headName":"Name",
            "isName":"True",
            "headValue":"Jit Bahadur Khamcha",
            "status":"True",
            "message":""
        },
        {
            "headName":"FathersName",
            "isName":"True",
            "headValue":"Fuck Bd fuck",
            "status":"True",
            "message":""
        },
        {
            "headName":"status",
            "headValue":"Fuck",
            "status":"True",
            "message":""
        },
        {
            "headName":"Discription",
            "headValue":"fuck you",
            "status":"True",
            "message":""
        }
    ]
}


# name_model = CheckName()
# print(name_model.check_name(data))

# def check_name(data):
# 	for name in data:
# 		if 'isName' in name:
# 			full_name = name['headValue']
# 			# print(full_name)
# 			name_status = name_model.testing_name_notname(full_name)
# 			# print(name_status)
# 	return name_status
			
    


# for data in data['data']:
# 	print(data)
# 	name = name_model.testing_name_notname(data['headValue'])
# 	if name=="False":
# 		data['isName'] = "False"

# print(data)
# # # full_name = "HDBF FDGDG"


# # # name_status = name_model.testing_name_notname(full_name)
# # # print(name_status)