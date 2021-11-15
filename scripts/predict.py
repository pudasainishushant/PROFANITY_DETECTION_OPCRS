from keras.models import model_from_json
import pickle
import pandas as pd
from keras.preprocessing.sequence import pad_sequences


class Predict:
	def __init__(self):
		json_file = open('models/model.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		self.loaded_model = model_from_json(loaded_model_json)
		self.loaded_model.load_weights("models/model.h5")
		with open('models/tokenizer.pickle', 'rb') as handle:
			self.tokenizer = pickle.load(handle)

	def predict_name(self,test_data_path):

		test_df = pd.read_csv(test_data_path,sep="\t")
		test = test_df["text"]
		#test_label = test_df["label"]
		sequences_test = self.tokenizer.texts_to_sequences(test)
		test_data = pad_sequences(sequences_test,maxlen = 10)
		test_pred = self.loaded_model.predict(test_data)
		new_pred = []
		for i in test_pred:
			if i>0.5:
				new_pred.append("NAME")
			else:
				new_pred.append("NOTNAME")
		result = [(i,j) for i,j in zip(test,new_pred)]
		return(result)

	def predict_single(self,name):
		sequences_test = self.tokenizer.texts_to_sequences(name)
		sequence = pad_sequences(sequences_test,maxlen = 10)
		prediction = self.loaded_model.predict(sequence)
		if (prediction>0.5):
			return("NAME")
		else:
			return("NOTNAME")
