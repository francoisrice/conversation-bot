# Collection of chatbot models, each held within its own class 
# The imports needed for each model are presented before that class

# Vanilla three lawyer artificial neural network

from keras.models import Sequential # basic Keras model
from keras.layers import Dense, Activation # Necessary functions for NN layers and computation

class NLPv1(object):
	def __init__(self,name):
		self.name = name

	def create(self, name):
		self.name = str(name)
		model = Sequential([

			# creates a hiidden layer with 12 nodes and an input layer
			# with 20 nodes
		   Dense(12, input_shape=(20,)), 
		  #Dense(12, input_dim=20), # same as above

		  # Set softmax as the normalizing function for hidden layer outputs
		   Activation('softmax'),

		   Dense(5), # create a 5 node output layer
		   Activation('softmax'), # Give the output layer a softmax function too
		])

		# This is for reference if you wanted the NN to be deeper
		#model.add to add layers
		#model.add(Dense(32, activation='relu'))

		return model

	def train(self,model,data,labels,batch,epoch):

		# Set the model to train with the following optimization function
		#to find the best 'metric' listed
		model.compile(optimizer='rmsprop',
			loss='categorical_crossentropy',
			metrics=['accuracy'])

		# Train the model to the data
		#train with 32 samples/batches (batch: number of samples before backproping)
		model.fit(data, labels, epochs=epoch, batch_size=batch)

		return model

	def save(self,name,model):
		# save the weights of the model once trained
		name = str(name)
		try:
			model.save_weights(name+".h5")
			print("Model weights saved successfully.")
		except:
			raise('Unknown error risen while saving weights. Perhaps try again.')
		return model

	def load(self,name):
		name= str(name)
		model = Sequential([
		   Dense(12, input_shape=(20,)),
		  #Dense(12, input_dim=20), # same as above
		   Activation('softmax'),
		   Dense(5),
		   Activation('softmax'),
		])

		#To reload the weights later, use:
		model.load_weights(name+".h5")
		return model

	def query(self,model,input,batch_number=1):
		return model.predict(input, batch_size=batch_number)

		#test the model
		#model.evaluate(input_data,output_data,batch_size=sample_number)

		#query the model
		#model.predict(input_data, batch_size=number_of_samples_for_prediction)

from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np

class SeqtoSeq(object):
	""" NLP model taken from Keras' example of Seq2Seq for language translation. Should also work 
	adequately on text generation and as a starting point for more advanced models. Keras' model 
	is built to translate based on character level, but this model uses a word level text 
	generation. This model uses Keras' built in LSTM function. It would also be good to build 
	this model from stratch, but then again, that would just be like rewritting keras, which isn't 
	useful."""
	
	def __init__(self,batch_size=64,epochs=100,latent_dim=256,num_samples=10000): # All of this inf
		self.batch_size = batch_size
		self.epochs = epochs
		self.latent_dim = latent_dim
		self.num_samples = num_samples
		
	def create(self):
		
		enoder_input_data = np.zeros()
		
		return True

	def train(self,sample,target):

		# Vectorize the data
		sample = sample.split(' ')
		target = target.split(' ')
		sample = sorted(list(sample))
		target = sorted(list(target))
		num_encoder_tokens = len(sample)
		num_decoder_tokens = len(target)
		
		# The max encoder length should be set based on multiple samples, 
		#however this function is designed to take in a single input. Thus
		#the following lines are commented out.
		#max_encoder_seq_length = max([len(txt) for txt in sample_texts])
		#max_decoder_seq_length = max([len(txt) for txt in target_texts])
		
		sample_token_index = dict([(char, i) for i, char in enumerate(sample)])
		target_token_index = dict([(char, i) for i, char in enumerate(target)])
		
		
		#TODO - Put lines 103 to 111 in the create() function
		#TODO - continue implementing Seq2Seq example, have read to line 113

		return True
	
	def save(self):
		return True

	def load(self):
		return True

	def query(self):
		return True
