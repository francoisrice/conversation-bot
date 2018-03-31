# Collection of chatbot models, each held within its own class 
# The imports needed for each model are presented before that class

# Vanilla three lawyer artificial neural network

from keras.models import Sequential # basic Keras model
from keras.layers import Dense, Activation # Necessary functions for NN layers and computation

class ANN(object):
	def __init__(self,name,inlayer,hidden,outlayer):
		self.name = name
		self.inlayer = int(inlayer)
		self.hidden =  int(hidden)
		self.outlayer = int(outlayer)


	def create(self,name=self.name,inlayer=self.inlayer,hidden=self.hidden,outlayer=self.outlayer):
		self.name = str(name)
		model = Sequential([

			# creates a hiidden layer with 12 nodes and an input layer
			# with 20 nodes
		   Dense(hidden, input_shape=(inlayer,)), 
		  #Dense(12, input_dim=20), # same as above

		  # Set softmax as the normalizing function for hidden layer outputs
		   Activation('softmax'),

		   Dense(outlayer), # create a 5 node output layer
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

	def load(self,name,inlayer,hidden,outlayer):
		name= str(name)
		model = Sequential([
		   Dense(hidden, input_shape=(inlayer,)),
		  #Dense(12, input_dim=20), # same as above
		   Activation('softmax'),
		   Dense(outlayer),
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
	
	def __init__(self,name,batch_size=64,epochs=100,latent_dim=256,num_samples=10000,max_encoder_seq_length=,max_decoder_seq_length): # All of this inf
		self.name = name
		self.batch_size = batch_size
		self.epochs = epochs
		self.latent_dim = latent_dim
		self.num_samples = num_samples
		
		#These variables may not be necessary
		#self.max_encoder_seq_length =
		#self.max_decoder_seq_length = 
		
	def create(self,sample,target):

		#I have no idea if these line actually define the 
		# creation of the model, or if they should go 
		# into another function, so that is left to the 
		# user :D

		#TDOD - instantiate the variables in the create function for it to work properly

		#Setup variables

		#This will likely have to be the maximum vocabulary of the model
		# since it is the number of tokens. It is assumed that "tokens"
		# for the original script means unique characters, which in this
		# scripts case would be unique words.
		num_encoder_tokens = len(sample)

		#Define the input sequence and process it
		encoder_inputs = Input(shape=(None, num_encoder_tokens))
		encoder = LSTM(self.latent_dim, return_state=True)
		encoder_outputs, state_h, state_c = encoder(encoder_inputs)
		# The LSTM encoder outputs are not needed, but the states are kept
		encoder_states = [state_h, state_c]

		# Pass the encoder states to the decoder 
		decoder_inputs = Input(shape=(None, num_decoder_tokens))
		# The decoder will return output sequences and internal states
		#but not return states. The return states will be used in 
		#inference, not in the training of the model
		decoder = LSTM(latent_dim, return_sequences=True, return_state=True)
		decoder_outputs, _, _ = decoder(decoder_inputs, 
			initial_state=encoder_states)
		decoder_dense = Dense(num_decoder_tokens, activation='softmax')
		decoder_outputs = decoder_dense(decoder_outputs)

		# Define the model
		model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

		return model

	def train(self,sample,target,model):

		#NOTE: Since only one sample is passed in at a time, but there will
		# be several samples used to train, there must be a dictionary of 
		# previously learned tokens passed into the training function. 

		# Vectorize the data
		sample = sample.split(' ')
		target = target.split(' ')
		sample = sorted(list(sample))
		target = sorted(list(target))
		num_encoder_tokens = len(sample)
		num_decoder_tokens = len(target)
		encoder_seq_length = 
		decoder_input_data = 

		input_token_index = dict([(word, i) for i, word in enumerate(sample)])
		target_token_index = dict([(word, i) for i, word in enumerate(target)])

		encoder_input_data = np.zeros((1, encoder_seq_length, num_encoder_tokens),
			dtype='float32')
		decoder_input_data = np.zeros((1, decoder_seq_length, num_decoder_tokens),
			dtype='float32')
		decoder_target_data = np.zeros((1, decoder_seq_length, num_decoder_tokens),
			dtype='float32')

		for t, word in enumerate(sample):
			encoder_input_data[1, t, input_token_index[word]] = 1
		for t, word in enumerate(target):
			decoder_input_data[1, t, target_token_index[word]] = 1
			if t > 0:
				# decoder_target_data is ahead by one time step
				decoder_target_data[1, t-1, target_token_index[word]] = 1

		# Begin the actual training
		model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
		model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
			batch_size= self.batch_size,
			epochs= self.epochs,
			validation_split=0.2)

		return model
	
	def save(self,name=self.name,model):
		model.save(name+'.h5')
		return True

	def load(self):
		return True

	def query(self,sample):

		# I think the following are for query but I'm not sure; querying is 
		#sampling right?
		
		#This is the general idea:
		# 1. encode input and retrieve initial decoder state
		# 2. run one step of decoder with this initial state and a "start
		#of sequence" token as target, and get the next target token
		# 3. Repeat with the next token and states


		# Define sampling models
		encoder_model = Model(encoder_inputs, encoder_states)

		decoder_state_input_h = Input(shape=(latent_dim,))	
		decoder_state_input_c = Input(shape=(latent_dim,))
		decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
		
		# THe decoder LSTM may need to be passed from creation
		decoder_outputs, state_h state_c = decoder(decoder_inputs, 
			initial_state=decoder_states_inputs)
		decoder_states = [state_h, state_c]

		# The decoder_dense layer may need to be passed from creation
		decoder_outputs = decoder_dense(decoder_outputs)
		decoder_model = Model([decoder_inputs] + decoder_states_inputs,
			[decoder_outputs] + decoder_states)

		# Look through the index in reverse to generate new sequence
		reverse_sample_index = dict((i, word) for word, i in sample_token_index.items())
		reverse_target_index = dict((i, word) for word, i in target_token_index.items())

		#TODO - continue implementing Seq2Seq example, start @ line 185

		def decode_sequence(input_seq,encoder_mode=encoder_model,
			decoder_model=decoder_model):

			# Encode the input as a state vector
			states_value = encoder_model.predict(input_seq)

			# Generate empty target sequence of length 1
			target_seq = np.zeros((1,1, num_decoder_tokens))

			#Samples and target data are not seperated by tabs for this script, so the 
			#  following line does not apply. But what is its equivalent? -> Likely 
			#  this will just be ignored and the script will need some custom mods
				# Populate the first character of target sequence with the start character
			target_seq[0,0, target_token_index['\t']] = 1

			
			# Sampling loop for a batch of sequences with a batch size of 1
			stop_condition = False
			decoded_sentence = ''
			while not stop_condition:
				output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

				# Sample the tokens
				sampled_token_index = np.argmax(output_tokens[0, -1, :])
				sampled_word = reverse_target_index[sampled_token_index]
				decoded_sentence += " " + sampled_word + " "

				# Exit Condition:
				#Just like with the "tab" denoting the break between samples and target
				#  data, there will need to be a custom way thought up for exiting. 
				#  Likely after a ., !, or ?; or perhaps an "EOL" word could be used.

				# Update the target sequence
				target_seq = np.zeros((1, 1, num_decoder_tokens))
				target_seq[0, 0, sampled_token_index] = 1

				# Update states
				states_value = [h, c]

			return decoded_sentence

		#TODO - see if this Seq2Seq model is able to generate responses for sequences
		#  that it is not trained on. It should be able to, for sure, but it should 
		#  still be varified. 

		#TODO - move this if to the start of the query function
		#TODO - create a sample_list and a target_list in the train function and 
		#  pass in.
		if sample in sample_list:
			seq_index = sample_list.index(sample)
			input_seq = encoder_input_data[seq_index:seq_index+1]
			decoded_sentence = decode_sequence(input_seq)

		else:
			#Synth response: "I'm sorry, but I do not understand."
			#Perhaps even follow up, by asking what the unfamiliar
			#  word/token means

		return decoded_sentence

	def decode(self):

		def decode_sequence(input_seq):

			#Encode the input as a state vector
			states_value = encoder_model.predict(input_seq)
			pass

		return True