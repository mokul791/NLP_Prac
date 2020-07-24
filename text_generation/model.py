from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding

# create LSTM model
def LSTM_model(vocabulary_size, seq_len):

	model = Sequential()

	model.add(Embedding(vocabulary_size, seq_len, input_length=seq_len))
	model.add(LSTM(seq_len*3, return_sequences=True))
	model.add(LSTM(seq_len*3))
	model.add(Dense(seq_len*3, activation='relu'))

	# output layer
	model.add(Dense(vocabulary_size, activation='softmax'))

	# model compile
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	# model summary
	model.summary()
	return model

LSTM_model(120000, 10)