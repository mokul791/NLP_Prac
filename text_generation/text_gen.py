import spacy
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import model
from pickle import dump,load
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import random

# load spacy for preprocessing
nlp = spacy.load('en_core_web_md', disable=['parser', 'tagger', 'ner'])
nlp.max_length = 1200000


# read file as text string
def read_file(filepath):
	with open(filepath) as f:
		text_str = f.read()
	return text_str


# separate punctuation and get the tokens
def separate_punc(text_doc):
	return [token.text.lower() for token in nlp(text_doc) if
				token.text not in '\n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n']


# Create sequence of text
def sequence_text(tokens, train_length):
	text_sequence = []
	for i in range(train_length, len(tokens)):
		seq = tokens[i-train_length:i]
		text_sequence.append(seq)
	return text_sequence


# reading a document and tokenize by separating punctuation
d = read_file('../textfiles/moby_dick_four_chapters.txt')
tokens = separate_punc(d)
print('Number of token is {}'. format(len(tokens)))
print('\n')

text_sequence = sequence_text(tokens, 26)

# tokenizing a text sequence with keras tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_sequence)

# text sequence to tokenize sequence
sequences = tokenizer.texts_to_sequences(text_sequence)
vocabulary_size = len(tokenizer.index_word)

print('Vocabulary Size: {}'.format(vocabulary_size))
print('\n')
print('Text sequence index for first sequence: {}'.format(sequences[0]))
print('\n')

# converting tokenize sequences to numpy array for deep learning
def sequence_array(sequence):
	return np.array(sequence)


sequences_arr = sequence_array(sequences)


# splitting training data into data and labels

X = sequences_arr[:, :-1]
y = sequences_arr[:, -1]
y = to_categorical(y, num_classes=vocabulary_size + 1)

print(X.shape)
seq_length = X.shape[1]

## define the model
# model = model.LSTM_model(vocabulary_size+1, seq_length)

# training the model
# model.fit(X, y, batch_size=128, epochs=300, verbose=1)

# save model and tokenizer
# model.save('epoch300.h5')
# dump(tokenizer, open('epoch300', 'wb'))


# function for generating text
def generate_text(model, tokenizer, seq_length,seed_text, num_gen_words):
	output_text = []
	input_text = seed_text

	for i in range(num_gen_words):

		# take the input text string and encode it to a sequence
		encoded_text = tokenizer.texts_to_sequences([input_text])[0]

		# pad sequence to our seq_length
		pad_encoded = pad_sequences([encoded_text], maxlen=seq_length, truncating='pre')

		# predict class probabilities for each word
		pred_word_index = model.predict_classes(pad_encoded, verbose=0)[0]

		# get the word
		pred_word = tokenizer.index_word[pred_word_index]

		# update the sequence of input text (shifting one over with the new word)
		input_text += ' ' + pred_word

		output_text.append(pred_word)

	return output_text


print(' '.join(text_sequence[0]))

# making some random text for testing the model
random.seed(101)
random_pick = random.randint(0, len(text_sequence))
random_seed_text = text_sequence[random_pick]
seed_text = ' '.join(random_seed_text)

# loading the trained and saved model
model_trained = load_model('epoch300.h5')
tokenizer_saved = load(open('epoch300', 'rb'))

# now calling the generate function
text_generated = generate_text(model_trained, tokenizer_saved, seq_length, seed_text=seed_text, num_gen_words=25)
print('Generated text: {}'.format(' '.join(text_generated)))