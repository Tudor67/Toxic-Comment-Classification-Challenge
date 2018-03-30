import csv
import gensim
import numpy as np
import pandas as pd
import sys

from nltk.stem.snowball import SnowballStemmer
from skmultilearn.problem_transform import LabelPowerset
from sklearn.naive_bayes import GaussianNB

IN_PATH = '../data/sample_submission_csv.csv'
OUT_PATH = '../data/submission_5.csv'
DATA_PATH = '../data/'
WORD2VEC_PATH = './GoogleNews-vectors-negative300.bin'
MAX_NR_OF_COMMENTS = 25000
MAX_COMMENT_LENGTH_IN_WORDS = 20
word2vec_model = None
word_embeddings = None
target_classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


def load_data(path):
	train_df = pd.read_csv(path + 'train.csv')
	test_df = pd.read_csv(path + 'test.csv')
	train_df = train_df.head(MAX_NR_OF_COMMENTS)
	test_df = test_df.head(MAX_NR_OF_COMMENTS)
	return (train_df, test_df)


def load_word2vec_model(path):
	word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
	return word2vec_model


def tokenization(df):
	tokens = [comment_row.replace('\n', ' ').split() for comment_row in df['comment_text']]
	return tokens


def stemming(tokens):
	stemmer = SnowballStemmer('english')
	stems = [[stemmer.stem(token) for token in token_row] for token_row in tokens]
	return stems


def get_word_embeddings(word_rows):
	global word2vec_model
	global word_embeddings
	if word2vec_model is None:
		print('load word2vec_model')
		word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(WORD2VEC_PATH, binary=True)

	word_embeddings = []

	for i in range(len(word_rows)):
		word_embeddings.append([])
		for word in word_rows[i]:
			if len(word_embeddings[i]) < MAX_COMMENT_LENGTH_IN_WORDS and word2vec_model.vocab.has_key(word):
				word_embeddings[i].append(word2vec_model.word_vec(word))


def data_preprocessing(df):
	print('-> tokenization')
	tokens = tokenization(df)
	print('-> stemming')
	stems = stemming(tokens)
	print('-> word embeddings')
	get_word_embeddings(stems)

	for i in range(len(word_embeddings)):
		for j in range(MAX_COMMENT_LENGTH_IN_WORDS - len(word_embeddings[i])):
			word_embeddings[i].append(np.zeros(300))

	x = []
	for i in range(len(word_embeddings)):
		x.append([])
		for j in range(len(word_embeddings[i])):
			x[i] = x[i] + word_embeddings[i][j].tolist()

	y = None
	if df.get('toxic') is not None:
		# TODO: atentie la tipul de date
		y = df[target_classes]

	return (x, y)


def classify(train_x, train_y, classifier_name='nb'):
	classifier = LabelPowerset(GaussianNB())
	if classifier_name == 'nb':
		classifier = LabelPowerset(GaussianNB())

	classifier.fit(train_x, train_y)

	global word2vec_model
	global word_embeddings
	test_df = pd.read_csv(DATA_PATH + 'test.csv')
	preds = []
	batch_nr = len(test_df)/MAX_COMMENT_LENGTH_IN_WORDS
	for batch_idx in range(batch_nr + 1):
		if batch_idx%200 == 0:
			print(batch_idx, batch_nr, sum([elem.shape[0] for elem in preds]))

		#print('-> tokenization')
		tokens = tokenization(test_df[batch_idx*MAX_COMMENT_LENGTH_IN_WORDS:
									min(len(test_df), (batch_idx + 1)*MAX_COMMENT_LENGTH_IN_WORDS)])
		#print('-> stemming')
		stems = stemming(tokens)
		#print('-> word embeddings')
		get_word_embeddings(stems)

		for i in range(len(word_embeddings)):
			for j in range(MAX_COMMENT_LENGTH_IN_WORDS - len(word_embeddings[i])):
				word_embeddings[i].append(np.zeros(300))

		test_x = []
		for i in range(len(word_embeddings)):
			test_x.append([])
			for j in range(len(word_embeddings[i])):
				test_x[i] = test_x[i] + word_embeddings[i][j].tolist()

		preds.append(classifier.predict(test_x))

	return preds


def write_results(preds, in_path, out_path):
	test_df = pd.read_csv('../data/test.csv')
	#res_df = pd.read_csv(in_path)

	print(preds)
	print(dir(preds))
	print(len(preds))
	print(sum([elem.shape[0] for elem in preds]))

	reload(sys)
	sys.setdefaultencoding('ascii')

	with open(out_path, 'w') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(['id'] + target_classes)
		idx = 0
		for pred in preds:
			for x in pred:
				probs = x.toarray()[0]
				#res_df['id'].set_value(idx, test_df['id'][idx].replace(' ', '0').rjust(16, '0'))
				current_id = str(test_df['id'][idx]).replace(' ', '0').rjust(16, '0')
				writer.writerow([current_id] + probs.tolist())
				'''
				for k in range(len(target_classes)):
					res_df[target_classes[k]].set_value(idx, probs[k])
				'''
				idx += 1
				
	#res_df.to_csv(out_path)


def main():
	reload(sys)
	sys.setdefaultencoding('utf8')

	print('\n---load data---')
	(train_df, test_df) = load_data(DATA_PATH)

	print('---train data preprocessing---')
	(train_x, train_y) = data_preprocessing(train_df)
	#print('---test data preprocessing---')
	#(test_x, _) = data_preprocessing(test_df)

	preds = classify(train_x, train_y)
	print('------results---------')
	print('\n-------pred--------')
	#print(preds)

	write_results(preds, IN_PATH, OUT_PATH)
	

if __name__ == '__main__':
	main()