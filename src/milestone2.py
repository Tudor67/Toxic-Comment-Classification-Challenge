import csv
import gensim
import numpy as np
import pandas as pd
import sys


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn import svm


IN_PATH = '../data/sample_submission.csv'
OUT_PATH = '../results/submission_36.csv'
DATA_PATH = '../data/'
WORD2VEC_PATH = '../data/GoogleNews-vectors-negative300.bin'
#MAX_NR_OF_COMMENTS = 10000
word2vec_model = None
word_embeddings = None
target_classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


def load_data(path):
    train_df = pd.read_csv(path + 'train.csv')
    test_df = pd.read_csv(path + 'test.csv')
    #train_df = train_df.head(MAX_NR_OF_COMMENTS)
    #test_df = test_df.head(MAX_NR_OF_COMMENTS)
    return (train_df, test_df)


def load_word2vec_model(path):
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
    return word2vec_model


def tokenization(df):
    tokens = [comment_row.replace('\n', ' ').replace('.', ' ').replace(',', ' ')
                .replace('?', ' ').replace('!', ' ').replace("'", ' ').replace('-', ' ')
                .replace('@',' ').replace('$', ' ').replace('%', ' ').replace('&', ' ')
                .replace('[', ' ').replace('(', ' ').replace(']', ' ').replace(')', ' ')
                .replace('+', ' ').replace('^', ' ').replace('"', ' ').replace(';', ' ')
                .replace(':', ' ').replace('=', ' ').replace('>', ' ').replace('<', ' ')
                .replace('~', ' ').replace('{', ' ').replace('}', ' ').replace('#', ' ')
                .lower().split() for comment_row in df['comment_text']]
    return tokens


def stemming(tokens):
    stemmer = SnowballStemmer('english')
    stems = [[stemmer.stem(token) for token in token_row] for token_row in tokens]
    return stems


def lemmatization(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmas = [[lemmatizer.lemmatize(token, pos='v') for token in token_row] for token_row in tokens]
    return lemmas


def stopwords_elimination(tokens):
    stopwords_set = set(stopwords.words('english'))
    filtered_tokens = [[token for token in token_row if token not in set(stopwords_set)] for token_row in tokens]   
    return filtered_tokens


def get_word_embeddings(word_rows, word_emb_dims=300):
    global word2vec_model
    global word_embeddings
    if word2vec_model is None:
        print('load word2vec_model')
        word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(WORD2VEC_PATH, binary=True)

    word_embeddings = np.zeros((len(word_rows), word_emb_dims))
    word_emb_nr = [0 for _ in range(len(word_rows))]

    for i in range(len(word_rows)):
        for word in word_rows[i]:
            if word2vec_model.vocab.has_key(word):
                word_embeddings[i] += word2vec_model.word_vec(word)
                word_emb_nr[i] += 1

    # a comment is represented by the average of word_embbedings
    for i in range(len(word_embeddings)):
        if word_emb_nr[i] != 0:
            word_embeddings[i] = word_embeddings[i] / word_emb_nr[i]


def data_preprocessing(df):
    print('-> tokenization')
    tokens = tokenization(df)

    #print('-> stemming')
    #stems = stemming(tokens)

    print('-> lemmatization')
    lemmas = lemmatization(tokens)

    print('-> stopwords elimination')
    filtered_tokens = stopwords_elimination(lemmas)

    print('-> word embeddings')
    get_word_embeddings(filtered_tokens)

    x = word_embeddings
    y = None
    if df.get('toxic') is not None:
        y = df[target_classes].as_matrix()

    return (x, y)


def nb_classifier(train_x, train_y, test_x, method='BR'):
    scaler = preprocessing.StandardScaler().fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)

    classifier = None
    if method == 'BR':
        classifier = BinaryRelevance(GaussianNB())
    elif method == 'CC':
        classifier = ClassifierChain(GaussianNB())
    elif method == 'LP':
        classifier = LabelPowerset(GaussianNB())

    classifier.fit(train_x, train_y)
    
    pred = classifier.predict(test_x)
    
    return pred


def svm_classifier(train_x, train_y, test_x, method='LinearSVC'):
    scaler = preprocessing.StandardScaler().fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)

    pred = np.zeros((test_x.shape[0], len(target_classes)))

    for i in range(len(target_classes)):
        print('class {}'.format(i))
        classifier = None
        if method == 'LinearSVC':
            classifier = svm.LinearSVC()
        elif method == 'SVC':
            classifier = svm.SVC(C=1.3)

        classifier.fit(train_x, train_y[:,i])
        pred[:,i] = classifier.predict(test_x)

    return pred


def write_results(pred, in_path, out_path):
    res_df = pd.read_csv(in_path)

    reload(sys)
    sys.setdefaultencoding('ascii')

    idx = 0
    for x in pred:
        # TODO: x.toarray()[0] for nb_classifier output
        probs = x #x.toarray()[0]
        
        for k in range(len(target_classes)):
            res_df[target_classes[k]].set_value(idx, probs[k])
        
        idx += 1
    
    res_df.to_csv(out_path, index=False)


def main():
    reload(sys)
    sys.setdefaultencoding('utf8')

    print('\n---load data---')
    (train_df, test_df) = load_data(DATA_PATH)

    print('---data preprocessing (train)---')
    (train_x, train_y) = data_preprocessing(train_df)

    print('---data preprocessing (test)---')
    (test_x, _) = data_preprocessing(test_df)
    
    print('---classify---')
    preds = svm_classifier(train_x, train_y, test_x, method='SVC')

    print('---write results---')
    write_results(preds, IN_PATH, OUT_PATH)


if __name__ == '__main__':
    main()
