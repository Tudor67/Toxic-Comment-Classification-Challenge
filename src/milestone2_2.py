import numpy as np
import pandas as pd

from milestone2 import tokenization, lemmatization
from milestone2 import nb_classifier, write_results
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


DATA_PATH = '../data/'
IN_PATH = '../data/sample_submission.csv'
OUT_PATH = '../results/submission_47.csv'
target_classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


def tf_idf_solution():
    print('---tf idf---')
    print('---read data---')
    train_df = pd.read_csv(DATA_PATH + 'train.csv')
    test_df = pd.read_csv(DATA_PATH + 'test.csv')

    train_text = train_df['comment_text']
    test_text = test_df['comment_text']
    all_text = pd.concat([train_text, test_text])

    word_vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 1),
        stop_words='english',
        max_df=0.95,
        max_features=1500)

    print('---word_vectorizer.fit()---')
    word_vectorizer.fit(all_text)

    print('---word_vectorizer.transform()---')
    train_x = word_vectorizer.transform(train_text)
    test_x = word_vectorizer.transform(test_text)
    train_y = train_df[target_classes]

    print('---nb_classifier---')
    preds = nb_classifier(train_x, train_y, test_x, method='BR')

    print('---write results---')
    write_results(preds, IN_PATH, OUT_PATH)


def main():
    tf_idf_solution()


if __name__ == '__main__':
    main()