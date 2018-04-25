import gensim
import numpy as np
import pandas as pd

from constants import *
from sklearn.feature_extraction.text import TfidfVectorizer


def get_pretrained_word_embeddings(word_rows, word_emb_dims=300):
    print('-> pretrained word embeddings')
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

    return word_embeddings


def tf_idf(train_df, test_df):
    print('---tf-idf---')
    train_text = train_df['comment_text']
    test_text = test_df['comment_text']
    all_text = pd.concat([train_text, test_text])

    word_vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        stop_words='english',
        max_df=0.95)

    print('---word_vectorizer.fit()---')
    word_vectorizer.fit(all_text)

    print('---word_vectorizer.transform()---')
    train_x = word_vectorizer.transform(train_text)
    test_x = word_vectorizer.transform(test_text)
    train_y = train_df[target_classes].as_matrix()

    return (train_x, train_y, test_x)
