# Toxic-Comment-Classification-Challenge
SSL Assignment

## Description
All details about the [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) can be found on the Kaggle platform.

## Methods
For the first two milestones, the proposed solutions have the following structure:
1. **Preprocessing**
    * _**Tokenization**_. At this step, I split comments in words, ignoring punctuation marks, spaces, newlines and special characters.
    * _**Stemming (SnowballStemmer) / Lemmatization (WordNetLemmatizer)**_[1]. I transform words from previous step to their word stems/lemmas.
      In my experiments, lemmatization always gives better results than stemming.
    * _**Stopwords elimination**_[2].
2. **Feature extraction**
    * _**Word embeddings**_. For this step, I used pretrained vectors for word stems/lemmas (from step 1), which were trained on part of Google News Dataset [3]. The model contains 300 - dimensional vectors for 3 million words and word phrases [4].
      Each comment is represented as the average of the word embeddings.
    * _**TF-IDF**_. Comments are represented using tf-idf features [5]. This representation gives encouraging results even if it's simple.
3. **Classification**. I decompose the problem in 6 binary classification problems (binary relevance approach). For classification I try the following classifiers:
    * SVM;
    * Naive Bayes;
    * Logistic Regression [6].

## Results

| Preprocessing | Feature extraction | Classification | Private/public score |
| :---: | :---: | :---: | :---: |
| tokenization + lemmatization + stopwords elimination | word embeddings | SVM | 0.6761 / 0.6842 |
| tokenization + lemmatization + stopwords elimination | tf-idf (ngram_range=(1, 2)) | SVM | 0.6838 / 0.6898 |
| tokenization + lemmatization + stopwords elimination | word embeddings | Naive Bayes | 0.7927 / 0.7936 |
| tokenization + lemmatization + stopwords elimination | tf-idf (ngram_range=(1, 2)) | Naive Bayes | 0.8086 / 0.8108 |
| tokenization + stopwords elimination | word embeddings | Logistic Regression | 0.9503 / 0.9493 |
| tokenization + stopwords elimination | tf-idf (ngram_range=(1, 2)) | Logistic Regression | **0.9742 / 0.9743** |

## References 
[1]: http://textminingonline.com/dive-into-nltk-part-iv-stemming-and-lemmatization
[2]: https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
[3]: http://ahogrammer.com/2017/01/20/the-list-of-pretrained-word-embeddings/
[4]: https://code.google.com/archive/p/word2vec/
[5]: https://stackoverflow.com/questions/47557417/understanding-text-feature-
extraction-tfidfvectorizer-in-python-scikit-learn
[6]: https://www.kaggle.com/tunguz/logistic-regression-with-words-and-char-n-grams
