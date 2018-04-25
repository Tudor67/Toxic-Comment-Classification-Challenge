# Toxic-Comment-Classification-Challenge
SSL Assignment

## Results

| Preprocessing | Feature extraction | Classification | Private/public score |
| :---: | :---: | :---: | :---: |
| tokenization + lemmatization + stopwords elimination | word embeddings | SVM | 0.6761 / 0.6842 |
| tokenization + lemmatization + stopwords elimination | tf-idf (ngram_range=(1, 2)) | ^ | 0.6838 / 0.6898 |
| tokenization + lemmatization + stopwords elimination | word embeddings | Naive Bayes | 0.7927 / 0.7936 |
| tokenization + lemmatization + stopwords elimination | tf-idf (ngram_range=(1, 2)) | ^ | 0.8086 / 0.8108 |
| tokenization + stopwords elimination | word embeddings | Logistic Regression | 0.9503 / 0.9493 |
| tokenization + stopwords elimination | tf-idf (ngram_range=(1, 2)) | ^ | **0.9742 / 0.9743** |
