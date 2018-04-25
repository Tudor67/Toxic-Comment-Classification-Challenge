import sys


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from sklearn import preprocessing


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
    reload(sys)
    sys.setdefaultencoding('utf8')
    lemmatizer = WordNetLemmatizer()
    lemmas = [[lemmatizer.lemmatize(token, pos='v') for token in token_row] for token_row in tokens]
    reload(sys)
    sys.setdefaultencoding('ascii')
    return lemmas


def stopwords_elimination(tokens):
    stopwords_set = set(stopwords.words('english'))
    filtered_tokens = [[token for token in token_row if token not in set(stopwords_set)] for token_row in tokens]   
    return filtered_tokens


def data_preprocessing(df):
    print('-> tokenization')
    tokens = tokenization(df)

    #print('-> stemming')
    #stems = stemming(tokens)

    print('-> lemmatization')
    lemmas = lemmatization(tokens)

    print('-> stopwords elimination')
    filtered_tokens = stopwords_elimination(lemmas)

    return filtered_tokens