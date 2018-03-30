import milestone2 as m2


def get_set(initial_tokens):
    set_initial_tokens = set()
    for token_row in initial_tokens:
        for token in token_row:
            set_initial_tokens.add(token)

    return set_initial_tokens


def get_lemmas(tokens):
    lemmas = set()
    lemmatizer = m2.WordNetLemmatizer()
    for token in tokens:
        lemmas.add(lemmatizer.lemmatize(token, pos='v'))

    return lemmas


def stopwords_elimination(tokens):
    stopwords_set = set(m2.stopwords.words('english'))
    filtered_tokens = set()

    for token in tokens:
        if token not in stopwords_set:
            filtered_tokens.add(token)

    return filtered_tokens


def nr_of_words():
    (train_df, test_df) = m2.load_data(m2.DATA_PATH)
    print('-----Initial dataframe-----')
    print('train: {}'.format(train_df.shape))
    print('test:  {}'.format(test_df.shape))

    train_initial_tokens = m2.tokenization(train_df)
    test_initial_tokens = m2.tokenization(test_df)
    train_set_initial_tokens = get_set(train_initial_tokens)
    test_set_initial_tokens = get_set(test_initial_tokens)
    train_test_set_initial_tokens = train_set_initial_tokens.union(test_set_initial_tokens)
    print('-----Initial tokens-----')
    print('train: {}'.format(len(train_set_initial_tokens)))
    print('test:  {}'.format(len(test_set_initial_tokens)))
    print('train&test: {}'.format(len(train_test_set_initial_tokens)))

    train_set_lemmas = get_lemmas(train_set_initial_tokens)
    test_set_lemmas = get_lemmas(test_set_initial_tokens)
    train_test_set_lemmas = train_set_lemmas.union(test_set_lemmas)
    print('-----Lemmas-----')
    print('train: {}'.format(len(train_set_lemmas)))
    print('test:  {}'.format(len(test_set_lemmas)))
    print('train&test: {}'.format(len(train_test_set_lemmas)))


    train_set = stopwords_elimination(train_set_lemmas)
    test_set = stopwords_elimination(test_set_lemmas)
    train_test_set = train_set.union(test_set)
    print('-----Stopwords elimination-----')
    print('train: {}'.format(len(train_set)))
    print('test:  {}'.format(len(test_set)))
    print('train&test: {}'.format(len(train_test_set)))


def main():
    reload(m2.sys)
    m2.sys.setdefaultencoding('utf-8')
    nr_of_words()


if __name__ == '__main__':
    main()