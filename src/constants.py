IN_PATH = '../data/sample_submission.csv'
OUT_PATH = '../results/submission_68.csv'
DATA_PATH = '../data/'
WORD2VEC_PATH = '../data/GoogleNews-vectors-negative300.bin'
#MAX_NR_OF_COMMENTS = 10000
word2vec_model = None
word_embeddings = None
target_classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
