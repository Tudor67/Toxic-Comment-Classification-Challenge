import pandas as pd


from constants import *


def load_data(path):
    print('---load_data---')
    train_df = pd.read_csv(path + 'train.csv')
    test_df = pd.read_csv(path + 'test.csv')

    #train_df = train_df.head(MAX_NR_OF_COMMENTS)
    #test_df = test_df.head(MAX_NR_OF_COMMENTS)

    return (train_df, test_df)


def write_results(pred, in_path, out_path):
    print('---write_results---')
    res_df = pd.read_csv(in_path)

    idx = 0
    for x in pred:
        # TODO: x.toarray()[0] for nb_classifier output
        #       x for svm_classifier output
        probs = x #x.toarray()[0]
        
        for k in range(len(target_classes)):
            res_df[target_classes[k]].set_value(idx, probs[k])
        
        idx += 1
    
    res_df.to_csv(out_path, index=False)