import numpy as np
import pandas as pd


from constants import *
from input_output import *
from preprocessing import *
from feature_extraction import *
from classification import *


def main():
    (train_df, test_df) = load_data(DATA_PATH)

    (train_tokens_x) = data_preprocessing(train_df)
    (test_tokens_x) = data_preprocessing(test_df)

    train_x = get_pretrained_word_embeddings(train_tokens_x)
    train_y = train_df[target_classes].as_matrix()
    test_x = get_pretrained_word_embeddings(test_tokens_x)

    preds = logistic_regression_with_word_embeddings(train_x, train_y, test_x)

    write_results(preds, IN_PATH, OUT_PATH)


if __name__ == '__main__':
    main()