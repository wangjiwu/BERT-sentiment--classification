import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# def train_valid_test_split(x_data, y_data,
#         validation_size=0.1, test_size=0.1, shuffle=True):
#     x_, x_test, y_, y_test = train_test_split(x_data, y_data, test_size=test_size, shuffle=shuffle)
#     valid_size = validation_size / (1.0 - test_size)
#     x_train, x_valid, y_train, y_valid = train_test_split(x_, y_, test_size=valid_size, shuffle=shuffle)
#     return x_train, x_valid, x_test, y_train, y_valid, y_test

if __name__ == '__main__':
    path = "glue/"
    pd_all = pd.read_csv(os.path.join(path, "train.tsv"), sep='\t' )
    pd_all = shuffle(pd_all)



    dev_set = pd_all.iloc[0:pd_all.shape[0]/10]
    dev_set.to_csv("glue/dev.tsv", index=False, sep='\t')