import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def train_valid_test_split(x_data, y_data,
        validation_size=0.1, test_size=0.1, shuffle=True):
    x_, x_test, y_, y_test = train_test_split(x_data, y_data, test_size=test_size, shuffle=shuffle)
    valid_size = validation_size / (1.0 - test_size)
    x_train, x_valid, y_train, y_valid = train_test_split(x_, y_, test_size=valid_size, shuffle=shuffle)
    return x_train, x_valid, x_test, y_train, y_valid, y_test

if __name__ == '__main__':
    path = "glue/"
    pd_all = pd.read_csv(os.path.join(path, "data.csv") )
    pd_all = shuffle(pd_all)
    x_data, y_data = pd_all.text, pd_all.classtype

    x_train, x_valid, x_test, y_train, y_valid, y_test = \
            train_valid_test_split(x_data, y_data, 0.1, 0.1)

    train = pd.DataFrame({'label':y_train, 'x_train': x_train})
    train.to_csv("glue/train.csv", index=False, sep=',')
    valid = pd.DataFrame({'label':y_valid, 'x_valid': x_valid})
    valid.to_csv("glue/dev.csv", index=False, sep=',')
    test = pd.DataFrame({'label':y_test, 'x_test': x_test})
    test.to_csv("glue/test.csv", index=False, sep=',')