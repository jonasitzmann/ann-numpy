import numpy as np
import pandas as pd

def normalize(array):
    ar = array[:]
    mean = sum(ar) / len(ar)
    ar = [el - mean for el in ar]
    std_dev = np.sqrt(sum([el**2 for el in ar]) / len(ar))
    ar = [el / std_dev for el in ar]
    return ar
def dec_to_binArr(dec, n_digits):
    arr = [int(bin_digit) for bin_digit in bin(dec)[2:]]
    return [0 for _ in range(n_digits - len(arr))] + arr

def arrToDict(arr, name):
    d = dict(enumerate(arr))
    d = {"{}_{}".format(name, k) : v for k, v in d.items()}
    return d

def encode_categorical(array):
    arr_unique = list(set(array))
    n_cols = int(np.ceil(np.log2(len(arr_unique))))
    d = {name : arrToDict(dec_to_binArr(idx, n_cols), array.name) for idx, name in enumerate(arr_unique)}
    newCols = [array.name + "_" + str(i) for i in range(n_cols)]
    df = pd.DataFrame()
    for _, el in array.items():
        df = df.append(d[el], ignore_index=True)
    return df


def preprocess(df):
    df.dropna(inplace=True, axis=1)
    newDf =  pd.DataFrame()
    for col in df:
        print("prepocessing column: {}".format(col))
        if(np.isreal(df[col][0])):
             newDf[col] = normalize(df[col])
        else: # categorical feature
            newCols = encode_categorical(df[col])
            newDf = pd.concat([newDf, newCols], axis=1)
    return newDf

# prepare features and lables
df = pd.read_csv("titanic/titanic.csv")
y = df["Survived"]
y.to_csv("titanic/labels.csv")
x_raw = df.drop(["Survived", "Name", "Ticket"], 1)
x = preprocess(x_raw)
x.to_csv("titanic/features.csv")
