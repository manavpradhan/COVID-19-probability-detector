import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

def split(data, ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_size = int(len(data)*ratio)
    test_indices = shuffled[:test_size]
    train_indices = shuffled[test_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

if __name__=="__main__":
    df = pd.read_csv("covid.csv")
    # print(df.head())
    # print(df.info())
    # print(df["breath_diff"].value_counts())
    # print(df.describe())

    train, test = split(df, 0.2)
    print(train)
    print(test)
    x_train = train[['fever','bodyPain','age','runnyNose','breath_diff']].to_numpy()
    #print(x_train)
    x_test = test[['fever','bodyPain','age','runnyNose','breath_diff']].to_numpy()
    #print(x_test )
    y_train = train[['infection_prob']].to_numpy().reshape(1907,)
    y_test = test[['infection_prob']].to_numpy().reshape(476,)
    #print(y_train)

    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    file = open('model.pkl', 'wb')
    pickle.dump(clf, file)
    file.close()

