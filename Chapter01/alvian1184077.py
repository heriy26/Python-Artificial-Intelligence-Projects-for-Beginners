from sklearn import tree
import pandas as pd

def prepoc(datapath):
    d = pd.read_csv(datapath, sep=',')
    len(d)
    
    # shuffle data
    d = d.sample(frac=1)
    d_train = d[:1152] #menampung data train
    d_test = d[800:] #menampung data test

    d_train_att = d_train.drop(['1'], axis=1)
    d_train_pass = d_train['1']
    
    d_test_att = d_test.drop(['1'], axis=1)
    d_test_pass = d_test['1']
    
    d_att = d.drop(['1'], axis=1)
    d_pass = d['1']
    return d_train_att,d_train_pass,d_test_att,d_test_pass,d_att,d_pass

def training(d_train_att,d_train_pass):
    t = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
    t = t.fit(d_train_att, d_train_pass)
    return t

def testing(t,testdataframe):
    return t.predict(testdataframe)