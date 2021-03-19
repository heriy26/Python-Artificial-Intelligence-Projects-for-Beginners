import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

def read(datapath):

    A = pd.read_csv(datapath, sep=';')
    
    A['pass'] = A.apply(lambda row: 1 if (row['Acceleration'])>= 15 else 0, axis=1)
    A = A.drop(['Acceleration'], axis=1)
        
    le_Car = LabelEncoder()
    le_Origin = LabelEncoder()
    
    A['Car_n'] = le_Car.fit_transform(A['Car'])
    A['Origin_n'] = le_Origin.fit_transform(A['Origin'])
    
    B = A.drop(['Car','Origin'], axis ='columns')
    
    #shuffle data
    C = B.sample(frac=1) 
    D_train = C[:350]
    D_test = C[350:] 
    
    D_train_x = D_train.drop(['pass'], axis=1) 
    D_train_y = D_train['pass'] 
    
    D_test_x = D_test.drop(['pass'], axis=1) 
    D_test_y = D_test['pass']
    
    D_x = C.drop(['pass'], axis=1)
    D_y = C['pass']
    
    return D_train_x, D_train_y, D_test_x, D_test_y, D_x, D_y

def dtrain(D_train_x, D_train_y):
    
    Dt_train = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
    Dt_train = Dt_train.fit(D_train_x, D_train_y)
    return Dt_train

def dpred(Dt_train, dt_test):
    return Dt_train.predict(dt_test)