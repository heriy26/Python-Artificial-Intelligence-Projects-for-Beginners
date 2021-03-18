#%% Decision Tree

import pandas as pd
from sklearn import tree
import graphviz
import numpy as np

heriy = pd.read_csv('dataset/cars.csv', sep=';')

heriy['pass'] = heriy.apply(lambda row: 1 if (row['MPG']+row['Cylinders']+row['Acceleration'])>= 35 else 0, axis=1)

heriy = heriy.drop(['MPG', 'Cylinders', 'Acceleration'], axis=1)

heriy = pd.get_dummies(heriy, columns=['Car', 'Origin'])

heriy = heriy.sample(frac=1)
heriy_train = heriy[:150]
heriy_test = heriy[150:]

heriy_train_att = heriy_train.drop(['pass'], axis=1)
heriy_train_pass = heriy_train['pass']

heriy_test_att = heriy_test.drop(['pass'], axis=1)
heriy_test_pass = heriy_test['pass']

heriy_att = heriy.drop(['pass'], axis=1)
heriy_pass = heriy['pass']

print ("Passing: %a out of %a (%.2f%%)" % (np.sum(heriy_pass), len(heriy_pass), 100*float(np.sum(heriy_pass)) / len(heriy_pass)))

anto = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
anto = anto.fit(heriy_train_att, heriy_train_pass)

dot_data = tree.export_graphviz(anto, out_file=None, label="all", impurity=False, 
                                proportion=True, feature_names=list(heriy_train_att), 
                                class_names=["fail", "pass"], filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph

#%% Predict
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

heriy_new = pd.read_csv('dataset/cars.csv', sep=';')

inputs = heriy_new.drop('Model',axis='columns')

target = heriy_new['Model']

le_Car = LabelEncoder()
le_Origin = LabelEncoder()

inputs['Car_n'] = le_Car.fit_transform(inputs['Car'])
inputs['Origin_n'] = le_Origin.fit_transform(inputs['Origin'])
inputs

inputs_n = inputs.drop(['Car','Origin'],axis='columns')

model = tree.DecisionTreeClassifier()

model.fit(inputs_n, target)
model.score(inputs_n,target)
model.predict([[1,1,1,1,1,1,1,1]])