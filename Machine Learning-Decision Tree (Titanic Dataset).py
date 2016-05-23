
# coding: utf-8

# In[27]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, preprocessing
from sklearn.preprocessing import Imputer, PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier


# In[28]:

#Preprocessor
le_sex = preprocessing.LabelEncoder()
#Training Data
training=pd.read_csv('/home/prajwal/Desktop/Machine-Learning-Titanic-Dataset/train.csv')
training_label=training[[1]]
training.drop(['Name', 'PassengerId', 'Ticket', 'Cabin','Survived'], axis=1, inplace=True)
training.Sex = le_sex.fit_transform(training.Sex)
training['Embarked'][(training['Embarked'] == 'S')] = 0
training['Embarked'][(training['Embarked'] == 'C')] = 1
training['Embarked'][(training['Embarked'] == 'Q')] = 2
training['Embarked'][(training['Embarked'].isnull())] = 3
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(training)
training = imp.transform(training)
#scale=preprocessing.MinMaxScaler()
#training=scale.fit_transform(training)
#poly = PolynomialFeatures(1)
#training=poly.fit_transform(training)


# In[29]:

#Test Data
test=pd.read_csv('/home/prajwal/Desktop/Machine-Learning-Titanic-Dataset/test.csv')
test_label=pd.read_csv('/home/prajwal/Desktop/Machine-Learning-Titanic-Dataset/genderclassmodel.csv')
test_label=test_label['Survived']
test.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)
#test.Name=le_sex.fit_transform(test.Name)
test.Sex = le_sex.fit_transform(test.Sex)
test['Embarked'][(test['Embarked'] == 'S')] = 0
test['Embarked'][(test['Embarked'] == 'C')] = 1
test['Embarked'][(test['Embarked'] == 'Q')] = 2
test['Embarked'][(test['Embarked'].isnull())] = 3
test = imp.transform(test)
#scale=preprocessing.MinMaxScaler()
#test=scale.fit_transform(test)
#poly = PolynomialFeatures(1)
#test=poly.fit_transform(test)


# In[30]:

regr = DecisionTreeClassifier()
regr.fit(training,training_label)
# The mean square error
print("Residual sum of squares: ",np.mean((regr.predict(test) - test_label) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: ',regr.score(test, test_label))


# In[ ]:



