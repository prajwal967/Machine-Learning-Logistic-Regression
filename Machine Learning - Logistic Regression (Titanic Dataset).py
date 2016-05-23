
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, preprocessing
from sklearn.preprocessing import Imputer, PolynomialFeatures


# In[3]:

#Preprocessor
le_sex = preprocessing.LabelEncoder()
#Training Data
training=pd.read_csv('/home/prajwal/Desktop/Machine Learning-Logistic Regression (Titanic Dataset)/train.csv')
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
#training=preprocessing.normailze(training, norm='l2')
#poly = PolynomialFeatures(1)
#training=poly.fit_transform(training)


# In[5]:

#Test Data
test=pd.read_csv('/home/prajwal/Desktop/Machine Learning-Logistic Regression (Titanic Dataset)/test.csv')
test_label=pd.read_csv('/home/prajwal/Desktop/Machine Learning-Logistic Regression (Titanic Dataset)/genderclassmodel.csv')
test_label=test_label['Survived']
test.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)
#test.Name=le_sex.fit_transform(test.Name)
test.Sex = le_sex.fit_transform(test.Sex)
test['Embarked'][(test['Embarked'] == 'S')] = 0
test['Embarked'][(test['Embarked'] == 'C')] = 1
test['Embarked'][(test['Embarked'] == 'Q')] = 2
test['Embarked'][(test['Embarked'].isnull())] = 3
test = imp.transform(test)
#test=preprocessing.normalize(test, norm='l2')
#poly = PolynomialFeatures(1)
#test=poly.fit_transform(test)


# In[7]:

regr = linear_model.LogisticRegressionCV(solver='liblinear')
regr.fit(training,training_label)
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean square error
print("Residual sum of squares: ",np.mean((regr.predict(test) - test_label) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: ',regr.score(test, test_label))


# In[8]:

#plt.plot(range(0,418),sorted((regr.predict(test)-test_label),reverse=True))
#plt.plot(range(0,891),sorted((regr.predict(training) - training_label.Survived),reverse=True))
#plt.show()


# In[ ]:



