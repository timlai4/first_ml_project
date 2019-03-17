# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 16:57:22 2019

@author: Tim
"""

#%matplotlib inline 

#from IPython.display import Image
#import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import svm

from sklearn import metrics
from sklearn import preprocessing

plt.style.use('fivethirtyeight') # Good looking plots
pd.set_option('display.max_columns', None) # Display any number of columns

suicide_dataframe = pd.read_csv("master.csv", sep=",")

Y = suicide_dataframe['suicides/100k pop'].as_matrix().astype(np.int)

label_encoder = preprocessing.LabelEncoder()       
suicide_dataframe['country'] = label_encoder.fit_transform(suicide_dataframe['country'])
suicide_dataframe['age'] = label_encoder.fit_transform(suicide_dataframe['age'])
suicide_dataframe['year'] = label_encoder.fit_transform(suicide_dataframe['year'])
suicide_dataframe['gender'] = suicide_dataframe['sex'] == 'Male'
suicide_dataframe.drop(['country-year','HDI for year','suicides_no',
                        'population',' gdp_for_year ($) ', 'generation', 
                        'sex','suicides/100k pop'],axis=1,inplace=True)

print(suicide_dataframe.columns)
print(suicide_dataframe.head())


X_train, X_test, y_train, y_test = train_test_split(suicide_dataframe,
                                                    Y,
                                                    test_size=0.2,
                                                    random_state=0)

classifier = svm.SVC(kernel = 'poly')
classifier.fit(X_train,y_train)
predictions = classifier.predict(X_test)

mse = metrics.mean_squared_error(y_test, predictions)
print('Mean Square Error: {:.3f}'.format(mse))

plt.figure(figsize=(16, 12))

plt.scatter(range(predictions.shape[0]), predictions, label='predictions', c='#348ABD', alpha=0.4)
plt.scatter(range(y_test.shape[0]), y_test, label='actual values', c='#A60628', alpha=0.4)
plt.ylim([y_test.min(), predictions.max()])
plt.xlim([0, predictions.shape[0]])
plt.legend();

