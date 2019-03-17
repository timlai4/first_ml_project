# Copyright 2017 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 22:41:51 2019

@author: Tim
I have modified this program which was originally created by Google.
"""

from __future__ import print_function

import math

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

suicide_dataframe = pd.read_csv("master.csv", sep=",")
Europe = ["Albania","Russian Federation","France","Ukraine","Germany","Poland","United Kingdom",
         "Italy","Spain","Hungary","Romania","Belgium","Belarus","Netherlands","Austria",
         "Czech Republic","Sweden","Bulgaria","Finland","Lithuania","Switzerland","Serbia",
         "Portugal","Croatia","Norway","Denmark","Slovakia","Latvia","Greece","Slovenia",
         "Turkey","Estonia","Georgia","Albania","Luxembourg","Armenia","Iceland","Montenegro",
         "Cyprus","Bosnia and Herzegovina","San Marino","Malta","Ireland"]
NorthAmerica = ["United States","Mexico","Canada","Cuba","El Salvador","Puerto Rico",
                "Guatemala","Costa Rica","Nicaragua","Belize","Jamaica"]
SouthAmerica = ["Brazil","Colombia", "Chile","Ecuador","Uruguay","Paraguay","Argentina",
                "Panama","Guyana","Suriname"]
MiddleEast = ["Kazakhstan","Uzbekistan","Kyrgyzstan","Israel","Turkmenistan","Azerbaijan",
              "Kuwait","United Arab Emirates","Qatar","Bahrain","Oman"]
Asia = ["Japan","Republic of Korea", "Thailand", "Sri Lanka","Philippines","New Zealand",
        "Australia","Singapore","Macau","Mongolia"]
for i in range(0,len(suicide_dataframe)):
  if suicide_dataframe.iloc[i,0] in Europe:
    suicide_dataframe.iloc[i,7] = "Europe"
  elif suicide_dataframe.iloc[i,0] in NorthAmerica:
    suicide_dataframe.iloc[i,7] = "North America"
  elif suicide_dataframe.iloc[i,0] in SouthAmerica:
    suicide_dataframe.iloc[i,7] = "South America"
  elif suicide_dataframe.iloc[i,0] in MiddleEast:
    suicide_dataframe.iloc[i,7] = "Middle East"
  elif suicide_dataframe.iloc[i,0] in Asia:
    suicide_dataframe.iloc[i,7] = "Asia"
  else:
    suicide_dataframe.iloc[i,7] = "Island Nation"

continent = pd.get_dummies(suicide_dataframe['country-year'])
gender = pd.get_dummies(suicide_dataframe['sex'])
age = pd.get_dummies(suicide_dataframe['age'])
suicide_dataframe = suicide_dataframe.drop('sex',axis=1)
suicide_dataframe = suicide_dataframe.drop('age',axis=1)
suicide_dataframe = suicide_dataframe.drop('country',axis=1)
suicide_dataframe = suicide_dataframe.drop('country-year',axis=1)
suicide_dataframe = suicide_dataframe.join(continent)
suicide_dataframe = suicide_dataframe.join(gender)
suicide_dataframe = suicide_dataframe.join(age)


suicide_dataframe = suicide_dataframe.reindex(np.random.permutation(suicide_dataframe.index))
new_names = ['year', 'suicides_no', 'population', 'suicides/100k pop',
             'HDI for year', 'annual_gdp', 'gdp_per_capita', 'generation', 
             'Asia', 'EU','Other','ME','NA','SA', 'female', 'male', 
             'age1', 'age2', 'age3', 'age4', 'age5', 'age6']
suicide_dataframe.columns = new_names


def log_normalize(series):
  return series.apply(lambda x:math.log(x+1.0)-5)
def linear_scale(series):
  min_val = series.min()
  max_val = series.max()
  scale = (max_val - min_val) / 2.0
  return series.apply(lambda x:((x - min_val) / scale) - 1.0)

def preprocess_features(suicide_dataframe):
  """Prepares input features from data set.

  Returns:
    A DataFrame that contains the features to be used for the model
  """
  selected_features = suicide_dataframe[
    ["Asia",
     "EU",
     "Other",
     "ME",
     "NA",
     "SA",
     "female",
     "male",
     "age1",
     "age2",
     "age3",
     "age4",
     "age5",
     "age6"]]
  selected_features["gdp_per_capita"] = log_normalize(suicide_dataframe["gdp_per_capita"])
  selected_features["year"] = linear_scale(suicide_dataframe["year"])
  processed_features = selected_features.copy()
  return processed_features

def preprocess_targets(suicide_dataframe):
  """Prepares target features (i.e., labels) from data set.

  Returns:
    A DataFrame that contains the target feature.
  """
  output_targets = pd.DataFrame()
  output_targets["suicides/100k"] = suicide_dataframe["suicides/100k pop"] 
  return output_targets
training_examples = preprocess_features(suicide_dataframe.head(20000))
training_examples.describe()

training_targets = preprocess_targets(suicide_dataframe.head(20000))
training_targets.describe()

validation_examples = preprocess_features(suicide_dataframe.tail(7000))
validation_examples.describe()

validation_targets = preprocess_targets(suicide_dataframe.tail(7000))
validation_targets.describe()

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of multiple features.
  
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
    
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                           
 
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(20000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def construct_feature_columns(input_features):
  """Construct the TensorFlow Feature Columns.

  Args:
    input_features: The names of the numerical input features to use.
  Returns:
    A set of feature columns
  """ 
  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])
def train_model(
    learning_rate,
    steps,
    batch_size,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
  """Trains a linear regression model of multiple features.
  
  In addition to training, this function also prints training progress information,
  as well as a plot of the training and validation loss over time.
  
  Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    training_examples: A `DataFrame` containing one or more columns from
      `suicide_dataframe` to use as input features for training.
    training_targets: A `DataFrame` containing exactly one column from
      `suicide_dataframe` to use as target for training.
    validation_examples: A `DataFrame` containing one or more columns from
      `suicide_dataframe` to use as input features for validation.
    validation_targets: A `DataFrame` containing exactly one column from
      `suicide_dataframe` to use as target for validation.
      
  Returns:
    A `LinearRegressor` object trained on the training data.
  """

  periods = 10
  steps_per_period = steps / periods
  
  # Create a linear regressor object.
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  linear_regressor = tf.estimator.LinearRegressor(
      feature_columns=construct_feature_columns(training_examples),
      optimizer=my_optimizer
  )
  
  # Create input functions.
  training_input_fn = lambda: my_input_fn(
      training_examples, 
      training_targets["suicides/100k"], 
      batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(
      training_examples, 
      training_targets["suicides/100k"], 
      num_epochs=1, 
      shuffle=False)
  predict_validation_input_fn = lambda: my_input_fn(
      validation_examples, validation_targets["suicides/100k"], 
      num_epochs=1, 
      shuffle=False)

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training model...")
  print("RMSE (on training data):")
  training_rmse = []
  validation_rmse = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    linear_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period,
    )
    # Take a break and compute predictions.
    training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
    training_predictions = np.array([item['predictions'][0] for item in training_predictions])
    
    validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
    validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
    
    
    # Compute training and validation loss.
    training_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(training_predictions, training_targets))
    validation_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(validation_predictions, validation_targets))
    # Occasionally print the current loss.
    print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
    # Add the loss metrics from this period to our list.
    training_rmse.append(training_root_mean_squared_error)
    validation_rmse.append(validation_root_mean_squared_error)
  print("Model training finished.")

  # Output a graph of loss metrics over periods.
  plt.ylabel("RMSE")
  plt.xlabel("Periods")
  plt.title("Root Mean Squared Error vs. Periods")
  plt.tight_layout()
  plt.plot(training_rmse, label="training")
  plt.plot(validation_rmse, label="validation")
  plt.legend()

  return linear_regressor

linear_regressor = train_model(
    learning_rate=0.01,
    steps=2000,
    batch_size=50,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)