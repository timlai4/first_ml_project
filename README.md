# Suicide Predictor

## Goal:  

Given basic geographical and demographical features, the algorithm attempts to predict the number of suicides.

### Dataset: 
Data set is in [master.csv](https://github.com/timlai4/suicide_project/blob/master/master.csv) and was obtained from [Kaggle]( https://www.kaggle.com/russellyates88/suicide-rates-overview-1985-to-2016). We cleaned the dataset removing some features that we felt were not as relevant or sparse and encoded categorical features with one-hot encoding. Numerical features were normalized to facilitate stochastic gradient descent. 

[suicide_nn.py](https://github.com/timlai4/suicide_project/blob/master/suicide_nn.py) is a fully connected feed forward DNN with TensorFlow, results
![](results.jpg?raw=true)

[suicide_project.py](https://github.com/timlai4/suicide_project/blob/master/suicide_project.py) is a linear regressor with TensorFlow, results 
![](results2.jpg?raw=true)

[suicide_project2.py](https://github.com/timlai4/suicide_project/blob/master/suicide_project2.py) is SVM with SKLearn

