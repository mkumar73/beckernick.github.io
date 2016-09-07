---
title:  "Predicting Human Activity from Smartphone Accelerometer Data"
date:   2016-09-07
tags: [data science]

header:
  image: "last_fm_spotify_edited.png"
  caption: "Photo credit: [**Slash Gear**](http://www.slashgear.com/last-fm-and-spotify-team-up-to-give-better-music-recommendations-29315027/)"

excerpt: "Code Focused. Human Activity Classification, Accelerometers, Support Vector Machines"
---


I've been using a lot of products with recommendation engines lately, so I decided it would be cool to build one myself. Recommender systems can be loosely broken down into three categories: **content based systems**, **collaborative filtering systems**, and **hybrid systems** (which use a combination of the other two).

Content based recommender systems use the features of items to recommend other similar items. For example, if I'm browsing for solid colored t-shirts on Amazon, a content based recommender might recommend me other t-shirts or solid colored sweatshirts because they have similar features (sleeves, single color, shirt, etc.).

Collaborative filtering based systems use the actions of users to recommend other items. In general, they can either be user based or item based. User based collaborating filtering uses the patterns of users similar to me to recommend a product (users like me also looked at _these other_ items). Item based collaborative filtering uses the patterns of users who browsed the same item as me to recommend me a product (users who looked at my item also looked at _these other_ items).



```python
import pandas as pd
from scipy.spatial.distance import cosine
import numpy as np
from scipy.sparse import csr_matrix
from __future__ import division
import matplotlib.pyplot as plt
%matplotlib inline

# display results to 3 decimal points, not in scientific notation
pd.set_option('display.float_format', lambda x: '%.3f' % x)
```


```python
with open('/Users/nickbecker/Downloads/HAPT Data Set/features.txt') as handle:
    features = handle.readlines()
    features = map(lambda x: x.strip(), features)
```


```python
with open('/Users/nickbecker/Downloads/HAPT Data Set/activity_labels.txt') as handle:
    activity_labels = handle.readlines()
    activity_labels = map(lambda x: x.strip(), activity_labels)
```


```python
activity_df = pd.DataFrame(activity_labels)
activity_df = pd.DataFrame(activity_df[0].str.split(' ').tolist(),
                           columns = ['activity_id', 'activity_label'])
activity_df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>activity_id</th>
      <th>activity_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>WALKING</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>WALKING_UPSTAIRS</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>WALKING_DOWNSTAIRS</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>SITTING</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>STANDING</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>LAYING</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>STAND_TO_SIT</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>SIT_TO_STAND</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>SIT_TO_LIE</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>LIE_TO_SIT</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>STAND_TO_LIE</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>LIE_TO_STAND</td>
    </tr>
  </tbody>
</table>
</div>




```python
x_train = pd.read_table('/Users/nickbecker/Downloads/HAPT Data Set/Train/X_train.txt',
             header = None, sep = " ", names = features)
x_train.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tBodyAcc-Mean-1</th>
      <th>tBodyAcc-Mean-2</th>
      <th>tBodyAcc-Mean-3</th>
      <th>tBodyAcc-STD-1</th>
      <th>tBodyAcc-STD-2</th>
      <th>tBodyAcc-STD-3</th>
      <th>tBodyAcc-Mad-1</th>
      <th>tBodyAcc-Mad-2</th>
      <th>tBodyAcc-Mad-3</th>
      <th>tBodyAcc-Max-1</th>
      <th>...</th>
      <th>fBodyGyroJerkMag-MeanFreq-1</th>
      <th>fBodyGyroJerkMag-Skewness-1</th>
      <th>fBodyGyroJerkMag-Kurtosis-1</th>
      <th>tBodyAcc-AngleWRTGravity-1</th>
      <th>tBodyAccJerk-AngleWRTGravity-1</th>
      <th>tBodyGyro-AngleWRTGravity-1</th>
      <th>tBodyGyroJerk-AngleWRTGravity-1</th>
      <th>tXAxisAcc-AngleWRTGravity-1</th>
      <th>tYAxisAcc-AngleWRTGravity-1</th>
      <th>tZAxisAcc-AngleWRTGravity-1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.044</td>
      <td>-0.006</td>
      <td>-0.035</td>
      <td>-0.995</td>
      <td>-0.988</td>
      <td>-0.937</td>
      <td>-0.995</td>
      <td>-0.989</td>
      <td>-0.953</td>
      <td>-0.795</td>
      <td>...</td>
      <td>-0.012</td>
      <td>-0.315</td>
      <td>-0.713</td>
      <td>-0.113</td>
      <td>0.030</td>
      <td>-0.465</td>
      <td>-0.018</td>
      <td>-0.842</td>
      <td>0.180</td>
      <td>-0.052</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.039</td>
      <td>-0.002</td>
      <td>-0.029</td>
      <td>-0.998</td>
      <td>-0.983</td>
      <td>-0.971</td>
      <td>-0.999</td>
      <td>-0.983</td>
      <td>-0.974</td>
      <td>-0.803</td>
      <td>...</td>
      <td>0.203</td>
      <td>-0.603</td>
      <td>-0.861</td>
      <td>0.053</td>
      <td>-0.007</td>
      <td>-0.733</td>
      <td>0.704</td>
      <td>-0.845</td>
      <td>0.180</td>
      <td>-0.047</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.040</td>
      <td>-0.005</td>
      <td>-0.023</td>
      <td>-0.995</td>
      <td>-0.977</td>
      <td>-0.985</td>
      <td>-0.996</td>
      <td>-0.976</td>
      <td>-0.986</td>
      <td>-0.798</td>
      <td>...</td>
      <td>0.440</td>
      <td>-0.404</td>
      <td>-0.762</td>
      <td>-0.119</td>
      <td>0.178</td>
      <td>0.101</td>
      <td>0.809</td>
      <td>-0.849</td>
      <td>0.181</td>
      <td>-0.042</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.040</td>
      <td>-0.012</td>
      <td>-0.029</td>
      <td>-0.996</td>
      <td>-0.989</td>
      <td>-0.993</td>
      <td>-0.997</td>
      <td>-0.989</td>
      <td>-0.993</td>
      <td>-0.798</td>
      <td>...</td>
      <td>0.431</td>
      <td>-0.138</td>
      <td>-0.492</td>
      <td>-0.037</td>
      <td>-0.013</td>
      <td>0.640</td>
      <td>-0.485</td>
      <td>-0.849</td>
      <td>0.182</td>
      <td>-0.041</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.039</td>
      <td>-0.002</td>
      <td>-0.024</td>
      <td>-0.998</td>
      <td>-0.987</td>
      <td>-0.993</td>
      <td>-0.998</td>
      <td>-0.986</td>
      <td>-0.994</td>
      <td>-0.802</td>
      <td>...</td>
      <td>0.138</td>
      <td>-0.366</td>
      <td>-0.702</td>
      <td>0.123</td>
      <td>0.123</td>
      <td>0.694</td>
      <td>-0.616</td>
      <td>-0.848</td>
      <td>0.185</td>
      <td>-0.037</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 561 columns</p>
</div>




```python
y_train = pd.read_table('/Users/nickbecker/Downloads/HAPT Data Set/Train/y_train.txt',
             header = None, sep = " ", names = ['activity_id'])
y_train.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>activity_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
x_test = pd.read_table('/Users/nickbecker/Downloads/HAPT Data Set/Test/X_test.txt',
             header = None, sep = " ", names = features)
x_test.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tBodyAcc-Mean-1</th>
      <th>tBodyAcc-Mean-2</th>
      <th>tBodyAcc-Mean-3</th>
      <th>tBodyAcc-STD-1</th>
      <th>tBodyAcc-STD-2</th>
      <th>tBodyAcc-STD-3</th>
      <th>tBodyAcc-Mad-1</th>
      <th>tBodyAcc-Mad-2</th>
      <th>tBodyAcc-Mad-3</th>
      <th>tBodyAcc-Max-1</th>
      <th>...</th>
      <th>fBodyGyroJerkMag-MeanFreq-1</th>
      <th>fBodyGyroJerkMag-Skewness-1</th>
      <th>fBodyGyroJerkMag-Kurtosis-1</th>
      <th>tBodyAcc-AngleWRTGravity-1</th>
      <th>tBodyAccJerk-AngleWRTGravity-1</th>
      <th>tBodyGyro-AngleWRTGravity-1</th>
      <th>tBodyGyroJerk-AngleWRTGravity-1</th>
      <th>tXAxisAcc-AngleWRTGravity-1</th>
      <th>tYAxisAcc-AngleWRTGravity-1</th>
      <th>tZAxisAcc-AngleWRTGravity-1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.031</td>
      <td>-0.009</td>
      <td>0.040</td>
      <td>-0.939</td>
      <td>-0.945</td>
      <td>-0.759</td>
      <td>-0.952</td>
      <td>-0.950</td>
      <td>-0.802</td>
      <td>-0.757</td>
      <td>...</td>
      <td>0.123</td>
      <td>-0.346</td>
      <td>-0.709</td>
      <td>0.006</td>
      <td>0.163</td>
      <td>-0.826</td>
      <td>0.271</td>
      <td>-0.721</td>
      <td>0.277</td>
      <td>-0.051</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.043</td>
      <td>0.001</td>
      <td>-0.026</td>
      <td>-0.976</td>
      <td>-0.978</td>
      <td>-0.960</td>
      <td>-0.987</td>
      <td>-0.979</td>
      <td>-0.967</td>
      <td>-0.757</td>
      <td>...</td>
      <td>-0.315</td>
      <td>-0.143</td>
      <td>-0.601</td>
      <td>-0.083</td>
      <td>0.017</td>
      <td>-0.434</td>
      <td>0.921</td>
      <td>-0.699</td>
      <td>0.281</td>
      <td>-0.077</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.038</td>
      <td>-0.012</td>
      <td>-0.026</td>
      <td>-0.994</td>
      <td>-0.979</td>
      <td>-0.973</td>
      <td>-0.994</td>
      <td>-0.981</td>
      <td>-0.978</td>
      <td>-0.799</td>
      <td>...</td>
      <td>0.115</td>
      <td>-0.210</td>
      <td>-0.645</td>
      <td>-0.035</td>
      <td>0.202</td>
      <td>0.064</td>
      <td>0.145</td>
      <td>-0.703</td>
      <td>0.280</td>
      <td>-0.072</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.036</td>
      <td>-0.018</td>
      <td>-0.025</td>
      <td>-0.995</td>
      <td>-0.982</td>
      <td>-0.976</td>
      <td>-0.995</td>
      <td>-0.983</td>
      <td>-0.981</td>
      <td>-0.798</td>
      <td>...</td>
      <td>0.165</td>
      <td>-0.359</td>
      <td>-0.738</td>
      <td>-0.017</td>
      <td>0.154</td>
      <td>0.340</td>
      <td>0.296</td>
      <td>-0.700</td>
      <td>0.284</td>
      <td>-0.070</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.038</td>
      <td>-0.013</td>
      <td>-0.033</td>
      <td>-0.994</td>
      <td>-0.977</td>
      <td>-0.984</td>
      <td>-0.994</td>
      <td>-0.977</td>
      <td>-0.986</td>
      <td>-0.798</td>
      <td>...</td>
      <td>-0.056</td>
      <td>-0.544</td>
      <td>-0.846</td>
      <td>-0.002</td>
      <td>-0.040</td>
      <td>0.737</td>
      <td>-0.119</td>
      <td>-0.693</td>
      <td>0.291</td>
      <td>-0.067</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 561 columns</p>
</div>




```python
y_test = pd.read_table('/Users/nickbecker/Downloads/HAPT Data Set/Test/y_test.txt',
             header = None, sep = " ", names = ['activity_id'])
y_test.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>activity_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



# Building a Human Activity Classifier

Now that we've got the train and test data loaded into memory, we can start building a model to predict the activity from the acceleration and angular velocity features.

Let's build a model and plot the cross-validation accuracy curves for the training data. We'll split `x_train` and `y_train` into training and validation sets since we only want to predict on the test set once. We'll use 5-fold cross validation to get a good sense of the model's accuracy at different values of C. To plot the curves, we'll use the `validation_curve` function from sklearn.


```python
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import validation_curve

C_params = np.logspace(-6, 3, 10)
svc_2 = LinearSVC(random_state = 12)

train_scores, test_scores = validation_curve(
    svc_2, x_train.values, y_train.values.flatten(),
    param_name="C", param_range=C_params,
    cv=5, scoring="accuracy", n_jobs=-1)
```


```python
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
```


```python
y_min = 0.5
y_max = 1.1

plt.title("SVM Training and Validation Accuracy")
plt.xlabel("C Value")
plt.ylabel("Accuracy")
plt.ylim(y_min, y_max)
plt.yticks(np.arange(y_min, y_max, .1))
plt.semilogx(C_params, train_scores_mean, label="CV Training Accuracy", color="r")
plt.fill_between(C_params, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2, color="r")
plt.semilogx(C_params, test_scores_mean, label="CV Validation Accuracy",
             color="b")
plt.fill_between(C_params, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2, color="b")
plt.legend(loc="best")
plt.show()
```


![png](output_12_0.png)


From the graph, it looks like the best value of C is at 10<sup>-1</sup>. The validation accuracy begins slowly decreasing after that 10<sup>-1</sup>, indicating we are overfitting. The validation curve looks great, but we only optimized on C. We don't have to use a linear kernel. We could do a grid search on different kernels and C values. With a larger search space, we might get a different set of optimal parameters.

Let's run a parameter grid search to see what the optimal parameters are for the SVM model. We'll switch to the general Support Vector Classifier in `sklearn` so we can have the option to try non-linear kernels.


```python
from sklearn.svm import SVC

Cs = np.logspace(-6, 3, 10)
parameters = [{'kernel': ['rbf'], 'C': Cs},
              {'kernel': ['linear'], 'C': Cs}]

svc = SVC(random_state = 12)

clf = GridSearchCV(estimator = svc, param_grid = parameters, cv = 5, n_jobs = -1)
clf.fit(x_train.values, y_train.values.flatten())
```


    GridSearchCV(cv=5, error_score='raise',
           estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=12, shrinking=True,
      tol=0.001, verbose=False),
           fit_params={}, iid=True, n_jobs=-1,
           param_grid=[{'kernel': ['rbf'], 'C': array([  1.00000e-06,   1.00000e-05,   1.00000e-04,   1.00000e-03,
             1.00000e-02,   1.00000e-01,   1.00000e+00,   1.00000e+01,
             1.00000e+02,   1.00000e+03])}, {'kernel': ['linear'], 'C': array([  1.00000e-06,   1.00000e-05,   1.00000e-04,   1.00000e-03,
             1.00000e-02,   1.00000e-01,   1.00000e+00,   1.00000e+01,
             1.00000e+02,   1.00000e+03])}],
           pre_dispatch='2*n_jobs', refit=True, scoring=None, verbose=0)




```python
print clf.best_estimator_
print clf.best_params_
print clf.best_score_
```

    SVC(C=1000.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=12, shrinking=True,
      tol=0.001, verbose=False)
    {'kernel': 'rbf', 'C': 1000.0}
    0.925968842539


So the best cross-validated parameters are a `rbf` kernel with `C = 1000`. Let's use the best model to predict on the test data.


```python
clf.score(x_test, y_test)
```




    0.94339025932953824



94% accuracy. That seems pretty high. But how do know how good that actually is?

# Evaluating the Model

How much better is it than the no-information rate? The no-information rate is the accuracy we'd get if we guessed the most common class for every observation. It's the best we could do with no features.


```python
y_test.activity_id.value_counts().values[0] / y_test.activity_id.value_counts().values.sum()
```




    0.17583807716635041



Our model's accuracy of 94% looks even better compared to the baseline of 18%. That's over five times more accurate than the no-information accuracy. But where does our model fail? Let's look at the crosstab.


```python
crosstab = pd.crosstab(y_test.values.flatten(), clf.predict(x_test),
                          rownames=['True'], colnames=['Predicted'],
                          margins=True)
crosstab
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicted</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>All</th>
    </tr>
    <tr>
      <th>True</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>488</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>496</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20</td>
      <td>449</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>471</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>16</td>
      <td>399</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>420</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>453</td>
      <td>52</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>508</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
      <td>536</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>556</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>544</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>545</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>16</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>23</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>20</td>
      <td>0</td>
      <td>11</td>
      <td>0</td>
      <td>32</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
      <td>0</td>
      <td>6</td>
      <td>25</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>11</td>
      <td>1</td>
      <td>34</td>
      <td>0</td>
      <td>49</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>3</td>
      <td>15</td>
      <td>27</td>
    </tr>
    <tr>
      <th>All</th>
      <td>513</td>
      <td>472</td>
      <td>404</td>
      <td>476</td>
      <td>588</td>
      <td>545</td>
      <td>20</td>
      <td>12</td>
      <td>32</td>
      <td>29</td>
      <td>50</td>
      <td>21</td>
      <td>3162</td>
    </tr>
  </tbody>
</table>
</div>



We do really well for `activity_ids` 1-3, 5-6, and 8, but much worse for 4, 7 and 9-12. Why is that?

One possible answer is that we don't have enough data. The `All` column on the right side of the crosstab bears that out. We have way fewer observations for activities 7-12.

Additionally, it's clear the model seems to be systematically mistaking some activities for others (activities 4 and 5, 9 and 11, and 10 and 12 are confused for each other more than others). Maybe there's a reason for this. Let's put the labels on the `activity_ids` and see if we notice any patterns.

I'll also convert the crosstab to percentages to make it a bit easier to interpret.


```python
#percentages_crosstab = crosstab.iloc[:-1, :-1].apply(lambda x: x/x.sum(), axis = 1)
percentages_crosstab = crosstab.iloc[:-1, :-1]
percentages_crosstab.columns = activity_df.activity_label.values
percentages_crosstab.index = activity_df.activity_label.values
percentages_crosstab
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>WALKING</th>
      <th>WALKING_UPSTAIRS</th>
      <th>WALKING_DOWNSTAIRS</th>
      <th>SITTING</th>
      <th>STANDING</th>
      <th>LAYING</th>
      <th>STAND_TO_SIT</th>
      <th>SIT_TO_STAND</th>
      <th>SIT_TO_LIE</th>
      <th>LIE_TO_SIT</th>
      <th>STAND_TO_LIE</th>
      <th>LIE_TO_STAND</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>WALKING</th>
      <td>488</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>WALKING_UPSTAIRS</th>
      <td>20</td>
      <td>449</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>WALKING_DOWNSTAIRS</th>
      <td>5</td>
      <td>16</td>
      <td>399</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>SITTING</th>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>453</td>
      <td>52</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>STANDING</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
      <td>536</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>LAYING</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>544</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>STAND_TO_SIT</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>16</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>SIT_TO_STAND</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>SIT_TO_LIE</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>20</td>
      <td>0</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>LIE_TO_SIT</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>STAND_TO_LIE</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>11</td>
      <td>1</td>
      <td>34</td>
      <td>0</td>
    </tr>
    <tr>
      <th>LIE_TO_STAND</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>3</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>



That makes it way more clear. The model struggles to classify those `activity_ids` because they are similar actions. 4 and 5 are both stationary (sitting and standing), 9 and 11 both involving lying down (sit-to-lie and stand-to-lie), and 10 and 12 both involve standing up from a resting position (lie-to-sit and lie_to_stand). It makes sense that accelerometer data on these actions would be similar.

So with 94% accuracy in this activity classifier scenario, can we start a Fitbit competitor? Maybe.

We don't necessarily need to distinguish between the 12 different activities themselves, only whether we should count it as walking. If we can predict whether someone is walking or not from the accelerometer with near perfect accuracy, we'd be in business.

## Predicting Walking vs. Not Walking

So how do we classify walking? First we need to convert the labels to a binary to indicate whether they represent walking or staying in place. From the activity labels, we know that 1-3 are walking and everything other activity involves staying in place.


```python
y_train['walking_flag'] = y_train.activity_id.apply(lambda x: 1 if x <= 3 else 0)
y_test['walking_flag'] = y_test.activity_id.apply(lambda x: 1 if x <= 3 else 0)
```

Now, we'll train a new SVM model and find the optimal parameters with cross-validation.


```python
from sklearn.svm import SVC

Cs = np.logspace(-6, 3, 10)
parameters = [{'kernel': ['rbf'], 'C': Cs},
              {'kernel': ['linear'], 'C': Cs}]

svc = SVC(random_state = 12)

clf_binary = GridSearchCV(estimator = svc, param_grid = parameters, cv = 5, n_jobs = -1)
clf_binary.fit(x_train.values, y_train.walking_flag.values.flatten())
```

    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)
    /Users/nickbecker/anaconda/lib/python2.7/site-packages/sklearn/externals/joblib/hashing.py:197: DeprecationWarning: Changing the shape of non-C contiguous array by
    descriptor assignment is deprecated. To maintain
    the Fortran contiguity of a multidimensional Fortran
    array, use 'a.T.view(...).T' instead
      obj_bytes_view = obj.view(self.np.uint8)





    GridSearchCV(cv=5, error_score='raise',
           estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=12, shrinking=True,
      tol=0.001, verbose=False),
           fit_params={}, iid=True, n_jobs=-1,
           param_grid=[{'kernel': ['rbf'], 'C': array([  1.00000e-06,   1.00000e-05,   1.00000e-04,   1.00000e-03,
             1.00000e-02,   1.00000e-01,   1.00000e+00,   1.00000e+01,
             1.00000e+02,   1.00000e+03])}, {'kernel': ['linear'], 'C': array([  1.00000e-06,   1.00000e-05,   1.00000e-04,   1.00000e-03,
             1.00000e-02,   1.00000e-01,   1.00000e+00,   1.00000e+01,
             1.00000e+02,   1.00000e+03])}],
           pre_dispatch='2*n_jobs', refit=True, scoring=None, verbose=0)




```python
print clf_binary.best_estimator_
print clf_binary.best_params_
print clf_binary.best_score_
```

    SVC(C=0.10000000000000001, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
      max_iter=-1, probability=False, random_state=12, shrinking=True,
      tol=0.001, verbose=False)
    {'kernel': 'linear', 'C': 0.10000000000000001}
    0.999098751127



```python
clf_binary.score(x_test, y_test.walking_flag)
```




    0.99905123339658441



99.9% accuracy! Let's look at the crosstab.


```python
crosstab_binary = pd.crosstab(y_test.walking_flag.values.flatten(), clf_binary.predict(x_test),
                          rownames=['True'], colnames=['Predicted'],
                          margins=True)
crosstab_binary
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicted</th>
      <th>0</th>
      <th>1</th>
      <th>All</th>
    </tr>
    <tr>
      <th>True</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1774</td>
      <td>1</td>
      <td>1775</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1385</td>
      <td>1387</td>
    </tr>
    <tr>
      <th>All</th>
      <td>1776</td>
      <td>1386</td>
      <td>3162</td>
    </tr>
  </tbody>
</table>
</div>



That's beautiful! We can almost perfectly distinguish walking from staying in place from smartphone accelerometer data. Now we just need to measure the distance traveled and we're ready to compete with Fitbit.


```python

```
