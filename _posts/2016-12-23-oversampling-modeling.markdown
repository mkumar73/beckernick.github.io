---
title:  "The Right Way to Over-Sample in Model Training"
date:   2016-12-23
tags: [machine learning]

header:
  image: "oversampling/billygoat_a_trail_lake.jpg"

excerpt: "Model Evaluation, Oversampling, Prediction"
---

Imbalanced data is everywhere. Amazon wants to classify fake reviews, banks want to predict fraudulent credit card charges, and Facebook researchers are probably wondering if they can predict which news articles are fake.

In each of these cases, only a small fraction of observations are actually positives. I'd guess that only 1 in 10,000 credit card charges are fradulent, at most. Recently, oversampling the minority class observations has become a common approach to improve the quality of models. By oversampling, models are sometimes better able to learn patterns that differentiate classes. However, this post isn't about why this can improve modeling.

Instead, I'm going to explore how the _**timing**_ of oversampling can affect the generalization ability of a model. Since one of the primary goals of model validation is to estimate how it will perform on unseen data (in production), oversampling correctly is critical.

# Preparing the Data

I'm going to try to predict whether someone will default on or a creditor will have to charge off a loan, using data from Lending Club. I'll start by importing some packages and loading the data.


```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from imblearn.over_sampling import SMOTE
```


```python
loans = pd.read_csv('/Users/nickbecker/Python_Projects/classification_course/lending-club-data.csv.zip')
loans.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>member_id</th>
      <th>loan_amnt</th>
      <th>funded_amnt</th>
      <th>funded_amnt_inv</th>
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>grade</th>
      <th>sub_grade</th>
      <th>...</th>
      <th>sub_grade_num</th>
      <th>delinq_2yrs_zero</th>
      <th>pub_rec_zero</th>
      <th>collections_12_mths_zero</th>
      <th>short_emp</th>
      <th>payment_inc_ratio</th>
      <th>final_d</th>
      <th>last_delinq_none</th>
      <th>last_record_none</th>
      <th>last_major_derog_none</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1077501</td>
      <td>1296599</td>
      <td>5000</td>
      <td>5000</td>
      <td>4975</td>
      <td>36 months</td>
      <td>10.65</td>
      <td>162.87</td>
      <td>B</td>
      <td>B2</td>
      <td>...</td>
      <td>0.4</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>8.14350</td>
      <td>20141201T000000</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1077430</td>
      <td>1314167</td>
      <td>2500</td>
      <td>2500</td>
      <td>2500</td>
      <td>60 months</td>
      <td>15.27</td>
      <td>59.83</td>
      <td>C</td>
      <td>C4</td>
      <td>...</td>
      <td>0.8</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>2.39320</td>
      <td>20161201T000000</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1077175</td>
      <td>1313524</td>
      <td>2400</td>
      <td>2400</td>
      <td>2400</td>
      <td>36 months</td>
      <td>15.96</td>
      <td>84.33</td>
      <td>C</td>
      <td>C5</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>8.25955</td>
      <td>20141201T000000</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1076863</td>
      <td>1277178</td>
      <td>10000</td>
      <td>10000</td>
      <td>10000</td>
      <td>36 months</td>
      <td>13.49</td>
      <td>339.31</td>
      <td>C</td>
      <td>C1</td>
      <td>...</td>
      <td>0.2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>8.27585</td>
      <td>20141201T000000</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1075269</td>
      <td>1311441</td>
      <td>5000</td>
      <td>5000</td>
      <td>5000</td>
      <td>36 months</td>
      <td>7.90</td>
      <td>156.46</td>
      <td>A</td>
      <td>A4</td>
      <td>...</td>
      <td>0.8</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>5.21533</td>
      <td>20141201T000000</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 68 columns</p>
</div>



There's a lot of cool person and loan-specific information in this dataset. The target variable is `bad_loans`, which is 1 if the loan was charged off or the lessee defaulted, and 0 otherwise. I know this data is imbalanced, how imbalanced is it?


```python
loans.bad_loans.value_counts()
```




    0    99457
    1    23150
    Name: bad_loans, dtype: int64



Charge offs occurred or people defaulted on about 19% of loans, so there's some imbalance in the data but it's not terrible. I'll remove a few observations with missing values for a relevant feature and then pick a handful of features to use in a random forest model.


```python
loans = loans[~loans.payment_inc_ratio.isnull()]
```


```python
model_variables = ['grade', 'home_ownership','emp_length_num', 'sub_grade','short_emp',
            'dti', 'term', 'purpose', 'int_rate', 'last_delinq_none', 'last_major_derog_none',
            'revol_util', 'total_rec_late_fee', 'payment_inc_ratio', 'bad_loans']

loans_data_relevent = loans[model_variables]
```

Next, I need to one-hot encode the categorical features as binary variables to use them in sklearn' random forest classifier.


```python
loans_relevant_enconded = pd.get_dummies(loans_data_relevent)
```

# Creating the Training and Test Sets

With the data prepared, I can create a training dataset and a test dataset. I'll use the training dataset to build and validate the model, and treat the test dataset as the "unseen" new data I'd see if the model were in production.


```python
training_features, test_features, \
training_target, test_target, = train_test_split(loans_relevant_enconded.drop(['bad_loans'], axis=1),
                                               loans_relevant_enconded['bad_loans'],
                                               test_size = .1,
                                               random_state=12)
```

# The Wrong Way to Oversample

With my training data created, I'll upsample the bad loans using the [SMOTE algorithm](https://www.jair.org/media/953/live-953-2037-jair.pdf) (Synthetic Minority Oversampling Technique). At a high level, SMOTE creates synthetic observations of the minority class (bad loans) by:

1. Finding the k-nearest-neighbors for minority class observations (finding similar observations)
2. Randomly choosing one of the k-nearest-neighbors and using it to create a similar, but randomly tweaked, new observation.

After upsampling with to a class ratio of 1.0, I should have a balanced dataset. There's no need (and often it's not smart) to balance the classes, but it magnifies the issue caused by incorrectly timed oversampling.


```python
sm = SMOTE(random_state=12, ratio = 1.0)
x_res, y_res = sm.fit_sample(training_features, training_target)
```


```python
print training_target.value_counts(), np.bincount(y_res)
```

    0    89493
    1    20849
    Name: bad_loans, dtype: int64 [89493 89493]


After upsampling, I'll split the data into separate training and validation sets and build a random forest model to classify the bad loans.


```python
x_train_res, x_val_res, y_train_res, y_val_res = train_test_split(x_res,
                                                    y_res,
                                                    test_size = .1,
                                                    random_state=12)
```


```python
clf_rf = RandomForestClassifier(n_estimators=25, random_state=12)
clf_rf.fit(x_train_res, y_train_res)
clf_rf.score(x_val_res, y_val_res)
```




    0.88468629532376108



88% accuracy looks good, but I'm not just interested in accuracy. I also want to know how well I can specifically classify bad loans, since they're more important. In statistics, this is called [recall](https://en.wikipedia.org/wiki/Sensitivity_and_specificity), and it's the number of correctly predicted "positives" divided by the total number of "positives".


```python
recall_score(y_val_res, clf_rf.predict(x_val_res))
```




    0.81192097332291546



81% recall. That means the model correctly identified 81% of the total bad loans. That's pretty great. But is this actually representative of how the model will perform? To find out, I'll calculate the accuracy and recall for the model on the test dataset I created initially.


```python
print clf_rf.score(test_features, test_target)
print recall_score(test_target, clf_rf.predict(test_features))
```

    0.801973737868
    0.129943502825


Only 80% accuracy and 13% recall on the test data. That's a **huge** difference!

# What Happened?

When the model is in production, it's predicting on unseen data. The entire point of using a validation set is to estimate how the model will generalize to that new data.

By oversampling before splitting into training and validation datasets, I "bleed" information from the validation set into the training of the model.

To see how this works, think about the case of simple oversampling (where I just duplicate observations). If I upsample a dataset before splitting it into a train and validation set, I could end up with the same observation in both datasets. As a result, the model will be able to perfectly predict the value for those observations when predicting on the validation set, inflating the accuracy and recall.

When upsampling using SMOTE, I don't create duplicate observations. However, because the SMOTE algorithm uses the nearest neighbors of observations to create synthetic data, it still "bleed" information. If the nearest neighbors of minority class observations in the training set end up in the validation set, their information is partially reflected by the synthetic data in the training set. Since I'm splitting the data randomly, we'd expect to have this happen. As a result, the model will be better able to predict validation set values than on completely new data.

When I predict on the unseen test data, though, the "false boost" disappears, and I get the true generalization results.

# The Right Way to Oversample

Okay, so I've gone through the wrong way to oversample. Now I'll go through the right way: oversampling on only the training data.


```python
x_train, x_val, y_train, y_val = train_test_split(training_features, training_target,
                                                  test_size = .1,
                                                  random_state=12)
```


```python
sm = SMOTE(random_state=12, ratio = 1.0)
x_train_res, y_train_res = sm.fit_sample(x_train, y_train)
```

By oversampling only on the training data, none of the information in the validation data is being used to create synthetic observations. So these results should be generalizable. Let's see if that's true.


```python
clf_rf = RandomForestClassifier(n_estimators=25, random_state=12)
clf_rf.fit(x_train_res, y_train_res)
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                n_estimators=25, n_jobs=1, oob_score=False, random_state=12,
                verbose=0, warm_start=False)




```python
print 'Validation Results'
print clf_rf.score(x_val, y_val)
print recall_score(y_val, clf_rf.predict(x_val))
```

    Validation Results
    0.800362483009
    0.138195777351



```python
print 'Test Results'
print clf_rf.score(test_features, test_target)
print recall_score(test_target, clf_rf.predict(test_features))
```

    Test Results
    0.803278688525
    0.142546718818


The validation results closely match the "unseen" test data results, which is exactly what I would want to see after putting a model into production.

# Conclusion

Oversampling is well-covered way to potentially improve models trained on imbalanced data. But it's important to remember that oversampling incorrectly can lead to thinking a model will generalize better than it actually does. Random forests are great because they don't overfit (see [Brieman 2001](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf) for a proof), but poor sampling practices can still lead to false conclusions about the quality of a model.

If the decision to put a model into production is based on how it performs on a validation set, it's critical that oversampling is done correctly.
