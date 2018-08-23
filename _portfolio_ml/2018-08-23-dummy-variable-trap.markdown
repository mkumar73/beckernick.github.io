---
title:  "Beware of Dummy Variable Trap"
date:   2018-08-23
tags: [machine learning]

excerpt: "categorical variable, label encoding, one-hot encoding"
---

## Need of dummy variables?

Before we start the discussion of dummy variables, it is important to know why we
need such constructs. When we talk about building machine learning models, we keep in mind
that we are dealing with different kinds of data types and one of such ubiquitous kind is 
"Categorical Variables" like Gender (Male, Female), Color (Red, Green, Blue) etc.

## How to deal with Categorical Variables?

Generally, not all machine learning models can deal with categorical variables, 
as a result, they have to be encoded to using some custom logic or using inbuilt functions
like LabelEncoder or OneHotEncoder.

## What are dummy Variables?