# XGBoost-for-Regression

XGBoost is an open-source software library that implements optimized distributed gradient boosting machine learning algorithms under the Gradient Boosting framework.

XGBoost, which stands for Extreme Gradient Boosting, is a scalable, distributed gradient-boosted decision tree (GBDT) machine learning library. It provides parallel tree boosting and is the leading machine learning library for regression, classification, and ranking problems.

It’s vital to an understanding of XGBoost to first grasp the machine learning concepts and algorithms that XGBoost builds upon: supervised machine learning, decision trees, ensemble learning, and gradient boosting.

Supervised machine learning uses algorithms to train a model to find patterns in a dataset with labels and features and then uses the trained model to predict the labels on a new dataset’s features.

Decision trees create a model that predicts the label by evaluating a tree of if-then-else true/false feature questions, and estimating the minimum number of questions needed to assess the probability of making a correct decision. Decision trees can be used for classification to predict a category, or regression to predict a continuous numeric value. In the simple example below, a decision tree is used to estimate a house price (the label) based on the size and number of bedrooms (the features).

![image](https://user-images.githubusercontent.com/86812576/167251237-195dcfa4-ab50-449b-bc2e-60a54430679e.png)

A Gradient Boosting Decision Trees (GBDT) is a decision tree ensemble learning algorithm similar to random forest, for classification and regression. Ensemble learning algorithms combine multiple machine learning algorithms to obtain a better model.

Both random forest and GBDT build a model consisting of multiple decision trees. The difference is in how the trees are built and combined.

Random forest uses a technique called bagging to build full decision trees in parallel from random bootstrap samples of the data set. The final prediction is an average of all of the decision tree predictions.

The term “gradient boosting” comes from the idea of “boosting” or improving a single weak model by combining it with a number of other weak models in order to generate a collectively strong model. Gradient boosting is an extension of boosting where the process of additively generating weak models is formalized as a gradient descent algorithm over an objective function. Gradient boosting sets targeted outcomes for the next model in an effort to minimize errors. Targeted outcomes for each case are based on the gradient of the error (hence the name gradient boosting) with respect to the prediction.

GBDTs iteratively train an ensemble of shallow decision trees, with each iteration using the error residuals of the previous model to fit the next model. The final prediction is a weighted sum of all of the tree predictions. Random forest “bagging” minimizes the variance and overfitting, while GBDT “boosting” minimizes the bias and underfitting.

XGBoost is a scalable and highly accurate implementation of gradient boosting that pushes the limits of computing power for boosted tree algorithms, being built largely for energizing machine learning model performance and computational speed. With XGBoost, trees are built in parallel, instead of sequentially like GBDT. It follows a level-wise strategy, scanning across gradient values and using these partial sums to evaluate the quality of splits at every possible split in the training set. 

![image](https://user-images.githubusercontent.com/86812576/167251257-39aa42e6-db52-4cc4-bb73-1e31935c7d4f.png)


# Dataset

In this project, the dataset that I use is life expectancy. So we will predict the life expectancy of a country, or we can get insight on how to increase our life expectancy. The dataset consists of 2928 rows and 22 columns. and we have prepared a description of each column below:

![image](https://user-images.githubusercontent.com/86812576/167155836-5a19d0b1-d712-4432-b155-437895b75d1f.png)

country           : Country

year              : Year

status                  : Developed or Developing

adult_mortality         : adult death rate (age 15-60 per 1000 population)

infant_deaths           : number of infant deaths per 1000 population

Alcohol                 : alcohol consumption per capita (age 15+, liters)

percentage_expenditure  : percentage of expenditure on health of GDP per capita

HepB                    : Hepatitis B immunization percentage for 1 year old

BMI                     : the average body mass index of the entire population

Measles                 : number of measles cases per 1000 population

u5_deaths               : number of under-five deaths per 1000 population

Polio                   : percentage of polio immunization for 1 year old

total_expenditure       : percentage of government spending on health (%)

DPT                     : percentage of diphtheria, pertussis and tetanus immunizations for 1 year old

HIV_AIDS                : death per 1000 births HIV/AIDS (age 0-4 years)

GDP                     : GDP per capita (USD)

population              : country population

Thinness_10_19          : the percentage of thinness in children aged 10-19 years

Thinness_5_9            : the percentage of thinness in children aged 5-9 years

HDI                     : human development index in terms of resource income composition (0 to 1)

school_year             : number of years of school

life_expectancy         : life expectancy (age) **as target**

# Import Package

import common package:

import **numpy** as **np**

import **pandas** as **pd**

from **sklearn.model_selection** import **train_test_split**

from **sklearn.pipeline** import **Pipeline**

from **sklearn.compose** import **ColumnTransformer**

from **jcopml.utils** import **save_model, load_model**

from **jcopml.pipeline** import **num_pipe, cat_pipe**

from **jcopml.plot** import **plot_missing_value**

from **jcopml.feature_importance** import **mean_score_decrease**

import Algorithm's Package:

from **xgboost** import **XGBRegressor**

from s**klearn.model_selection** import **RandomizedSearchCV**

from **jcopml.tuning** import **random_search_params as rsp**

from _**skopt**_ import **_BayesSearchCV_**

from **jcopml.tuning** import **bayes_search_params as bsp**

import fine parameter

from j**copml.tuning.space** import **Integer, Real**

# Import Data

which i have explained before.

# Mini Exploratory Data Analysis

I always work on data science projects with simple think so that I can benchmark. Using a simple model to benchmark. And most of the time it's more efficient and sometimes find a good one. but at the beginning I did mini Exploratory Data Analysis. **_because i focus more on the algorithm_**.

If we look at the data there are many missing values. Actually it's not that bad, and we can impute it at the _preprocessor_. So, no data will be discarded except in the life_expectancy column as a target, because in 'life_expectancy' column there is a missing value and for the target column this should not happen because we are not working on _Unsupervised Learning_. **_But keep in mind that what is discarded is the row with the missing value, not the entire target column_**.

**_lesson_: _in Supervised Learning the target column must not have missing values_.**


# Dataset Splitting

split the data into X, and y

X = all columns except the target column.

y = 'life_expectancy' as target

test_size = 0.2 (which means 80% for train, and 20% for test)

# Training

In the Training step there are 3 main things that I specify.

First, the preprocessor: here the columns will be grouped into numeric and categoric.

included in the numeric column are: 'year', 'adult_mortality', 'infant_deaths', 'alcohol', 'percentage_expenditure', 
                            'HepB', 'measles', 'BMI', 'u5_deaths', 'Polio', 'total_expenditure', 'DPT', 
                            'HIV_AIDS', 'GDP', 'population', 'thinness_10_19', 'thinness_5_9', 'HDI', 'school_year'.

and in the categoric column are: 'country', 'status'. In categorical column I do encoding with onehot encoder.

second, pipeline: contains the preprocessor as 'prep' which I defined earlier, and the algorithm as 'algo' which in this case I use _**XGBRegressor**_.

and third, tuning with RandomizedSearchCV: in this case I use the tuning recommendations (rsp.xgb_params) that often occur in many cases. but does not rule out hyperparameter tuning if the model results are not good. with cross validation = 3, and n_iter = 50 (trials in Random Search)

**Random Search Parameters Recommendation** :

**{'algo__max_depth': Integer(low=1, high=10),**

 **'algo__learning_rate': Real(low=-2, high=0, prior='log-uniform'),**
 
 **'algo__n_estimators': Integer(low=100, high=200),**
 
 **'algo__subsample': Real(low=0.3, high=0.8, prior='uniform'),**
 
 **'algo__gamma': Integer(low=1, high=10),**
 
 **'algo__colsample_bytree': Real(low=0.1, high=1, prior='uniform'),**
 
 **'algo__reg_alpha': Real(low=-3, high=1, prior='log-uniform'),**
 
 **'algo__reg_lambda': Real(low=-3, high=1, prior='log-uniform')}**

About Pamaterers:

_subsample_ = data to subsample (rows)

_colsample_bytree_ = _max_features_ = subsampling column (feature) 

_gamma_ = _min_impurity_decrease_ = minimum loss reduction for split to happen 

_reg_alpha_, and _reg_lambda_ are reguralization (to reduce overfit)

Why do i prefer to use Random Search over Grid Search? because XGBoost has many parameters and it will take a long time to use Grid Search because it will try all combinations. If I use Random Search then it will only trial the number that I have determined so that it will save a lot of time and be more computationally efficient.

# Results and Many Others

![1m](https://user-images.githubusercontent.com/86812576/167234320-4658d3eb-1876-457c-8582-b02c9bdd7e37.png)

from the result above it can be seen that we have a good model with 96% score reached. After this section i will go to Feature Importance.

### Feature Importance

![image](https://user-images.githubusercontent.com/86812576/167234596-41793603-878d-41c5-9f5e-9f3950d33c14.png)

It turns out that if we look at the Feature Importance (Mean Square Decrease) the most important features are:

1.  'adult_mortality',
  
2.  'HDI',
  
3.  'HIV_AIDS',
  
4.  'u5_deaths',
  
5.  'country',
  
6.  'BMI',
  
7.  'thinness_5_9',
  
8  'school_year',
  
9.  'year',
  
10. 'thinness_10_19']

And other features don't seem to have any effect. So we can cut out the features, and focus on those ten features. why do we want to cut features? why don't we use all the features? because the machine has a weakness. If we provide more information, then the pattern will be more difficult to find. because the machine only looks for patterns. it's like a maze, but if we give things that are important, specific to something then it will be easier to find, generalizing and produce a better model

We can't guarantee, but most of the time it can helps. it could be a better model, or it could be worst.

# Model After Feature Importance

### Mean Score Decrease

Now i only select 10 features to the model after Feature Importance, and all these 5 features is categoric columns. Here's the result:

![2m](https://user-images.githubusercontent.com/86812576/167235061-d933f0f4-51f9-40ca-b7e3-860eae96f0bc.png)

The result is not much change in the score. both produce a score of 96%. 

# Polynomial

I also put Polynomial in the model. Here's the result:

![pol](https://user-images.githubusercontent.com/86812576/167235844-e31040d9-9c66-45eb-911b-cfbe4e02d5ea.png)

When I add Poly, the resulting score actually decreases. And it turns out the second model after feature importance is the best model.

# Other Technique: BayesianSearchCV

The main difference between Bayesian search and the other methods is that the tuning algorithm optimizes its parameter selection in each round according to the previous round score. Thus, instead of randomly choosing the next set of parameters, the algorithm optimizes the choice, and likely reaches the best parameter set faster than the previous two methods. Meaning, this method chooses only the relevant search space and discards the ranges that will most likely not deliver the best solution. Thus, it can be beneficial when you have a large amount of data, the learning is slow, and you want to minimize the tuning time.

Bayesian Search will fit 3 every trial and create probability. Depending on the cross validation that we set, in this case cv = 3. I have also added Bayesian Search to the first model and the results are slightly better although not significant which is around 0.7

![bay](https://user-images.githubusercontent.com/86812576/167236796-4c972ca5-f342-4d02-a3c4-b268f182f1cd.png)

# Feature Importance for Bayesian Search Model

![image](https://user-images.githubusercontent.com/86812576/167236867-38436655-dbb2-4d31-add4-55bc03969008.png)

not too many changes to feature importance. Here's the top 10 feature importance and the resulting model:

1.  'adult_mortality',
  
2.  'HDI',
  
3.  'HIV_AIDS',
  
4.  'thinness_5_9',
  
5.  'u5_deaths',
  
6.  'BMI',
  
7.  'school_year',
  
8   'country',
  
9.  'year',
  
10. 'alcohol']

![imp](https://user-images.githubusercontent.com/86812576/167237129-5035bf9e-fd28-40a1-bb39-3bd457638b5a.png)

looks like the best score we can get is 0.96. 

# Polynomial 

The last is i add polynomial after feature importane in Bayesian Search Model: 

![pol2](https://user-images.githubusercontent.com/86812576/167237284-31aebe5f-8719-47aa-a112-76439a1a1337.png)

Well... it looks like I'm stuck in the model with a score of 0.96. this is actually a good model.

# Prediction

I want to make sure whether the model is working well, for that I will try to predict the X_test data.
