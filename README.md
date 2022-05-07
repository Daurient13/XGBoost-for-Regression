# XGBoost-for-Regression
SKIP

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

and in the categoric column are: 'country', 'status'.

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




# New Technique: BayesianSearchCV
The main difference between Bayesian search and the other methods is that the tuning algorithm optimizes its parameter selection in each round according to the previous round score. Thus, instead of randomly choosing the next set of parameters, the algorithm optimizes the choice, and likely reaches the best parameter set faster than the previous two methods. Meaning, this method chooses only the relevant search space and discards the ranges that will most likely not deliver the best solution. Thus, it can be beneficial when you have a large amount of data, the learning is slow, and you want to minimize the tuning time.
