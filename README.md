# Predict Customer Churn using Random Forest Classification

Customer retention is one of the primary growth pillars for products with a returning customers and subscription models.

Customer churn (or customer attrition) is a tendency of customers to abandon a brand and stop being a paying client of a particular business. The percentage of customers that discontinue using a company’s products or services during a particular time period is called a customer churn (attrition) rate. 

Churn rate is a health indicator for businesses whose customers are subscribers and paying for services on a recurring basis


# Data Source and Preperation

Dataset is Telecommunication Churn from Kaggle Dataset 

The training dataset contains 4250 samples. Each sample contains 19 features and 1 boolean variable "churn" which indicates the class of the sample. The 19 input features and 1 target variable

In Data cleaning and preparation, We clean the dataset and we handle the categorical variables into machine learning format (Check the Jupyter notebook)

## Exploratory Data Analysis


## 1. No of Users per churn (If the churn number is 1 the customer has discontinued)

![image](Images/chrun_count_plot.png)

## 2. Number of Customer Service Calls made per churn

![image](Images/number_customer_service_calls_churn_rate.png)

## 3. Churn Rate per state in the US

![image](Images/state_chrun_plot.png)

## 4. Total Daily calls and how it affects churn rate

![image](Images/total_daily_calls_and_churn_rate.png)

## 5. Total Evening Calls and how it affects churn rate

![image](Images/total_eve_call_and_churn_rate.png)


## Data Cleaning and Preprocessing

## 1. Check for missing values

Make there is no missing date using train.isnull().sum()

## 2. Deleting highly corelated variables

We check for highly corelated variables using corr() function and delete variables which are highly corelated with other variables

## 3. Handling Categorical Variables

We handle categorical features like area code using map function 

# Building a churn prediction model

## 1. Feature Scaling 

We use "from sklearn.preprocessing import StandardScaler" to normalize the all the values 

## 2. We split the dataset into train and test

## 3. Random Forest Regressor Model

We fit the model using from sklearn.ensemble import RandomForestRegressor with random state 42 for reproducability

## 4. Fine Tune Model

We fine tune the model using GridSearchCV by using different parameters

## 5. Predict

We use predict function to find the accuracy of the model

# License
This code pattern is licensed under the Apache Software License, Version 2.  Separate third party code objects invoked within this code pattern are licensed by their respective providers pursuant to their own separate licenses. Contributions are subject to the [Developer Certificate of Origin, Version 1.1 (DCO)](https://developercertificate.org/) and the [Apache Software License, Version 2](https://www.apache.org/licenses/LICENSE-2.0.txt).

[Apache Software License (ASL) FAQ](https://www.apache.org/foundation/license-faq.html#WhatDoesItMEAN)
