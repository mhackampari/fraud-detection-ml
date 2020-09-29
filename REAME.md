# Fraud detection ML case study

## Summary

This study shows how machine learning algorithms can effectively to detect fraudulent payments thus automatically blocking suspicious transactions.

## Our approach

1. Open file dataset and collect first statistics
2. Divide dataset into crossvalidation and test respectively 80%-20%
3. Build baseline models
4. Data exploration
5. Feature transformation
6. Feature selection
7. Model hyperparameter tuning
8. Model crossvalidation
9. Model performance evaluation on test dataset
10. Conclusions

## How to run our scripts

In order to successfully to run our scripts you need to use `Python 3.7` ours is __Python 3.7.7__. 

We reccomend to use for this purpose a python virtualenvirment in order to avoid clashing the system libraries with ones indicated in requirements.txt. 

Install depedencies from pip:  
`pip install -r requirements.txt`

Ones installed all dependencies simply run:  
`$ jupyter-notebook Fraud\ detection\ management.ipynb`

### Dependencies

The following listing is present in `requirements.txt`

imbalanced-learn=0.7.0  
jupyter==1.0.0  
matplotlib==3.0.3  
numpy==1.18.5  
pandas==0.25.3  
scikit-learn==0.23.2  
scipy==1.4.1  
seaborn==0.9.1  
xgboost==1.2.0  
