# Portfolio Optimization

**Problem**: Is it possible to use macroeconomic data to "predict" an optimal asset allocation for a portfolio to 
achieve better risk-adjusted returns?

In this project we aim to provide an answer (or at least some insights into this question).

We structured the project as a Data Science showcase, in which we explain the various steps we took in a series of
Jupyter notebooks, containing both the code and the thought process.

[Start here](notebooks/data_collection.ipynb)

We recommend you explore the project using the notebooks provided, but you are also welcome to clone the project code
and play with it on your own if you want to try different modeling techniques or use different data.

### What we did:
1. [Data Collection](notebooks/data_collection.ipynb) (get features and targets data and store it in MongoDB)
2. [Data Cleaning](notebooks/data_cleaning.ipynb) (Wrangle data and store cleaned data in PostgreSQL)
3. [EDA](notebooks/eda.ipynb) (Explore data and gain insights to select features)
4. [Feature Selection](notebooks/feature_selection.ipynb) (Choose the best features to use in our model)
5. [Modeling](notebooks/modeling.ipynb) (Try different models and compare their performance metrics)
6. [Deployment](notebooks/deployment.ipynb) (Deploy a schedulable ETL process w/ AWS Fargate and a prediction API w/ AWS Lambda)