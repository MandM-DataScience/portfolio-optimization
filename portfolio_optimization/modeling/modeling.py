import json
import sys
import time
import warnings
from datetime import date, datetime

from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_halving_search_cv # noqa

from dateutil.relativedelta import relativedelta
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression, LogisticRegressionCV, \
    PassiveAggressiveClassifier, RidgeCV, SGDRegressor, ElasticNet, Lars, LassoLars, LassoLarsIC, \
    OrthogonalMatchingPursuit, ARDRegression, BayesianRidge, MultiTaskElasticNet, MultiTaskLasso, HuberRegressor, \
    QuantileRegressor, RANSACRegressor, TheilSenRegressor, PoissonRegressor, TweedieRegressor, GammaRegressor, \
    PassiveAggressiveRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, HalvingGridSearchCV, RandomizedSearchCV, \
    train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVR, NuSVR, SVR
from sklearn.tree import DecisionTreeRegressor
from statsmodels.formula.api import ols
from statsmodels.graphics.tsaplots import plot_acf

from portfolio_optimization.data_lake.mongodb import upsert_document, insert_document, get_collection_documents
from portfolio_optimization.db import get_df_from_table
import numpy as np

from portfolio_optimization.eda import get_indicator_name
from portfolio_optimization.helper import get_feature_target_shifted, standardize_data, get_stationarity, \
    convert_to_stationary
from sklearn.decomposition import PCA
import sklearn.metrics as metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

def get_readytomodel_df(shift_periods, stationary=False, mode="diff", n_diff=1,
                        start_date=date(1960, 1, 1), end_date=date(2022, 12, 31)):
    where_clause = None
    if start_date is not None:
        where_clause = f"where date >= '{start_date}'"

    if end_date is not None:
        where_clause = f"where date <= '{end_date}'"

    if start_date is not None and end_date is not None:
        where_clause = f"where date between '{start_date}' and '{end_date}'"

    df = get_df_from_table("pivot", where_clause)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values(by="date").set_index("date")
    df = df.apply(pd.to_numeric)
    df = df.asfreq('MS')

    if stationary:
        df = convert_to_stationary(df, mode, n_diff)

    return get_feature_target_shifted(df, shift_periods)

def split_features_target(df, feature_columns, t_col, holdout_years=0):

    r_df = df[feature_columns + [t_col]]

    # Because of stationary transformation we have a series of 0 and only a single value for yearly/quarterly data
    # We want to replicate the difference between the period and the previous one on all the months in that period
    r_df = r_df.replace(0, np.nan)
    r_df = r_df.fillna(method="ffill", limit=11)

    r_df = r_df.dropna()

    max_date = list(r_df.index)[-1]
    holdout_date = max_date - relativedelta(years=holdout_years)

    r_df = r_df[r_df.index <= holdout_date]
    # print(r_df.index)

    # print(t_col, len(r_df), holdout_date)

    # Standardize
    X = r_df[feature_columns]
    X = standardize_data(X)
    y = r_df[t_col]

    return X, y

def apply_pca(X, pca_components):
    p_ = PCA(n_components=pca_components)
    idx = X.index
    cols = [f"featurePCA{i + 1}" for i in range(pca_components)]
    X = p_.fit_transform(X)
    X = pd.DataFrame(X, columns=cols)
    X.index = idx
    return X

def apply_poly(X, poly_degree):

    p_ = PolynomialFeatures(degree=poly_degree)
    idx = X.index
    cols = X.columns
    X = p_.fit_transform(X)

    cols = p_.get_feature_names_out(cols)

    # cols = [f"featurePoly{i + 1}" for i in range(p_.n_output_features)]

    X = pd.DataFrame(X, columns=cols)
    X.index = idx

    return X

def cv_optimization(X, y, search="grid", scoring={"mae":"neg_mean_absolute_error"}, refit="mae",
                    n_splits=20, test_size=6, verbose=2):

    pipe = Pipeline(steps=[
        ("reduce_dim", None),
        ("model", None)
    ])

    common_params = {
        'reduce_dim': [PCA(n_components=10), 'passthrough'],
    }

    param_grid = [
        {
            # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
            "model": [LinearRegression()],
            # "model__positive": [True, False],
        },
        {
            # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
            "model": [Ridge()],
            "model__alpha": [1, 10, 100],
            # "model__positive": [True, False],
            # "model__solver":['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'],
        },
        {
            # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html
            "model": [SGDRegressor()],
            # "model__loss": ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
            # "model__penalty": ['l2', 'l1', 'elasticnet', None],
            # "model__alpha": [0.0001],
            "model__l1_ratio": [0.15, 0.5],
            # "model__tol": [0.001, 0.01, 0.0001],
            # "model__epsilon": [0.1, 0.01, 0.5], # only if loss is in 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive',
            "model__learning_rate": ['invscaling', 'adaptive'],
            "model__eta0": [0.001],
            "model__power_t": [0.25],
            "model__early_stopping": [True],
            "model__validation_fraction": [0.1],
        },
        {
            # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
            "model": [ElasticNet()],
            "model__alpha": [2, 3],
            "model__l1_ratio": [1],
            # "model__tol": [0.0001, 0.001, 0.00001],
            # "model__positive": [True, False],
            "model__selection": ['cyclic', 'random'],
        },
        {
            # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lars.html
            "model": [Lars()],
            "model__normalize": [False],
            # "model__n_nonzero_coefs": [500, np.inf],
            # "model__fit_path": [True, False],
        },
        {
            # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
            "model": [Lasso()],
            "model__alpha": [2, 3],
            # "model__tol": [0.0001, 0.001, 0.00001],
            # "model__positive": [True, False],
            "model__selection": ['cyclic', 'random'],
        },
        {
            # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html
            "model": [LassoLars()],
            "model__alpha": [1, 2, 5],
            "model__normalize": [False],
            "model__fit_path": [True, False],
            # "model__positive": [True, False],
        },
        {
            # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLarsIC.html
            "model": [LassoLarsIC()],
            "model__criterion": ['aic','bic'],
            "model__normalize": [False],
            # "model__positive": [True, False],
        },
        {
            # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuit.html
            "model": [OrthogonalMatchingPursuit()],
            "model__normalize": [False],
        },
        {
            # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html
            "model": [ARDRegression()],  # molto lento
            # "model__n_iter": [300],
            "model__tol": [0.1],
            # "model__alpha_1": [0.000001],
            # "model__alpha_2": [0.000001],
            # "model__lambda_1": [0.00001],
            # "model__lambda_2": [0.000001],
            # "model__compute_score": [True, False],
            "model__threshold_lambda": [5000],
        },
        {
            # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html
            "model": [BayesianRidge()],
            # "model__n_iter": [300],
            # "model__tol": [0.001],
            # "model__alpha_1": [0.000001],
            # "model__alpha_2": [0.000001],
            # "model__lambda_1": [0.000001],
            # "model__lambda_2": [0.000001],
            # "model__compute_score": [True, False],
        },
        {
            # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html
            "model": [HuberRegressor()],
            "model__epsilon": [1],
            # "model__max_iter": [100, 500],
            "model__alpha": [0.001],
            "model__tol": [0.0001],
        },
        {
            # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html
            "model": [QuantileRegressor()],  # Molto lento con interior-point e 'revised simplex'
            "model__quantile": [0.5],
            "model__alpha": [1, 2, 5],
            "model__solver":  ['highs-ds', 'highs'],
        },
        {
            # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html
            "model": [TweedieRegressor()],
            "model__power": [0], # 0=Normal, 1=Poisson, 1.2=CompoundPoissonGamma, 2=Gamma, 3=InverseGaussian
            "model__alpha": [1, 2],
            # "model__link": ['auto'], # 'auto', 'identity', 'log'
            # "model__solver": ['lbfgs', 'newton-cholesky'], new in version 1.2
            # "model__max_iter": [100, 500],
            "model__tol": [0.0001, 0.001, 0.00001],
        },
        {
            # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveRegressor.html
            "model": [PassiveAggressiveRegressor()],
            "model__C": [0.1],
            "model__max_iter": [1000],
            "model__tol": [0.001, 0.01, 0.0001],
            "model__early_stopping": [True],
            "model__validation_fraction": [0.1, 0.01, 0.5],
            "model__n_iter_no_change": [5],
            "model__loss": ['epsilon_insensitive'], # The loss function to be used: epsilon_insensitive: equivalent to PA-I in the reference paper. squared_epsilon_insensitive: equivalent to PA-II in the reference paper.
            "model__epsilon": [0.1, 0.05],
        },
        {
            # https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html
            "model": [KernelRidge()],
            "model__alpha": [1.0],
            "model__kernel": ["laplacian"],
            # "model__degree": [2, 3, 4, 5], #only for polynomial
            # "model__coef0": [1], #only for polynomial/sigmoid
        },
        {
            # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
            "model": [KNeighborsRegressor()],
            "model__n_neighbors": [10],
            "model__weights": ["uniform","distance"],
            "model__algorithm": ["auto"],
            "model__p": [1, 2] # 1 manhattan distance, 2 euclidian distance
        },
        {
            # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
            "model": [DecisionTreeRegressor()],
            "model__criterion": ["squared_error","friedman_mse","absolute_error"],
            "model__splitter": ["best"],
            # "model__max_depth": [None]
        },
        {
            # https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html
            "model": [LinearSVR()],
            "model__epsilon": [0, 1, 2],
            "model__tol": [0.0001, 0.001, 0.00001],
            "model__C": [2, 5],
            "model__loss": ["epsilon_insensitive"],
            "model__dual": [True],
            # "model__max_iter": [100]
        },
        # {
        #     # https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html
        #     "model": [NuSVR()],
        #     # "model__nu": [0.5],
        #     "model__kernel": ["poly","sigmoid"],
        #     "model__degree": [2, 3, 4], # for poly
        #     # "model__gamma": ["scale", "auto"],
        #     # "model__coef0": [0.0] # for poly, sigmoid
        # },
        {
            # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
            "model": [SVR()],
            "model__C": [1.0, 5],
            "model__kernel": ["poly"],
            # "model__gamma": ["scale", "auto"],
            # "model__coef0": [0.0], # for poly, sigmoid
            "model__epsilon": [0.1, 0.01, 0.5]
        },
        {
            # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
            "model": [MLPRegressor()],
            "model__hidden_layer_sizes": [(10,),(10,5,)],
            "model__activation": ["relu"],
            "model__solver": ["adam"],
            "model__alpha": [0.0001],
            "model__learning_rate": ["invscaling"],
            "model__learning_rate_init": [0.001],
            # "model__early_stopping": [True],
            "model__batch_size": [5],
            "model__max_iter": [100],
            # "model__tol": [1]
        }
    ]

    for d in param_grid:
        d.update(common_params)

    print(param_grid)

    # scoring methods
    # https://scikit-learn.org/stable/modules/model_evaluation.html

    start = time.time()

    if search == "grid":
        cv = GridSearchCV(pipe, param_grid, cv=TimeSeriesSplit(n_splits=n_splits, test_size=test_size),
                          scoring=scoring, verbose=verbose, refit=refit)

    elif search == "random":
        cv = RandomizedSearchCV(pipe, param_grid, cv=TimeSeriesSplit(n_splits=n_splits, test_size=test_size),
                          scoring=scoring, verbose=verbose, refit=refit, n_iter=1000)

    cv.fit(X, y)
    df = pd.DataFrame(cv.cv_results_)

    columns = ["params"]
    for _s in scoring:
        columns += [f"mean_test_{_s}", f"rank_test_{_s}"]
    df = df[columns]
    df = df.sort_values(by=f"rank_test_{list(scoring)[0]}")

    if verbose > 1:
        print(df)
        print("time: ",time.time()-start)

    return df


def single_model_regression_CV(X, y, y_original, model, pca=False, pca_components=10, poly=False, poly_degree=2, plot=False,
                            n_splits=None, test_size=1):

    if poly:
        X = apply_poly(X, poly_degree)
    if pca:
        X = apply_pca(X, pca_components)

    # print(X.tail(25))

    # TimeSeriesSplit
    if n_splits is None:
        n_splits = int(len(X) / 2)
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

    dates_list = []
    actual_list = []
    prediction_list = []
    training_r2_list = []
    test_r2_list = []

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        y_original_train, y_original_test = y_original.iloc[train_index], y_original.iloc[test_index]

        # print(X_train.head(100))
        # print(y_train.head(100))
        # print(X_test)
        # print(y_test)
        # return


        model.fit(X_train, y_train)
        # Show MLPRegressor loss curve
        # print(f"loss after {len(model.loss_curve_)} epochs: {model.loss_curve_[-1]}")
        # pd.DataFrame(model.loss_curve_).plot()
        # plt.show()
        # return

        y_hat = model.predict(X_test)

        if test_size == 1:
            dates_list.append(X_test.index[0])
            actual_list.append(y_test.iloc[0])
            prediction_list.append(y_hat[0])
        else:
            dates_list.extend(X_test.index)
            actual_list.extend(y_test)
            prediction_list.extend(y_hat)
            test_r2_list.append(r2_score(y_test, y_hat))

        training_r2_list.append(model.score(X_train, y_train))

        # print(X_test.index[0], round(y_test[0],0), round(y_hat[0],0))

        # val = 0
        # for i in range(len(X_test.columns)):
        #     val += model.coef_[i] * X_test[X_test.columns[i]][0]
        #     print(X_test.columns[i], round(model.coef_[i],2), "x", round(X_test[X_test.columns[i]][0],2), "=",
        #           round(model.coef_[i]*X_test[X_test.columns[i]][0],2), ". Value = ", round(val,2))
        # print("intercept", model.intercept_)
        # val += model.intercept_
        # print("final value = ", val)


    r2_train = np.mean(training_r2_list)

    if test_size == 1:
        r2_test = r2_score(actual_list, prediction_list)
    else:
        r2_test = np.mean(test_r2_list)

    # rmse = mean_squared_error(actual_list, prediction_list, squared=False)
    # mape = mean_absolute_percentage_error(actual_list, prediction_list)

    print("Performance on training: ", r2_train)
    print("Performance on test:", r2_test)

    if plot:
        # print("R2 metrics=", r2, "MAPE", mape)
        plt.plot(X.index[:], y[:], color="blue")
        plt.plot(X_train.index, model.predict(X_train), color="red", linestyle="dashed")
        plt.plot(dates_list, prediction_list, color="red")
        plt.plot(dates_list, actual_list, color="green")

        plt.show()

    return r2_train, r2_test


def single_model_regression_single_split(X, y, y_original, model, pca=False, pca_components=10, poly=False, poly_degree=2,
                            plot=False,
                            n_splits=None, test_size=1):
    if poly:
        X = apply_poly(X, poly_degree)
    if pca:
        X = apply_pca(X, pca_components)

    test_size = 20

    X_train, X_test = train_test_split(X, test_size=test_size, shuffle=False)
    y_train, y_test = train_test_split(y, test_size=test_size, shuffle=False)
    y_original_train, y_original_test = train_test_split(y_original, test_size=test_size, shuffle=False)

    model.fit(X_train, y_train)

    y_fit = model.predict(X_train)
    y_fit = y_fit.cumsum()
    y_fit += y_original_train[0]

    y_hat = model.predict(X_test)
    y_hat = y_hat.cumsum()
    y_hat += y_original_train[-1]

    if plot:

        plt.plot(y_original.index, y_original, color="blue")  # train + test

        plt.plot(X_train.index, y_fit, color="red")  # fit
        plt.plot(X_test.index, y_hat, color="red", linestyle="dashed")  # prediction

        r2_train = r2_score(y_original_train[1:], y_fit)
        r2_test = r2_score(y_original_test, y_hat)

        print("R2 metrics=", r2_train, r2_test)

        plt.show()

    return r2_train, r2_test

def multiple_targets_cross_validation():

    warnings.filterwarnings(action='ignore', category=ConvergenceWarning, module='sklearn')

    mongo_cv = get_collection_documents("cross_validation")

    # _, _, original_df = get_readytomodel_df([1, 6, 12], stationary=False)
    feature_columns, target_columns, df = get_readytomodel_df([1, 6, 12], stationary=True, mode="diff")

    for t_col in target_columns:

        search = "grid"
        scoring = {"mae":"neg_mean_absolute_error", "rmse":"neg_root_mean_squared_error",
                   "ev":"explained_variance","r2":"r2"}

        # 20 splits w/ test size 6 or 10 splits w/ test size 12 for testing on the last 10 years
        n_splits = 10
        test_size = 12
        _id = None
        verbose = 1 # 1 to suppress, 2 to print executions

        # CHECK IN MONGO IF COMBINATION ALREADY DONE
        found = False
        for m_cv in mongo_cv:
            target_mongo = m_cv["target_column"]
            search_mongo = m_cv["search"]
            scoring_mongo = m_cv["scoring_method"]
            split_mongo = m_cv["n_split"]
            test_mongo = m_cv["test_size"]
            if t_col == target_mongo and search == search_mongo and scoring == scoring_mongo and n_splits == split_mongo and test_size == test_mongo:
                found = True
            if found:
                break
        if found:
            print("Found:", t_col)
            continue

        cv_start_time = time.time()

        # We use all data - 1 year for training + CV, and keep the last year for test
        X, y = split_features_target(df, feature_columns, t_col, holdout_years=1)

        cv_df = cv_optimization(X, y, search=search, scoring=scoring, n_splits=n_splits, test_size=test_size,
                             verbose=verbose, refit="mae")

        print(f"CV time for {t_col}:", time.time() - cv_start_time)

        params = cv_df["params"].tolist()
        for p in params:
            for k in p:
                if k in ["model", "reduce_dim"]:
                    p[k] = str(p[k])

        d_ = {
            "params": params,
            "target_column": t_col,
            "search": search,
            "scoring_method": scoring,
            "n_split": n_splits,
            "test_size": test_size
        }

        for s_ in scoring:
            d_[f"mean_test_{s_}"] = cv_df[f"mean_test_{s_}"].tolist()
            d_[f"rank_test_{s_}"] = cv_df[f"rank_test_{s_}"].tolist()

        if _id is not None:
            d_["_id"] = _id

        insert_document("cross_validation", d_)

    # df.to_csv("searchCV.csv")

def analyze_cv_results():
    mongo_cv = get_collection_documents("cross_validation")

    df = None

    for document in mongo_cv:
        list = []
        target = document["target_column"]
        for idx, model in enumerate(document["params"]):
            rank1 = document["rank_test_mae"][idx]
            rank2 = document["rank_test_rmse"][idx]
            rank3 = document["rank_test_ev"][idx]
            rank4 = document["rank_test_r2"][idx]
            model["model"] = model["model"].split("(")[0]
            list.append({"model":json.dumps(model), f"{target}_rank1":rank1, f"{target}_rank2":rank2, f"{target}_rank3":rank3,
                         f"{target}_rank4":rank4})
        document_df = pd.DataFrame(list, columns=["model", f"{target}_rank1", f"{target}_rank2", f"{target}_rank3",
                                                  f"{target}_rank4"]).set_index("model")
        if df is None:
            df = document_df
        else:
            df = pd.merge(df, document_df, left_index=True, right_index=True, how="outer")
            print(df.shape)
