# VECTOR AUTOREGRESSION
# https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests

from portfolio_optimization.modeling.arima import find_d_arima
from statsmodels.tsa.api import VAR

from portfolio_optimization.helper import convert_to_non_stationary, forecast_accuracy
from portfolio_optimization.modeling.modeling import get_readytomodel_df
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table
    are the P-Values. P-Values lesser than the significance level (0.05), implies
    the Null Hypothesis that the coefficients of the corresponding past values is
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """

    maxlag = 12

    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i + 1][0][test][1], 4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

def cointegration_test(df, alpha=0.05, summary=True):
    """Perform Johanson's Cointegration Test and Report Summary"""
    out = coint_johansen(df,-1,5)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)

    # Summary
    if summary:
        print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    not_sign_cols = []
    for col, trace, cvt in zip(df.columns, traces, cvts):
        if summary:
            print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)
        if not trace > cvt:
            not_sign_cols.append(col)

    return not_sign_cols

def train_var():

    feature_columns, target_columns, df = get_readytomodel_df([0], stationary=False)
    df = df.asfreq("MS")
    df = df.dropna()

    print(df.shape)  # (123, 8)
    print(df.tail())

    # # Plot
    # fig, axes = plt.subplots(nrows=7, ncols=3, dpi=120, figsize=(10, 6))
    # for i, ax in enumerate(axes.flatten()):
    #     data = df[df.columns[i]]
    #     ax.plot(data, color='red', linewidth=1)
    #     # Decorations
    #     ax.set_title(df.columns[i])
    #     ax.xaxis.set_ticks_position('none')
    #     ax.yaxis.set_ticks_position('none')
    #     ax.spines["top"].set_alpha(0)
    #     ax.tick_params(labelsize=6)
    #
    # plt.tight_layout()
    # plt.show()

    # caus_matrix = grangers_causation_matrix(df, variables=df.columns)
    # print(caus_matrix)

    not_sign_cols = cointegration_test(df, summary=False)

    df = df.drop(not_sign_cols, axis=1)
    print(df.tail())

    test_size = 36
    df_train, df_test = df[0:-test_size], df[-test_size:]

    # Check size
    print(df_train.shape)  # (119, 8)
    print(df_test.shape)  # (4, 8)

    for name, column in df.iteritems():
        d = find_d_arima(column)
        print(name, d)

    df_diff = df_train.copy()
    for i in range(1):
        df_diff = df_diff.diff().dropna()

    model = VAR(df_diff)
    # for i in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
    #     result = model.fit(i)
    #     print('Lag Order =', i)
    #     print('AIC : ', result.aic)
    #     print('BIC : ', result.bic)
    #     print('FPE : ', result.fpe)
    #     print('HQIC: ', result.hqic, '\n')
    #
    # x = model.select_order(maxlags=12)
    # print(x.summary())

    p = 3 # selected order with min AIC
    model_fitted = model.fit(p)
    print(model_fitted.summary())

    # Get the lag order
    lag_order = model_fitted.k_ar
    print(lag_order)  # > 4

    # Input data for forecasting
    forecast_input = df_diff.values[-lag_order:]
    print(forecast_input)

    # Forecast
    fc = model_fitted.forecast(y=forecast_input, steps=test_size)
    df_forecast = pd.DataFrame(fc, index=df.index[-test_size:], columns=df.columns)
    print(df_forecast)

    df_results = convert_to_non_stationary(df_train, df_forecast, order=1)
    print(df_results)

    fig, axes = plt.subplots(nrows=int(len(df.columns) / 2), ncols=2, dpi=150, figsize=(10, 10))
    for i, (col, ax) in enumerate(zip(df.columns, axes.flatten())):
        df_train[col].plot(legend=False, ax=ax, color="blue") # train
        df_results[col].plot(legend=False, ax=ax, color="red").autoscale(axis='x', tight=True) # forecast
        df_test[col][-test_size:].plot(legend=False, ax=ax, color="green") # actual
        ax.set_title(col + ": Forecast vs Actuals")
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.spines["top"].set_alpha(0)
        ax.tick_params(labelsize=6)

    plt.tight_layout()
    plt.show()

    for c in df_results.columns:
        print(c)
        accuracy_prod = forecast_accuracy(df_results[c].values, df_test[c.split("_")[0]])
        for k, v in accuracy_prod.items():
            print(k, ': ', round(v, 4))
        print()