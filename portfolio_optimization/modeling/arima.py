import warnings

from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import pandas as pd

import matplotlib.pyplot as plt
import pmdarima as pm

from portfolio_optimization.data_lake.mongodb import insert_document, get_collection_documents, get_collection
from portfolio_optimization.db import get_df_from_table
from portfolio_optimization.eda import get_indicator_name
from portfolio_optimization.helper import forecast_accuracy
from portfolio_optimization.modeling.modeling import get_readytomodel_df


def find_d_arima(series, threshold=0.05):

    d = 0
    tmp = series.copy()

    while(d < 10):
        p_value = adfuller(tmp)[1]
        if p_value < threshold:
            return d
        else:
            # print(d, p_value)
            tmp = tmp.diff().dropna()
            d += 1

    return d

def cv_arima():

    # col = get_collection("cv_arima")
    # col.delete_many({})
    # return

    warnings.filterwarnings("ignore")

    feature_columns, target_columns, df = get_readytomodel_df([0], stationary=False)
    df = df.asfreq("MS")
    name_df = get_df_from_table("indicator_name").set_index("id")

    exo_df = df[feature_columns]

    plot = True
    auto = True

    for t in target_columns:

        mongo_doc = []
        title = get_indicator_name(name_df, t)
        print(f"{title} - {t}")

        y = df[[t]]
        y[t] = pd.to_numeric(y[t])
        y = y.dropna()

        # "arima", "sarima", "sarimax"
        for model in ["arima", "sarima", "sarimax"]:

            test, pred = train_arima(y, auto=auto, model_type=model, exo=exo_df, plot=plot)
            acc = forecast_accuracy(pred, test)

            doc = {
                "target": {"id": t, "name": title},
                "auto": auto,
                "model": model,
                "accuracy": acc
            }
            mongo_doc.append(doc)
            print(doc)

        doc = {"data": mongo_doc}
        insert_document("cv_arima", doc)

def analyze_cv_arima():

    doc = get_collection_documents("cv_arima")
    df = []
    for d in doc:
        for data in d["data"]:
            target_id = data["target"]["id"]
            target_name = data["target"]["name"]
            auto = data["auto"]
            p = data["p"]
            q = data["q"]
            train_r2 = data["train_r2"]
            test_r2 = data["test_r2"]
            df.append([target_id, target_name, auto, p, q, train_r2, test_r2])

    df = pd.DataFrame(df, columns=["target_id","target_name","auto","p","q","train_r2","test_r2"])
    print(df)

    df["train_r2"] = pd.to_numeric(df["train_r2"])
    df["test_r2"] = pd.to_numeric(df["test_r2"])
    df = df.groupby(["auto","p","q"])[["train_r2","test_r2"]].mean().reset_index()
    df = df.sort_values(by="test_r2", ascending=False)
    print(df)

# https://www.kaggle.com/code/prashant111/arima-model-for-time-series-forecasting/notebook
def train_arima(y, test_size = 20, auto=True, model_type="arima", exo=None, plot=False):

    train, test = train_test_split(y, test_size=test_size, shuffle=False)

    if model_type == "arima":

        if auto:
            model = pm.auto_arima(train, start_p=1, start_q=1,
                                  test='adf',  # use adftest to find optimal 'd'
                                  max_p=3, max_q=3,  # maximum p and q
                                  m=1,  # frequency of series
                                  d=None,  # let model determine 'd'
                                  seasonal=False,  # No Seasonality
                                  start_P=0,
                                  D=0,
                                  trace=plot,
                                  error_action='ignore',
                                  suppress_warnings=True,
                                  stepwise=True)
            fc = model.predict(n_periods=len(test))
        else:
            model = ARIMA(train, order=(1, 1, 1), trend=[0, 1], freq="MS").fit()
            fc = model.forecast(len(test))

    # order=(p,d,q) #seasonal_order=(P,D,Q,m)
    elif model_type == "sarima":

        if auto:
            model = pm.auto_arima(train, start_p=1, start_q=1,
                                   test='adf',
                                   max_p=3, max_q=3, m=12,
                                   start_P=0, seasonal=True,
                                   d=None, D=1, trace=plot,
                                   error_action='ignore',
                                   suppress_warnings=True,
                                   stepwise=True)
            fc = model.predict(n_periods=len(test))
        else:
            model = ARIMA(train, order=(1, 0, 2), seasonal_order=(1,1,2,12), trend=[0,1], freq="MS").fit()
            fc = model.forecast(len(test))

    elif model_type == "sarimax":

        if exo is not None:
            train_exo, test_exo = train_test_split(exo, test_size=test_size, shuffle=False)

        else:
            # multiplicative seasonal component
            result_mul = seasonal_decompose(train[-36:],  # 3 years
                                            model='multiplicative',
                                            extrapolate_trend='freq')

            seasonal_index = result_mul.seasonal[-12:].to_frame()
            seasonal_index['month'] = pd.to_datetime(seasonal_index.index).month

            train_ex = train.copy()
            train_ex['month'] = train_ex.index.month
            test_ex = test.copy()
            test_ex['month'] = test_ex.index.month

            old_idx = train_ex.index
            train_ex = pd.merge(train_ex, seasonal_index, how='left', on='month')
            train_ex.columns = ['value', 'month', 'seasonal_index']
            train_ex.index = old_idx

            old_idx = test_ex.index
            test_ex = pd.merge(test_ex, seasonal_index, how='left', on='month')
            test_ex.columns = ['value', 'month', 'seasonal_index']
            test_ex.index = old_idx

            train = train_ex[["value"]]
            train_exo = train_ex[["seasonal_index"]]
            test = test_ex[["value"]]
            test_exo = test_ex[["seasonal_index"]]

        train_exo = train_exo.dropna()
        train, train_exo = train.align(train_exo, join="inner", axis=0)

        if auto:
            model = pm.auto_arima(train, X=train_exo,
                                    start_p=1, start_q=1,
                                    test='adf',
                                    max_p=3, max_q=3, m=12,
                                    start_P=0, seasonal=True,
                                    d=None, D=1, trace=plot,
                                    error_action='ignore',
                                    suppress_warnings=True,
                                    stepwise=True)
            fc = model.predict(n_periods=len(test), X=test_exo)
        else:
            model = ARIMA(train, exog=train_exo, order=(1, 0, 2),
                          seasonal_order=(1,1,2,12), trend=[0,1], freq="MS").fit()
            fc = model.forecast(len(test), exog=test_exo)


    if plot:
        print(model.summary())

        plt.figure(figsize=(12, 5), dpi=100)
        plt.plot(train, label='training')
        plt.plot(test, label='actual')
        plt.plot(model.fittedvalues(), label="fit")
        plt.plot(fc, label='forecast')
        plt.legend(loc='upper left', fontsize=8)

        plt.title("Forecast")
        plt.show()

    return test[test.columns[0]], fc

