import json
import os
from configparser import ConfigParser

import pandas as pd
import pickle
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import Ridge
import pmdarima as pm

from portfolio_optimization import PORTFOLIO_BASE_DIR
from portfolio_optimization.data_cleaning import etl_yf, etl_investing, etl_fred, etl_oecd
from portfolio_optimization.data_collection import retrieve_series, download_yahoo_finance_price_data, \
    retrieve_series_metadata, get_investing_data, get_oecd_live_dataset
from portfolio_optimization.data_lake.mongodb import upsert_document
from portfolio_optimization.data_lake.s3_sandbox import put_object_to_s3, get_object_from_s3
from portfolio_optimization.feature_selection import load_pivot
from portfolio_optimization.helper import standardize_data
from portfolio_optimization.modeling.modeling import get_readytomodel_df, split_features_target
from portfolio_optimization.modeling import hybrid_models

FRED_IDS = ['EMVOVERALLEMV', 'HOUST', 'RECPROUSM156N', 'FEDFUNDS', 'FRBKCLMCILA', 'TRESEGUSM052N', 'AMTMTI', 'USSTHPI']
YAHOO_FINANCE_IDS = ['^RUT', '^TNX', '^GSPC', '^DJI', '^IXIC']
INVESTING_URLS = {
    'GOLD': 'https://www.investing.com/commodities/gold-historical-data',
    'OIL': 'https://www.investing.com/commodities/crude-oil-historical-data',
    'WHEAT': 'https://www.investing.com/commodities/us-wheat-historical-data',
}

OECD_NAMES = ['Inflation (CPI) | Total | Annual growth rate (%)', 'Inflation (CPI) | Total | 2015=100',
        'Housing prices | Nominal house prices | 2015=100', 'Long-term interest rates | Total | % per annum',
        'Short-term interest rates | Total | % per annum']

FRED_DATASET_COLLECTION_NAME = 'fred_datasets'
YAHOO_FINANCE_COLLECTION_NAME = 'yf_target_datasets'
INVESTING_COLLECTION_NAME = 'investing_target_datasets'
OECD_COLLECTION_NAME = 'oecd_datasets'


def extract():

    parser = ConfigParser()
    _ = parser.read(os.path.join(PORTFOLIO_BASE_DIR, "credentials.cfg"))
    fred_api_key = parser.get("fred", "fred_api_key")

    # Collect FRED data
    for s in FRED_IDS:
        print(s)
        data = retrieve_series(s, fred_api_key)
        metadata = retrieve_series_metadata(s, fred_api_key)
        data['metadata'] = metadata
        data["_id"] = s
        upsert_document(FRED_DATASET_COLLECTION_NAME, data)

    # Collect YAHOO_FINANCE data
    for ticker in YAHOO_FINANCE_IDS:
        print(ticker)
        df = download_yahoo_finance_price_data(ticker)
        records = df.reset_index().to_json(orient="records")
        _js = {}
        _js["data"] = json.loads(records)
        _js["_id"] = ticker
        upsert_document(YAHOO_FINANCE_COLLECTION_NAME, _js)

    # # Collect INVESTING.COM data
    for _id, url in INVESTING_URLS.items():
        print(_id, url)
        df = get_investing_data(url)
        records = df.to_json(orient="records")
        _js = {}
        _js["data"] = json.loads(records)
        _js["_id"] = _id
        upsert_document(INVESTING_COLLECTION_NAME, _js)
    # # Collect OECD LIVE data
    upsert_document(OECD_COLLECTION_NAME, get_oecd_live_dataset())


def transform_and_load():
    etl_fred(dataset_ids=FRED_IDS)
    etl_yf(dataset_ids=YAHOO_FINANCE_IDS)
    etl_investing(dataset_ids=list(INVESTING_URLS.keys()))

    oecd_indicators = []
    for el in OECD_NAMES:
        s = el.split('|')
        oecd_indicators.append({
          "INDICATOR": s[0].strip(),
          "SUBJECT": s[1].strip(),
          "MEASURE": s[2].strip()
        })
    etl_oecd(oecd_indicators)

    indicators_to_keep = FRED_IDS + list(INVESTING_URLS.keys()) + YAHOO_FINANCE_IDS + OECD_NAMES
    print("CREATE PIVOT")
    load_pivot(indicators_to_keep)


def training():

    parser = ConfigParser()
    _ = parser.read(os.path.join(PORTFOLIO_BASE_DIR, "credentials.cfg"))
    BUCKET_ID = parser.get("s3", "BUCKET_ID")

    feature_columns, target_columns, df = get_readytomodel_df([1], stationary=True, mode="diff",
                                                              start_date=None, end_date=None)
    _, _, non_stationary_df = get_readytomodel_df([1], stationary=False,
                                                                                start_date=None, end_date=None)

    for t_col in target_columns:
        X, y = split_features_target(df, feature_columns, t_col, holdout_years=0)
        ridge = Ridge()
        ridge_fitted = ridge.fit(X, y)
        put_object_to_s3(pickle.dumps(ridge_fitted), BUCKET_ID, f"ridge_estimator-{t_col}")

    for t_col in target_columns:
        # We keep 1 year to predict the "future"
        X_ns, y_ns = split_features_target(non_stationary_df, feature_columns, t_col, holdout_years=1)
        arima_fitted = pm.auto_arima(y_ns, X=X_ns, test='adf', m=12, seasonal=True,
                                     error_action='ignore', suppress_warnings=True)
        put_object_to_s3(pickle.dumps(arima_fitted), BUCKET_ID, f"arima_estimator-{t_col}")

    for t_col in target_columns:
        X_ns, y_ns = split_features_target(non_stationary_df, feature_columns, t_col, holdout_years=0)

        model_trend, model_cycle = hybrid_models.train_model(non_stationary_df[[t_col]].dropna(), t_col, other_features=X_ns)
        put_object_to_s3(pickle.dumps(model_trend), BUCKET_ID, f"hybrid_trend_estimator-{t_col}")
        put_object_to_s3(pickle.dumps(model_cycle), BUCKET_ID, f"hybrid_cycle_estimator-{t_col}")


def predict(model_name, target, start_date, end_date):

    """

    :param model_name: model name
    :param target: target name
    :param start_date: start date of the prediction
    :param end_date: end date of the prediction
    :return:
    """

    parser = ConfigParser()
    _ = parser.read(os.path.join(PORTFOLIO_BASE_DIR, "credentials.cfg"))
    BUCKET_ID = parser.get("s3", "BUCKET_ID")

    # target is formatted with targetXXX
    target_column_0 = f"{target}p0m"
    target_column_1 = f"{target}p1m"
    start_date = start_date - relativedelta(months=1)
    _, _, df0 = get_readytomodel_df([0], stationary=False, mode="diff", start_date=None, end_date=None)
    df0 = df0[(df0.index >= start_date) & (df0.index <= end_date)]

    feature_columns, _, df = get_readytomodel_df([1], stationary=True, mode="diff", start_date=None, end_date=None)
    X = standardize_data(df[feature_columns].dropna())
    X = X[(X.index > start_date) & (X.index <= end_date)]

    feature_columns, target_ns, non_stationary_df = get_readytomodel_df([1], stationary=False, mode="diff", start_date=None, end_date=None)
    X_ns = standardize_data(non_stationary_df[feature_columns].dropna())
    X_ns = X_ns[(X_ns.index >= start_date) & (X_ns.index <= end_date)]
    y_ns = non_stationary_df[[target_column_1]].dropna()

    if model_name == 'RIDGE':
        model = pickle.loads(get_object_from_s3(f"ridge_estimator-{target_column_1}", BUCKET_ID))
        outcome = model.predict(X)
        result = outcome.cumsum() + df0[target_column_0][0]
        result_df = pd.DataFrame([[x] for x in result], columns=[target], index=X.index.astype(str))

    if model_name == 'ARIMA':
        model = pickle.loads(get_object_from_s3(f"arima_estimator-{target_column_1}", BUCKET_ID))
        result = model.fittedvalues()

        max_date = result.index[-1]
        months_after = (end_date.year - max_date.year) * 12 + (end_date.month - max_date.month)
        result_df = pd.DataFrame([[x] for x in result], columns=[target], index=result.index)
        if months_after > 0:
            X_ns = X_ns[(X_ns.index > max_date) & (X_ns.index <= end_date)]
            outcome = model.predict(n_periods=months_after, X=X_ns)
            outcome_df = pd.DataFrame([[x] for x in outcome], columns=[target], index=outcome.index)
            result_df = pd.concat([result_df, outcome_df])
        result_df = result_df[(result_df.index > start_date) & (result_df.index <= end_date)]
        result_df.index = result_df.index.astype(str)

    if model_name == 'HYBRID':
        model_trend = pickle.loads(get_object_from_s3(f"hybrid_trend_estimator-{target_column_1}", BUCKET_ID))
        X = hybrid_models.get_features_trend(y_ns, target_column_1)
        X = X[(X.index > start_date) & (X.index <= end_date)]
        trend_outcome = model_trend.predict(X)

        model_cycle = pickle.loads(get_object_from_s3(f"hybrid_cycle_estimator-{target_column_1}", BUCKET_ID))
        trend_outcome = pd.DataFrame(
            trend_outcome,
            index=X.index,
            columns=y_ns.columns,
        )
        y_ns, X_ns = y_ns.align(X_ns, join='inner', axis=0)
        X = hybrid_models.get_features_cycle(y_ns, X_ns, target_column_1)
        cycle_outcome = model_cycle.predict(X)
        cycle_outcome = pd.DataFrame(
            cycle_outcome,
            index=X.index,
            columns=y_ns.columns,
        )

        result_df = trend_outcome + cycle_outcome
        result_df = result_df[(result_df.index > start_date) & (result_df.index <= end_date)]
        result_df.index = result_df.index.astype(str)

    result_df.index.name = 'date'
    result_df = result_df.reset_index()
    return result_df.to_dict(orient='records')


def lambda_handler(event, context):

    """
    event['body'] : {
        "model":["HYBRID" | "RIDGE" | "ARIMA" ],
        "target": targetXXX
        "start_date": YYYY-MM-DD,
        "end_date": YYYY-MM-DD,
    :param event:
    :param context:
    :return:
    """
    error = ""
    try:
        print("EVENT", event)
        if 'body' in event:
            print("READ BODY")
            d = json.loads(event['body'])
        else:
            d = event

        if d is not None:
            print("START PREDICTING ", d)
            r = predict(d["model"], d["target"], start_date=pd.Timestamp(d["start_date"]),
                        end_date=pd.Timestamp(d["end_date"]))
            return {
                'statusCode': 200,
                'body': json.dumps(r)
            }
    except Exception as e:
        import traceback
        error = e
        print(e)
        print()
        traceback.print_exc()

    return {
        'statusCode': 400,
        'body': f"NO EVENT DATA {error}"
    }


def aws_fargate():
    extract()
    transform_and_load()
    training()