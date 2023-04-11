import math
from datetime import datetime, date
import numpy as np
from dateutil.relativedelta import relativedelta
import pandas as pd
from pandas import Series
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller

from portfolio_optimization.data_lake.mongodb import get_collection, get_feature_selection_collection
from portfolio_optimization.db import get_df_from_table, execute_db_commands, insert_df_into_table
import matplotlib.pyplot as plt


def adjust_date(row, date_header):
    d = row[date_header]
    if d.day != 1:
        d = d.replace(day=1)
        d += relativedelta(months=1)
    return d

def oecd_time_to_datetime(row, date_header):
    if "Q" in row[date_header]:
        return date_from_quarter(row, date_header)
    else:
        return date_from_year(row, date_header)

def date_from_year(row, date_header):
    if "/" in str(row[date_header]):
        return pd.to_datetime(row[date_header])
    if "-" in str(row[date_header]):
        return pd.to_datetime(row[date_header])
    return datetime(int(row[date_header]),1,1)

def date_from_quarter(row, date_header):
    split = row[date_header].split("-")
    if "Q" in split[0]:
        quarter = split[0]
        year = int(split[1])
    else:
        quarter = split[1]
        year = int(split[0])
    if quarter == "Q1":
        return datetime(year,1,1)
    if quarter == "Q2":
        return datetime(year,4,1)
    if quarter == "Q3":
        return datetime(year,7,1)
    if quarter == "Q4":
        return datetime(year,10,1)

def get_last_data_from_df(df):
    df = df.fillna(value=np.nan)
    columns = df.columns
    l = []
    last_data = df.apply(Series.last_valid_index)

    for index, value in last_data.items():
        if value is None or math.isnan(value):
            l.append(None)
        else:
            l.append(df.loc[value, index])

    return pd.DataFrame(np.array([l], dtype=np.float64), columns=columns)

def get_last_data_from_locations_df(df):
    locations = df["location"].unique()

    r = []
    for location in locations:
        df_loc = df[df["location"] == location]
        df_loc = df_loc[df_loc['date'] <= datetime.now().date()]
        df_loc = df_loc.drop(["date", "location"], axis=1)
        last_data = get_last_data_from_df(df_loc)
        last_data["location"] = location
        r.append(last_data)
    df = pd.concat(r)

    return df

def substitute_eur_money_supply(df):

    df2 = df.copy(deep=True)
    df2 = df2[df2["location"] == "EA19"]
    df2 = df2[["date","m1","m3"]]
    df2 = df2.rename(columns={"m1":"m1_EUR", "m3":"m3_EUR"})

    df = pd.merge(df, df2, left_on=["date"], right_on=["date"], how="left")
    df["m1"] = df.apply (lambda row: fix_eur_money_supply(row, "m1_EUR", "m1"), axis=1)
    df["m3"] = df.apply(lambda row: fix_eur_money_supply(row, "m3_EUR", "m3"), axis=1)
    df = df.drop(['m1_EUR', 'm3_EUR'], axis=1)
    return df

def fix_eur_money_supply(row, label_eur, label):
    if row["location"] in ['AUT','BEL','EST','FIN','FRA','DEU','GRC','IRL','ITA','LVA','LTU','LUX','NLD','PRT','SVK','SVN','ESP']:
        return row[label_eur]
    return row[label]

def keep_only_max_date(df, group_cols):

    df = df[df["date"] <= datetime.now().date()]

    max_dates = df.groupby(group_cols)["date"].max().reset_index()
    max_dates = max_dates.rename(columns={"date": "maxDate"})
    df = pd.merge(df, max_dates, left_on=group_cols, right_on=group_cols, how="inner")
    df = df[df["date"] == df["maxDate"]]
    df = df.drop(["maxDate"], axis=1)

    return df

def transform_unavailable_data(row, column, date):
    if row["date"] <= date:
        return row[column]
    return math.nan

def remove_data_not_available_at_the_time(df, created_at):

    # print(df)

    month_columns = ["shprice","emp","ppi","cpi","ltint","m1","m3","bci","cci","stint"]
    quarter_columns = ["gdphrwkdforecast","cpiforecast","realgdpforecast","nomgdpforecast","hhsavforecast",
                       "ltintforecast","stintforecast","pricerent","priceincome","rent","real","nominal"]
    year_columns = ["ggdebt","hhdebt","gdp","gdpltforecast","gdphrwkd","hhexp","fincorp","bankleverage","taxwedge",
                    "avwage","hhsav","hhfa","hhwealth","incomeineq","poverty","povertygap","trustgov","edupubexp",
                    "pisascience","tradegoodserv","ictinvst","gdexprd","ulc"]

    month_last_available_date = created_at - relativedelta(months=2)
    quarter_last_available_date = date(created_at.year, 3 * ((created_at.month - 1) // 3) + 1, 1) \
                                  - relativedelta(months=1)
    if (created_at.month - 1) % 3 == 0:
        quarter_last_available_date -= relativedelta(months=3)
    year_last_available_date = date(created_at.year-1, 12, 1)
    if created_at.month <= 2:
        year_last_available_date -= relativedelta(years=1)

    # print(month_last_available_date)
    # print(quarter_last_available_date)
    # print(year_last_available_date)

    _df = df.copy()

    for col in month_columns:
        if col in _df.columns:
            _df[col] = _df.apply(lambda row: transform_unavailable_data(row, col, month_last_available_date), axis=1)
    for col in quarter_columns:
        if col in _df.columns:
            _df[col] = _df.apply(lambda row: transform_unavailable_data(row, col, quarter_last_available_date), axis=1)
    for col in year_columns:
        if col in _df.columns:
            _df[col] = _df.apply(lambda row: transform_unavailable_data(row, col, year_last_available_date), axis=1)

    # print(_df)

    return _df

def transform_investing_date(row, date_header):
    d = row[date_header]

    if isinstance(d, date):
        return d

    if not "-" in d:
        # print(date)
        return d

    half1 = d.split("-")[0]
    half2 = d.split("-")[1]
    if len(half1) == 2:
        year = half1
        month = half2
    elif len(half1) == 1:
        year = "0"+half1
        month = half2
    else:
        year = half2
        month = half1
    return datetime.strptime(month+" "+year,'%b %y').strftime('%Y-%m-%d')


def get_dfs(remove_outlier=True, remove_corr=True):
    df = get_indicator_df_from_postgresql(f"where date between '{date(1960, 1, 1)}' and '{date(2022, 12, 31)}'",
                                          remove_outlier=remove_outlier, remove_corr=remove_corr)

    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"])

    df = df.pivot_table(index="date", columns="column_name", values="value", aggfunc="sum").reset_index()
    df = df.set_index("date")

    name_df = get_df_from_table("indicator_name")
    name_df = name_df.set_index("id")

    return df, name_df


def get_indicator_df_from_postgresql(where=None, remove_outlier=True, remove_corr=True):

    df = get_df_from_table("indicator", where)
    df = df.drop("id", axis=1)
    id_df = get_df_from_table("indicator_name")
    df = pd.merge(df, id_df, left_on=["name"], right_on=["indicator"], how="inner")
    df = df.drop("indicator", axis=1)

    # print(df.head())
    # print(max(df["date"]))
    if remove_outlier:
        df = df[~df["name"].isin(['TOTBORR', 'BORROW ', 'TOTRESNS', 'NONBORRES ', '^DJUSRE', '^DJCI', 'USCI',
                                  'CL=F', 'GC=F', '^SP500BDT', 'TREASURY', 'DDDM03USA156NWDB', 'M0263AUSM500NNBR',
                                  'M14062USM027NNBR', 'M0264AUSM500NNBR', 'M1490AUSM157SNBR',
                                  'Quarterly GDP | Total | Percentage change', 'Quarterly GDP | Total | Percentage change',
                                  'Q09084USQ507NNBR',
                                  'M0263AUSM500NNBR', 'M0264AUSM500NNBR', 'M1490AUSM157SNBR', 'M14062USM027NNBR',
                                  'M09075USM476NNBR', 'Q09084USQ507NNBR',
                                  'M09086USM156NNBR', 'DDDM03USA156NWDB', 'DDDM01USA156NWDB', 'DDEM01USA156NWDB',
                                  'LABSHPUSA156NRUG', 'RTFPNAUSA632NRUG',
                                  'SIPOVGINIUSA', 'DDDI06USA156NWDB', 'ITNETUSERP2USA',
                                  'Electricity generation | Total | Gigawatt-hours'])]

    df["target_feature"] = "feature"
    df.loc[df["source"].isin(["yahoo_finance", "investing"]), "target_feature"] = "target"
    df.loc[df["name"].isin(['USSTHPI',
                            'Short-term interest rates | Total | % per annum',
                            'Long-term interest rates | Total | % per annum',
                            'Housing prices | Nominal house prices | 2015=100']), "target_feature"] = "target"

    # print(df.head())
    df["column_name"] = df["target_feature"] + df["id"].astype(str)
    # print(df.head())
    # print(max(df["date"]))

    if remove_corr:
        to_remove_corr = get_feature_selection_collection().find({'_id': "feature_to_remove_corr"}).next()['data']
        df = df[~df["column_name"].isin(to_remove_corr)]

    # date, value, column_name + ...

    return df

def shift_targets(df, columns, periods):
    targets = []
    for period in periods:
        for col in columns:
            t = f"{col}p{period}m"
            targets.append(t)
            df[t] = df[col].shift(periods=-period)

    return df, targets

def get_feature_target_shifted(df, shift_periods):
    feature_columns = [col for col in df.columns if 'feature' in col]
    t_columns = [col for col in df.columns if 'target' in col]

    df, target_columns = shift_targets(df, t_columns, shift_periods)
    df = df[[x for x in df.columns if x not in t_columns]]
    return feature_columns, target_columns, df

def standardize_data(df):
    # print(df.head())
    # print(df.tail())
    scaled_features = StandardScaler().fit_transform(df.values)
    scaled_features_df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)
    # print(scaled_features_df.head())
    # print(scaled_features_df.tail())
    return scaled_features_df

def convert_to_stationary(df, mode="diff", n_diff=1):
    if mode == "log_minus_mean":
        df = np.log(df)
        rolling_mean = df.rolling(window=12).mean()
        df = df - rolling_mean
        return df
    elif mode == "log_shift":
        df = np.log(df)
        df = df.diff()
        return df
    elif mode == "diff":
        for i in range(n_diff):
            df = df.diff()
        return df


def convert_to_non_stationary(original_df, stationary_df, order):

    """Revert back the differencing to get the forecast to original scale."""
    tmp = stationary_df.copy()

    columns = original_df.columns
    for col in columns:

        cumsum = tmp[col].cumsum()
        # print(cumsum)

        for i in range(order):

            # print("\nPASSAGGIO",i,"\n")

            # print("diff n = ", order - i - 1)
            diff_df = original_df.copy()
            for i in range(order - i - 1):
                diff_df = diff_df.diff().dropna()

            first_element = diff_df[col].iloc[0]
            # print("First element", first_element)

            tmp = first_element + cumsum
            # print("tmp")
            # print(tmp)

            tmp = pd.concat([pd.Series(first_element), tmp])
            cumsum = tmp.cumsum()
            # print("cumsum")
            # print(cumsum)

            tmp = pd.DataFrame(tmp, columns=[col])
            # print("final_tmp")
            # print(tmp)

    return tmp


def get_stationarity(df, col, plot=True):

    # rolling statistics
    # if rolling mean and/or std move over time (line is not horizontal) data is NOT stationary. NOT Stationary = value does depend on date
    rolling_mean = df.rolling(window=12).mean()
    rolling_std = df.rolling(window=12).std()

    # Dickey–Fuller test
    # if p-value is > 0.05 we can conclude data is NOT stationary. NOT Stationary = value does depend on date
    result = adfuller(df[col])
    # print('ADF Statistic: {}'.format(result[0]))
    print('Adfuller p-value: {}'.format(result[1]))
    # print('Critical Values:')
    # for key, value in result[4].items():
    #     print('\t{}: {}'.format(key, value))

    # rolling statistics plot
    if plot:
        original = plt.plot(df, color='blue', label='Original')
        mean = plt.plot(rolling_mean, color='red', label='Rolling Mean')
        std = plt.plot(rolling_std, color='black', label='Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        plt.show()


def get_indicator_name(df, column):
    period = ""
    if "p" in column:
        period = column.split("p")[1]
        column = column.split("p")[0]
    id = int(column.replace('target', '').replace('feature', ''))
    return df.loc[id, "indicator"] + period


def get_eda_df(remove_outliers=True, remove_correlation=False, remove_selected=False):
    # We get dataFrame from table "indicator"
    df = get_df_from_table("indicator", f"where date between '{date(1960, 1, 1)}' and '{date(2022, 12, 31)}'")
    df = df.drop("id", axis=1)

    # We get dataFrame from table "indicator_name", we do this to convert features and target names to something more simple to handle
    name_df = get_df_from_table("indicator_name")
    df = pd.merge(df, name_df, left_on=["name"], right_on=["indicator"], how="inner")
    df = df.drop("indicator", axis=1)
    name_df = name_df.set_index("id")

    # This new code remove the outliers we identified previously
    if remove_outliers:
        df = df[~df["name"].isin(['TOTBORR', 'BORROW ', 'TOTRESNS', 'NONBORRES ', '^DJUSRE', '^DJCI', 'USCI',
                                  'CL=F', 'GC=F', '^SP500BDT', 'TREASURY', 'DDDM03USA156NWDB', 'M0263AUSM500NNBR',
                                  'M14062USM027NNBR', 'M0264AUSM500NNBR', 'M1490AUSM157SNBR',
                                  'Quarterly GDP | Total | Percentage change',
                                  'Quarterly GDP | Total | Percentage change',
                                  'Q09084USQ507NNBR',
                                  'M0263AUSM500NNBR', 'M0264AUSM500NNBR', 'M1490AUSM157SNBR', 'M14062USM027NNBR',
                                  'M09075USM476NNBR', 'Q09084USQ507NNBR',
                                  'M09086USM156NNBR', 'DDDM03USA156NWDB', 'DDDM01USA156NWDB', 'DDEM01USA156NWDB',
                                  'LABSHPUSA156NRUG', 'RTFPNAUSA632NRUG',
                                  'SIPOVGINIUSA', 'DDDI06USA156NWDB', 'ITNETUSERP2USA',
                                  'Electricity generation | Total | Gigawatt-hours'])]

    # We select which indicators are feature and which are targets based on these name list
    df["target_feature"] = "feature"
    df.loc[df["source"].isin(["yahoo_finance", "investing"]), "target_feature"] = "target"
    df.loc[df["name"].isin(['USSTHPI',
                            'Short-term interest rates | Total | % per annum',
                            'Long-term interest rates | Total | % per annum',
                            'Housing prices | Nominal house prices | 2015=100']), "target_feature"] = "target"
    df["column_name"] = df["target_feature"] + df["id"].astype(str)

    if remove_correlation:
        to_remove_corr = get_feature_selection_collection().find({'_id': "feature_to_remove_correlation"}).next()['data']
        df = df[~df["column_name"].isin(to_remove_corr)]

    if remove_selected:
        to_remove_corr = get_feature_selection_collection().find({'_id': "selected_features"}).next()['data']
        df = df[~df["column_name"].isin(to_remove_corr)]

    # Converting dataFrame values to datetime and numeric
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"])

    # In the end we pivot df to obtain a pivot table where columns are feature and targets and rows are dates.
    df = df.pivot_table(index="date", columns="column_name", values="value", aggfunc="sum").reset_index()
    df = df.set_index("date")

    return df, name_df

def forecast_accuracy(forecast, actual):

    mape = np.mean(np.abs(forecast - actual) / np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)  # ME
    mae = np.mean(np.abs(forecast - actual))  # MAE
    mpe = np.mean((forecast - actual) / actual)  # MPE
    rmse = np.mean((forecast - actual) ** 2) ** .5  # RMSE
    corr = np.corrcoef(forecast, actual)[0, 1]  # corr

    return ({'mape': mape, 'me': me, 'mae': mae,
             'mpe': mpe, 'rmse': rmse, 'corr': corr})

