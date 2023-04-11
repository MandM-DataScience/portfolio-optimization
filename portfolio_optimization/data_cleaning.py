import json

from portfolio_optimization.data_lake.mongodb import get_fred_datasets_collection, upsert_document, get_datasets_collection, \
    get_collection, get_yf_target_datasets_collection, get_investing_target_datasets_collection, insert_document
from portfolio_optimization.db import insert_df_into_table, execute_db_commands, get_df_from_table
from portfolio_optimization.helper import oecd_time_to_datetime
import pandas as pd

def get_fred_datasets_titles():
    datasets = get_fred_datasets_collection()
    datasets = datasets.find({})

    result = []

    i = 1
    for d in datasets:
        print(i)
        id = d["_id"]
        title = d["pippo"][0]["title"]
        notes = d["pippo"][0]["notes"].replace('\n', '').replace('\r', '') if "notes" in d["pippo"][0] else ""
        result.append({"id": id, "title": title, "notes": notes})
        i += 1

    df = pd.DataFrame(result, columns=["id", "title", "notes"])
    # df.to_csv("fred_datasets_titles")


def create_dataframe_fred(dataset_id):
    datasets = get_fred_datasets_collection()
    datasets = datasets.find({"_id":dataset_id})

    result = []
    units = None
    frequency = None

    for d in datasets:
        # print(d)

        units = d["metadata"]["units"]
        frequency = d["metadata"]["frequency"]
        for o in d["observations"]:
            result.append({key: o[key] for key in ["date","value"]})

    df = pd.DataFrame(result, columns=["date","value"])

    # print(df)

    # Convert Types
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])

    # Adjust % values
    if units is not None and "Percent" in units:
        df["value"] /= 100

    # Interpolate values
    df = interpolate_date_fred(df, frequency)

    # Remove rows with any null present
    df = df.dropna()
    df["feature"] = dataset_id

    return df


def etl_fred(dataset_ids=None):
    print("ETL FRED")
    datasets = get_fred_datasets_collection()
    query = {}
    if dataset_ids is not None:
        query = {"_id": {"$in": dataset_ids}}
    datasets = datasets.find(query)

    result = []

    for d in datasets:
        _id = d['_id']
        df = create_dataframe_fred(_id)
        result.append(df)

    df = pd.concat(result)
    df["source"] = "FRED"

    df = df.rename(columns={"feature":"name"})
    df = df[["source", "name", "date", "value"]]

    # print(df)
    # print(df.dtypes)

    insert_df_into_table(df, 'indicator')


def interpolate_date_fred(df, frequency):

    if "Monthly" in frequency:
        return df
    if "Weekly" in frequency:
        df = df.set_index(["date"]).resample('MS').mean().reset_index()
        return df
    if "Quarterly" in frequency or "Annual" in frequency:
        df = df.set_index(["date"]).resample('MS').ffill().reset_index()
        return df


def get_dataframe(dataset, column_key='name', value_key='name'):
    data = dataset['dataset']
    values = []
    obs = data['dataSets'][0]['observations']
    for e in obs:
        i = [int(x) for x in e.split(":")]
        i.append(obs[e][0])
        values.append(i)

    obs_cols = data['structure']['dimensions']['observation']
    replacement = {}
    cols = []
    for c in obs_cols:
        n = c[column_key]
        replacement[n] = {i: x[value_key] for i, x in enumerate(c['values'])}
        cols.append(n)
    # for r in replacement:
    #     print(r, replacement[r])
    columns = cols.copy()
    columns.append("Value")
    df = pd.DataFrame(values, columns=columns)
    for c in cols:
        df[c] = df[c].replace(replacement[c])

    # df = df[[df.columns[1], df.columns[2]]]
    # df = df.sort_values('Subject')
    # print(df.drop_duplicates())
    # print(columns)
    # print(dataset['metadata'])
    return df


def create_dataframe(dataset_id):
    oecd_datasets = get_collection('oecd_datasets')
    dataset = oecd_datasets.find_one({'metadata.dataset_id':dataset_id})
    if dataset is not None:
        return get_dataframe(dataset, "id")


def concat_column_values(row, columns):
    return ' | '.join(list(row[columns]))


def create_interpolated_df(indicators=None):
    oecd_collection = get_datasets_collection()
    if indicators is None:
        r = oecd_collection.find({"_id": "shortlist_oecd"})
        shortlist = r.next()['values']
    else:
        shortlist = indicators
    # print(shortlist)
    df = create_dataframe("DP_LIVE")
    time_column = "TIME_PERIOD"
    df[time_column] = df.apply(lambda row: oecd_time_to_datetime(row, time_column), axis=1)
    # print(len(df))
    results = []
    for v in shortlist:
        temp_df = df[
            (df["INDICATOR"] == v["INDICATOR"]) & (df["SUBJECT"] == v["SUBJECT"]) & (df["MEASURE"] == v["MEASURE"])]
        # print(f'{v["INDICATOR"]} - {v["SUBJECT"]} - {v["MEASURE"]}  -------> {temp_df["FREQUENCY"].unique()}')
        int_df = interpolate_data_oecd(temp_df)
        # print(f"{len(temp_df)} - {len(int_df)}")
        results.append(int_df)
    df = pd.concat(results)
    return df


def convert_percentage(row):
    percentage_keys = ["%", "percentage"]
    for el in percentage_keys:
        if el in row["MEASURE"].lower():
            return row["Value"] / 100
    return row["Value"]


def interpolate_data_oecd(df):
    frequencies = df["FREQUENCY"].unique()
    if "Monthly" in frequencies:
        df = df[df["FREQUENCY"] == "Monthly"]
    elif "Quarterly" in frequencies:
        df = df[df["FREQUENCY"] == "Quarterly"]
        df = df.set_index(["TIME_PERIOD"]).resample('MS').ffill().reset_index()
    elif "Annual" in frequencies:
        df = df[df["FREQUENCY"] == "Annual"]
        df = df.set_index(["TIME_PERIOD"]).resample('MS').ffill().reset_index()

    return df


def get_clean_oecd_df(indicators=None):
    df = create_interpolated_df(indicators)
    # print(df)

    # check null
    df = df.dropna()
    # print(len(df))

    # convert values type for measure %
    df["value"] = df.apply(lambda row: convert_percentage(row), axis=1)

    # create single column
    to_join_columns = ["INDICATOR", "SUBJECT", "MEASURE"]
    df['name'] = df.apply(lambda row: concat_column_values(row, to_join_columns), axis=1)
    to_drop_columns = ["INDICATOR", "SUBJECT", "MEASURE", "LOCATION", "FREQUENCY"]
    df = df.drop(to_drop_columns, axis=1)

    df = df.reset_index(drop=True)

    return df


def etl_oecd(indicators=None):
    print("ETL OECD")
    df = get_clean_oecd_df(indicators)
    df['source'] = 'OECD'
    df = df.rename(columns={"TIME_PERIOD": "date"})
    df = df[["source", "name", "date", "value"]]
    insert_df_into_table(df,'indicator')


def etl_yf(dataset_ids=None):
    print("ETL YAHOO FINANCE")
    datasets = get_yf_target_datasets_collection()
    query = {}
    if dataset_ids is not None:
        query = {"_id": {"$in": dataset_ids}}
    datasets = datasets.find(query)

    list_of_df = []
    for d in datasets:
        result = []
        for dd in d["data"]:
            result.append({"date":dd["Date"], "value":dd["Close"], "name":dd["ticker"]})

        list_of_df.append(pd.DataFrame(result, columns=["date","value","name"]))

    df = pd.concat(list_of_df)

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"],unit='ms')

    df = df.dropna()

    df["source"] = "yahoo_finance"
    df = df[["source", "name", "date", "value"]]
    df = df.set_index(["date"]).groupby(["name", "source"]).resample('MS') \
        .mean().reset_index()
    insert_df_into_table(df, 'indicator')


def etl_investing(dataset_ids=None):
    datasets = get_investing_target_datasets_collection()
    query = {}
    if dataset_ids is not None:
        query = {"_id": {"$in": dataset_ids}}
    datasets = datasets.find(query)

    list_of_df = []
    for d in datasets:
        name = d["_id"]
        result = []

        for dd in d["data"]:
            # print(dd)

            result.append({"date":dd["date"], "value":dd["close"], "name":name})
        # return

        list_of_df.append(pd.DataFrame(result, columns=["date","value","name"]))

    df = pd.concat(list_of_df)

    df["value"] = pd.to_numeric(df["value"].str.replace(",",""), errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna()

    df["source"] = "investing"
    df = df[["source", "name", "date", "value"]]
    df = df.set_index(["date"]).groupby(["name", "source"]).resample('MS') \
        .mean().reset_index()
    insert_df_into_table(df, 'indicator')


def import_and_etl_investing_csv():
    path = {"OIL":"data_lake/oil.csv", "GOLD":"data_lake/xauusd.csv", "WHEAT":"data_lake/wheat.csv"}

    list_of_df = []
    for k in path:
        df = pd.read_csv(path[k])
        records = df.to_json(orient="records")
        _js = {}
        _js["data"] = json.loads(records)
        _js["_id"] = k

        upsert_document("investing_target_datasets", _js)

        df["name"] = k
        list_of_df.append(df)

    df = pd.concat(list_of_df)

    df["value"] = pd.to_numeric(df["Price"].astype(str).str.replace(",",""), errors="coerce")
    df["date"] = pd.to_datetime(df["Date"])
    df["source"] = "investing"
    df = df[["source", "name", "date", "value"]]

    df = df.dropna()
    insert_df_into_table(df, 'indicator')