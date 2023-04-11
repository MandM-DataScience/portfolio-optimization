import json
import numpy as np
import pandas as pd

from portfolio_optimization.data_lake.mongodb import get_collection, upload_dataset, get_datasets_collection
from portfolio_optimization.helper import oecd_time_to_datetime

def save_id_and_title_from_oecd():
    oecd_datasets = get_collection('oecd_datasets')
    r = oecd_datasets.aggregate([{"$group":{"_id":{"id":"$metadata.dataset_id", "title":"$metadata.title", "url":"$metadata.url",}}}])
    with open("oecd_ids.csv", "w", encoding="utf-8") as f:
        f.write(f"id;title;url\n")
        for rr in r:
            el = rr['_id']
            f.write(f"{el['id']};{el['title']};{el['url']}\n")


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



def get_dimensions():
    oecd_dataset = get_collection('oecd_datasets')
    r = oecd_dataset.aggregate([
        {"$unwind": {"path": "$dataset.structure.dimensions.observation"}},
        {"$group": {"_id": {"id":"$dataset.structure.dimensions.observation.id", "name": "$dataset.structure.dimensions.observation.name"}}}
    ])
    dimensions = [f"{el['_id']['id']};{el['_id']['name']}" for el in r]

    for d in dimensions:
        print(d)

def get_time_periods():
    oecd_dataset = get_collection('oecd_datasets')
    r = oecd_dataset.aggregate([
        {"$unwind": {"path": "$dataset.structure.dimensions.observation"}},
        {"$match": {"dataset.structure.dimensions.observation.id": "TIME_PERIOD"}},
        {"$project": {
            "metadata.dataset_id": 1,
            "dataset.structure.dimensions.observation": 1
        }}
    ])

    for el in r:
        print(el)
        return


def set_columns():
    oecd_dataset = get_collection('oecd_datasets')
    r = oecd_dataset.find()
    # with open("dataset_columns2.json", "r", encoding="utf-8") as f:
    #     s = json.loads(f.read())
    s = []
    print(f"LEN: {len(s)}")
    for doc in r:
        dataset_id = doc['metadata']['dataset_id']
        if dataset_id != 'PT8':
            continue
        if dataset_id not in s:
            print(f"{dataset_id}")
            print()
            df = get_dataframe(doc)
            print()
            confirmed = False
            while not confirmed:
                cols = input("Which columns to select?\n\n")
                cols_name = [df.columns[int(i)] for i in cols.split('.')]
                print(cols_name)
                confirmed = int(input("Confirm 0=no, 1=yes ?\n\n"))
                s[dataset_id] = cols_name
                with open("dataset_columns2.json", "w", encoding="utf-8") as f:
                    f.write(json.dumps(s))
    print("END")


def get_unique_values():
    location_column_ids = ['METRO_ID', 'REGIONS', 'COU', 'REG_ID', 'REPORTER', 'LOCATION', 'COUNTRY', 'PART']
    time_column_ids = ['FREQUENCY']
    value_column_name = 'Value'
    columns_to_exclude = []
    columns_to_exclude.extend(location_column_ids)
    columns_to_exclude.extend(time_column_ids)
    columns_to_exclude.append(value_column_name)
    oecd_dataset = get_collection('oecd_datasets')
    r = oecd_dataset.find({"metadata.dataset_id": "DP_LIVE"})

    result = []

    i = 1

    for doc in r:

        df = get_dataframe(doc,'id')
        dataset_id = doc['metadata']['dataset_id']
        print(i, dataset_id)
        i+=1

        # print(dataset_id)
        # print(df.columns)
        valid_columns = []
        if 'TIME_PERIOD' not in df.columns:
            continue
        for c in df.columns:
            if c not in columns_to_exclude:
                valid_columns.append(c)
        # print(valid_columns)
        valid = df[valid_columns]
        df = valid.drop_duplicates()

        try:
            df["TIME_PERIOD"] = df.apply(lambda row: oecd_time_to_datetime(row, "TIME_PERIOD"), axis=1)
        except:
            continue

        valid_columns.remove("TIME_PERIOD")
        df = df.groupby(valid_columns).agg({"TIME_PERIOD": [np.min, np.max, "count"]})
        # print(df)
        df = df['TIME_PERIOD'].reset_index()
        # print(df)

        df["Valid"] = df.apply(lambda row: concat_valid_columns(row, valid_columns), axis=1)
        df = df.drop(valid_columns, axis=1)
        df["Valid_ID"] = ' | '.join(valid_columns)
        df["dataset_id"] = dataset_id

        # print(df)
        result.append(df)
        # for i, r in valid.iterrows():
        #     # print(i, ' | '.join(list(r)))

    df = pd.concat(result)
    df.to_csv("prova.csv", sep=";", index=False)


def concat_valid_columns(row, valid_columns):
    return ' | '.join(list(row[valid_columns]))


def create_selected_oecd():
    df = pd.read_csv("oecd_selected.csv")
    # df = df[df["dataset_id"] == 'DP_LIVE']
    # df = df[df["remove"] != 'x']
    df = df[["INDICATOR", "SUBJECT", "MEASURE"]]
    df = df.reset_index(drop=True)
    data = {"_id": "shortlist_oecd", "dataset_id": "DP_LIVE"}
    values = []
    for i, r in df.iterrows():
        values.append(dict(r))
    data['values'] = values
    upload_dataset(data)


# DATA CLEANING #
def create_selected_df_csv():
    oecd_collection = get_datasets_collection()
    r = oecd_collection.find({"_id":"shortlist_oecd"})
    shortlist = r.next()
    print(shortlist)
    df = create_dataframe("DP_LIVE")
    time_column = "TIME_PERIOD"
    df[time_column] = df.apply(lambda row: oecd_time_to_datetime(row, time_column), axis=1)
    print(len(df))
    filter_condition = None
    for v in shortlist['values']:
        if filter_condition is None:
            filter_condition = ((df["INDICATOR"] == v["INDICATOR"]) & (df["SUBJECT"] == v["SUBJECT"]) & (df["MEASURE"] == v["MEASURE"]))
        else:
            filter_condition |= ((df["INDICATOR"] == v["INDICATOR"]) & (df["SUBJECT"] == v["SUBJECT"]) & (df["MEASURE"] == v["MEASURE"]))
    print(filter_condition)
    df = df[filter_condition]
    df.to_csv("selected_df.csv")


def create_interpolated_df():
    oecd_collection = get_datasets_collection()
    r = oecd_collection.find({"_id": "shortlist_oecd"})
    shortlist = r.next()
    # print(shortlist)
    df = create_dataframe("DP_LIVE")
    time_column = "TIME_PERIOD"
    df[time_column] = df.apply(lambda row: oecd_time_to_datetime(row, time_column), axis=1)
    # print(len(df))
    results = []
    for v in shortlist['values']:
        temp_df = df[(df["INDICATOR"] == v["INDICATOR"]) & (df["SUBJECT"] == v["SUBJECT"]) & (df["MEASURE"] == v["MEASURE"])]
        # print(f'{v["INDICATOR"]} - {v["SUBJECT"]} - {v["MEASURE"]}  -------> {temp_df["FREQUENCY"].unique()}')
        int_df = interpolate_data_oecd(temp_df)
        # print(f"{len(temp_df)} - {len(int_df)}")
        results.append(int_df)
    df = pd.concat(results)
    # print(len(df))
    return df


def convert_percentage(row):
    percentage_keys = ["%", "percentage"]
    for el in percentage_keys:
        if el in row["MEASURE"].lower():
            return row["Value"]/100
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


def clean_oecd_data():
    df = create_interpolated_df()
    print(df)

    # check null
    df.dropna()
    print(len(df))

    # convert values type for measure %
    df["Value"] = df.apply(lambda row: convert_percentage(row), axis=1)
    print(df)
