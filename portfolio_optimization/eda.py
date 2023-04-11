import traceback

import pandas as pd
import matplotlib.pyplot as plt
from portfolio_optimization.data_lake.mongodb import upsert_document
import numpy as np

from portfolio_optimization.helper import get_dfs, get_feature_target_shifted, convert_to_stationary


def get_indicator_name(df, column):
    period = ""
    if "p" in column:
        period = column.split("p")[1]
        column = column.split("p")[0]
    id = int(column.replace('target', '').replace('feature', ''))
    return df.loc[id, "indicator"] + period


def remove_outliers(df, col, t_min, t_max):
    df.loc[df[col] < t_min, col] = np.nan
    df.loc[df[col] > t_max, col] = np.nan
    return df


def count_outliers(df, col, t_min, t_max):
    c_min = df[df[col] < t_min][col].count()
    c_max = df[df[col] > t_max][col].count()
    return c_min, c_max


def single_indicator_plot():
    df, name_df = get_dfs()

    # print(df.head())
    # print(name_df.head())

    # print(df.head())
    # print(df.dtypes)
    # print(df.columns)
    for col in df.columns:
        title = get_indicator_name(name_df, col)
        # if col == 'feature121':
        #     df['feature121b'] = df[col].rolling(12).mean()
        d = df[col].describe()
        # print(d)
        iqr = (d['75%'] - d['25%']) * 1.5
        t_min = d['25%'] - iqr
        t_max = d['75%'] + iqr
        c_min, c_max = count_outliers(df,col, t_min, t_max)

        # if (round(c_min/d['count']*100,2) <= 5 and round(c_max/d['count']*100,2) <= 5):
        #     continue
        print(f" {col} ---- {title}  COUNT:{d['count']} ----------> BELOW: {round(c_min/d['count']*100,2)}%, ABOVE: {round(c_max/d['count']*100,2)}%")

        # df = remove_outliers(df, col, t_min, t_max)
        d = df[col].describe()
        print(d)

        fig = plt.figure(figsize=(17,6))
        h = fig.add_subplot(131)
        h = df[col].plot(kind='line', title=title, grid=True)
        h = fig.add_subplot(132)
        h = df[col].plot(kind='hist', title=title, grid=True)
        h.axvline(x=d['mean'], color='r', linestyle='--', lw=2 )
        h.axvline(x=d['50%'], color='g', linestyle='--', lw=2 )
        h.axvline(x=d['mean']-d['std'], color='b', linestyle='--', lw=1 )
        h.axvline(x=d['mean']+d['std'], color='b', linestyle='--', lw=1 )
        h.axvline(x=d['mean']-2*d['std'], color='b', linestyle='--', lw=2 )
        h.axvline(x=d['mean']+2*d['std'], color='b', linestyle='--', lw=2 )
        box = fig.add_subplot(133)
        box = df[col].plot(kind='box', title=title, grid=True)

        plt.show()


def calculate_feature_pair_corr(threshold=0.95):

    odf, name_df = get_dfs(remove_corr=False)

    # print(df.head())
    # print(df.tail())

    odf = convert_to_stationary(odf)

    # print(df.head())
    # print(df.tail())

    feature_columns, target_columns, odf = get_feature_target_shifted(odf, [1, 6, 12])
    print("# FEATURES:", len(feature_columns))

    corr_df = odf.corr()
    corr_df = corr_df[target_columns]

    results = {"_id": f"feature_correlation_{threshold}", "data": []}
    for col1 in odf.columns:

        if 'target' in col1:
            continue

        title1 = get_indicator_name(name_df, col1)
        id1 = int(col1.replace('feature',''))

        for col2 in odf.columns:

            if 'target' in col2:
                continue
            if col1 == col2:
                continue

            id2 = int(col2.replace('feature',''))
            if id2 < id1:
                continue

            title2 = get_indicator_name(name_df, col2)

            try:
                df = odf[[col1, col2]].dropna()

                corr = df[col1].corr(df[col2])

                if corr < -threshold or corr > threshold:
                    print(f"{col1} - {col2}: {corr * 100}%")

                    c_f, m_f = int(df[col1].count()), corr_df.loc[col1, :].abs().mean()
                    c_t, m_t = int(df[col2].count()), corr_df.loc[col2, :].abs().mean()

                    d = {"col_1": col1, "col_2": col2, "corr": corr, "c_1": c_f, "c_2": c_t, "corr_1": m_f, "corr_2": m_t}
                    results['data'].append(d)

                    # plt.scatter(df[col1], df[col2])
                    # plt.plot(x, pred, color="green")
                    # plt.axvline(x=0, ymin=-1, ymax=1, linestyle="dashed", color="gray")
                    # plt.axhline(y=0, xmin=-1, xmax=1, linestyle="dashed", color="gray")
                    # plt.xlabel(f_title)
                    # plt.ylabel(t_title)
                    # plt.show()

            except:
                traceback.print_exc()
                print(f"{col1} : {title1}")
                print(f"{col2} : {title2}")
                print(df[col2].describe())
                return

    upsert_document('feature_selection', results)