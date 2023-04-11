import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, SGDRegressor, BayesianRidge
from sklearn.tree import DecisionTreeRegressor

from portfolio_optimization.data_lake.mongodb import get_collection, \
    get_feature_selection_collection, upsert_document
from portfolio_optimization.db import insert_df_into_table, execute_db_commands

from sklearn.feature_selection import SelectKBest, f_regression, r_regression, mutual_info_regression, RFE
from numpy.core.arrayprint import set_printoptions

from portfolio_optimization.eda import get_indicator_name
from portfolio_optimization.helper import get_dfs, get_feature_target_shifted, \
    get_indicator_df_from_postgresql, standardize_data, convert_to_stationary


def extract_features(model, fitted_model, features_df, df):
    features = fitted_model.transform(features_df)
    # # summarize selected features
    mask = model.get_support()  # list of booleans
    new_features = []  # The list of your K best features

    for bool_val, feature in zip(mask, features_df.columns):
        if bool_val:
            new_features.append(feature)
    print(features.shape, )

    feature_df = pd.DataFrame(features, columns=new_features, index=df.index)

    print(feature_df.head())
    print(df[new_features].head())
    print(feature_df.tail())
    print(df[new_features].tail())
    return df

def univariate(df, feature_columns, target_columns, threshold):

    score_df = pd.DataFrame([[x] for x in feature_columns], columns=["feature"])

    score_functions = {"f_regr": f_regression, "r_regr": r_regression, "m_regr": mutual_info_regression}

    for score_f in score_functions:
        kbest_model = SelectKBest(score_func=score_functions[score_f], k='all')
        for t_col in target_columns:
            r_df = df[feature_columns + [t_col]]
            r_df = r_df.dropna()
            X = r_df[feature_columns]
            Y = r_df[t_col]
            fit = kbest_model.fit(X, Y)

            # summarize scores
            set_printoptions(precision=3)
            print("SCORES", t_col, score_f)

            score_df[f"{t_col}_{score_f}"] = fit.scores_
            # score_df = score_df.set_index("feature")
            # return

            # print(len(fit.scores_))
            # for i in range(len(fit.scores_)):
            #     print(X.columns[i], "\t", round(fit.scores_[i], 2), "\t", round(fit.pvalues_[i],4))
            # return

            # print("PVALUES")
            # print(fit.pvalues_)
            # print(X.shape)


    score_df = score_df.set_index("feature")
    # print(score_df)
    # print(score_df.apply(pd.DataFrame.describe, axis=1))

    for col in score_df.columns:
        if "r_regr" in col:
            score_df[col] = abs(score_df[col])
        score_df[col] = score_df[col].rank()

    # print("\n\n\n\n")
    # print(score_df)
    describe_df = score_df.apply(pd.DataFrame.describe, axis=1)

    # print(describe_df)
    original_len = len(describe_df)
    remove_df = describe_df[describe_df["25%"] >= threshold]
    # print(original_len, "rimossi:", len(remove_df))

    features_to_remove = remove_df.index
    # print(features_to_remove)
    # print(type(features_to_remove))

    features_to_remove = list(features_to_remove)
    # print(features_to_remove)
    # print(type(features_to_remove))

    return features_to_remove

def rfe(df, feature_columns, target_columns, threshold):

    models = {"lin_regr":LinearRegression, "tree_regr":DecisionTreeRegressor, #"boost_regr":GradientBoostingRegressor,
              "sgd_regr": SGDRegressor, "ridge_regr": BayesianRidge}

    score_df = pd.DataFrame([[x] for x in feature_columns], columns=["feature"])

    for m in models:
        for t_col in target_columns:
            r_df = df[feature_columns + [t_col]]
            r_df = r_df.dropna()
            X = r_df[feature_columns]
            Y = r_df[t_col]

            model = models[m]()
            rfe = RFE(model, n_features_to_select=1)
            fit = rfe.fit(X, Y)
            # print("Feature Ranking: %s" % fit.ranking_)
            print("SCORES", t_col, m)
            score_df[f"{t_col}_{m}"] = fit.ranking_

    score_df = score_df.set_index("feature")
    # print(score_df)

    describe_df = score_df.apply(pd.DataFrame.describe, axis=1)
    # print(describe_df)

    remove_df = describe_df[describe_df["25%"] >= threshold]
    features_to_remove = remove_df.index
    features_to_remove = list(features_to_remove)
    # print(features_to_remove)
    return features_to_remove

def feature_selection(threshold=None, shift_periods=[1, 6, 12]):

    df, name_df = get_dfs()
    df = convert_to_stationary(df)
    df = standardize_data(df)

    feature_columns, target_columns, df = get_feature_target_shifted(df, shift_periods)

    if threshold is None:
        threshold = int(len(feature_columns)/3)

    print("Initial # of features: ", len(feature_columns))

    univariate_remove = univariate(df, feature_columns, target_columns, threshold=threshold)
    print("Remove from univariate:", len(univariate_remove))

    rfe_remove = rfe(df, feature_columns, target_columns, threshold=threshold)
    print("Remove from RFE:", len(rfe_remove))

    result = list(set(univariate_remove + rfe_remove))
    print("Total remove:", len(result))

    d_ = {"_id":"feature_to_remove_selection","data":result}
    upsert_document("feature_selection", d_)


def load_pivot(indicators_to_keep=None):

    # This already filters for "feature_to_remove_corr"
    remove_corr = True
    where = None
    feature_selection_to_remove = []
    if indicators_to_keep is not None:
        where = f"WHERE name in ({str(indicators_to_keep).replace('[','').replace(']','')});"
        remove_corr = False
    else:
        feature_selection_to_remove = \
        get_feature_selection_collection().find({"_id": "feature_to_remove_selection"}).next()["data"]

    df = get_indicator_df_from_postgresql(where=where, remove_corr=remove_corr)
    df = df[~df["column_name"].isin(feature_selection_to_remove)]

    # pivot data
    df["value"] = pd.to_numeric(df["value"])
    df = df.pivot_table(index="date", columns="column_name", values="value", aggfunc="sum").reset_index()

    # drop pivot materialized view if exists
    drop_statement = f"DROP TABLE IF EXISTS pivot"
    create_statement = "CREATE TABLE pivot (date date"
    # alter_statement = "ALTER TABLE pivot OWNER TO easymap_us"
    for col in df.columns:
        if col == "date":
            continue
        create_statement += "," + col + " numeric"
    create_statement += ")"


    execute_db_commands([drop_statement, create_statement])
    insert_df_into_table(df, "pivot")


def correlated_features_dogfight(plot=False):

    data = get_collection('feature_selection').find({'_id': "feature_correlation"}).next()
    corr_df = pd.DataFrame(data['data'], columns=data['data'][0].keys())

    to_remove = []

    for i, r in corr_df.iterrows():

        col_1 = r['col_1']
        col_2 = r['col_2']

        if col_1 in to_remove or col_2 in to_remove:
            continue

        count_1 = r['c_1']
        count_2 = r['c_2']
        corr_1 = r['corr_1']
        corr_2 = r['corr_2']

        removed = False
        if count_1 >= count_2 and corr_1 >= corr_2:
            to_remove.append(col_2)
            print(f"REMOVE {col_2}")
            removed = True
        elif count_2 >= count_1 and corr_2 >= corr_1:
            to_remove.append(col_1)
            print(f"REMOVE {col_1}")
            removed = True

        if removed and plot:

            print(r)
            odf, name_df = get_dfs(remove_corr=False)
            title_1 = get_indicator_name(name_df, col_1)
            title_2 = get_indicator_name(name_df, col_2)
            fig, ax = plt.subplots()
            df = odf[[col_1, col_2]].dropna()
            ax.plot(df.index, df[col_1], label=f"{col_1} - {title_1}")
            ax.set_ylabel(f"{col_1} - {title_1}")
            ax2 = ax.twinx()
            ax2.plot(df.index, df[col_2], label=f"{col_2} - {title_2}", color='orange')
            ax2.set_ylabel(f"{col_2} - {title_2}")
            lines = ax.get_lines() + ax2.get_lines()
            ax.legend(lines, [l.get_label() for l in lines], loc='upper center')
            ax.set_ylim(ymin=min(df[col_1]))
            ax2.set_ylim(ymin=min(df[col_2]))
            plt.show()

    print(f"TO REMOVE: {len(to_remove)}")
    print(to_remove)

    upsert_document("feature_selection", {'_id': "feature_to_remove_corr", "data": to_remove})

