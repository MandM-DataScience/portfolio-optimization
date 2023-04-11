import warnings

import matplotlib
from matplotlib.offsetbox import AnchoredText
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, QuantileRegressor, LassoLars
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.graphics.tsaplots import plot_pacf
import matplotlib.pyplot as plt

from portfolio_optimization.data_lake.mongodb import insert_document, get_collection_documents, get_collection
from portfolio_optimization.db import get_df_from_table
import pandas as pd
import numpy as np
import seaborn as sns
from portfolio_optimization.eda import get_indicator_name
from portfolio_optimization.helper import convert_to_stationary
from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier

from portfolio_optimization.modeling.modeling import split_features_target, get_readytomodel_df

# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc(
    "figure",
    autolayout=True,
    figsize=(11, 4),
    titlesize=18,
    titleweight='bold',
)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
)

target_lags = {
    'target254': 1,
    'target256': 2,
    'target259': 1,
    'target260': 1,
    'target263': 1,
    'target265': 1,
    'target266': 1,
    'target267': 1,
    'target268': 2,
    'target55': 11,
    'target71': 3,
    'target82': 1,
}
target_seasonal = {
    'target254': 3,
    'target256': 6,
    'target259': 2,
    'target260': 2,
    'target263': 2,
    'target265': 4,
    'target266': 2,
    'target267': 2,
    'target268': 5,
    'target55': 2,
    'target71': 3,
    'target82': 2,
}
target_trend = {
    'target254': 2,
    'target256': 2,
    'target259': 3,
    'target260': 3,
    'target263': 3,
    'target265': 5,
    'target266': 4,
    'target267': 2,
    'target268': 6,
    'target55': 5,
    'target71': 3,
    'target82': 1,
}

def lagplot(x, y=None, lag=1, standardize=False, ax=None, **kwargs):
    from matplotlib.offsetbox import AnchoredText
    x_ = x.shift(lag)
    if standardize:
        x_ = (x_ - x_.mean()) / x_.std()
    if y is not None:
        y_ = (y - y.mean()) / y.std() if standardize else y
    else:
        y_ = x
    corr = y_.corr(x_)
    if ax is None:
        fig, ax = plt.subplots()
    scatter_kws = dict(
        alpha=0.75,
        s=3,
    )
    line_kws = dict(color='C3', )
    ax = sns.regplot(x=x_,
                     y=y_,
                     scatter_kws=scatter_kws,
                     line_kws=line_kws,
                     lowess=True,
                     ax=ax,
                     **kwargs)
    at = AnchoredText(
        f"{corr:.2f}",
        prop=dict(size="large"),
        frameon=True,
        loc="upper left",
    )
    at.patch.set_boxstyle("square, pad=0.0")
    ax.add_artist(at)
    ax.set(title=f"Lag {lag}", xlabel=x_.name, ylabel=y_.name)
    return ax


def plot_lags(x, y=None, lags=6, nrows=1, lagplot_kwargs={}, **kwargs):
    import math
    kwargs.setdefault('nrows', nrows)
    kwargs.setdefault('ncols', math.ceil(lags / nrows))
    kwargs.setdefault('figsize', (kwargs['ncols'] * 2, nrows * 2 + 0.5))
    fig, axs = plt.subplots(sharex=True, sharey=True, squeeze=False, **kwargs)
    for ax, k in zip(fig.get_axes(), range(kwargs['nrows'] * kwargs['ncols'])):
        if k + 1 <= lags:
            ax = lagplot(x, y, lag=k + 1, ax=ax, **lagplot_kwargs)
            ax.set_title(f"Lag {k + 1}", fontdict=dict(fontsize=14))
            ax.set(xlabel="", ylabel="")
        else:
            ax.axis('off')
    plt.setp(axs[-1, :], xlabel=x.name)
    plt.setp(axs[:, 0], ylabel=y.name if y is not None else x.name)
    fig.tight_layout(w_pad=0.1, h_pad=0.1)
    return fig


def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)


def plot_lags_and_autocorrelation(target_column):
    pivot = get_df_from_table("pivot")
    name_df = get_df_from_table("indicator_name").set_index("id")
    title = get_indicator_name(name_df, target_column)
    y = pivot.copy()
    y = y.set_index(pd.DatetimeIndex(y['date'])).loc[:, target_column]
    y = y.apply(pd.to_numeric).dropna()
    y = y.to_period('M')
    plt.get_current_fig_manager().window.geometry("+100+100")
    y.plot(title=f"{target_column} -- {title}", **plot_params)
    _ = plot_lags(y, lags=12, nrows=2)
    move_figure(_, 100, 800)
    _ = plot_pacf(y, lags=12)
    move_figure(_, 1800, 100)
    plt.show()


def plot_seasonality(t_col):
    df = get_df_from_table("pivot")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values(by="date").set_index("date")
    df = df.apply(pd.to_numeric)
    df = df[[t_col]]
    df[t_col] = pd.to_numeric(df[t_col])
    df = df.dropna()

    df = df.asfreq("MS")
    # X = pivot.copy().set_index(pd.DatetimeIndex(pivot['date']))[column].dropna()
    # # X = X.apply(pd.to_numeric).dropna().to_period('M')
    # # days within a week
    # X["day"] = X.index.dayofweek  # the x-axis (freq)
    # X["week"] = X.index.week  # the seasonal period (period)
    # # days within a year
    # X["dayofyear"] = X.index.dayofyear
    # X["year"] = X.index.year
    # ig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 6))
    # seasonal_plot(X, y=column, period="week", freq="day", ax=ax0)
    # seasonal_plot(X, y=column, period="year", freq="dayofyear", ax=ax1)
    # plot_periodogram(X[column])

    X = df.copy()
    # months within a year
    X["year"] = X.index.year
    X["month"] = X.index.month

    # for y in X["year"].unique():
    #     year_data = X[X["year"] == y]
    #     plt.plot(year_data["month"], year_data[t_col], label=y)
    # plt.legend()

    from scipy.signal import periodogram
    fs = pd.Timedelta("1Y") / pd.Timedelta("1W")
    frequencies, spectrum = periodogram(
        df[t_col].dropna(),
        fs=fs,
        detrend="linear",
        window="boxcar",
        scaling='spectrum',
    )
    _, ax = plt.subplots()
    ax.step(frequencies, spectrum, color="purple")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12])
    ax.set_xticklabels(
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)"
        ],
        rotation=30,
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title(t_col)

    plt.show()


def make_lags(df, lags):
    res = df.copy()
    old_cols = df.columns
    for i in range(1, lags + 1):
        for col in old_cols:
            res[f"{col}_lag_{i}"] = res[col].shift(i)
    res = res.drop(old_cols, axis=1)
    return res


def make_multistep_target(df, steps):
    res = df.copy()
    old_cols = df.columns
    for i in range(steps):
        for col in old_cols:
            res[f"{col}_step_{i+1}"] = res[col].shift(-i)
    res = res.drop(old_cols, axis=1)
    return res


def fit_trend(y, idx_train, idx_test, trend_order=2, seasonal_order=2, base_fit=None, base_pred=None, ax=None, model=None):

    # Create trend features
    fourier = CalendarFourier(freq="A", order=seasonal_order)
    dp = DeterministicProcess(
        index=y.index,  # dates from the training data
        order=trend_order,  # trend
        additional_terms=[fourier],
        drop=True,  # drop terms to avoid collinearity
    )
    X = dp.in_sample()  # features for the training data
    X_train, X_test = X.iloc[idx_train, :], X.iloc[idx_test, :]
    y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]

    # Fit trend model
    if model is None:
        model = LinearRegression()

    y_residual = y_train
    if base_fit is not None:
        y_residual = y_train - base_fit
    model.fit(X_train, y_residual.values.ravel())

    # Make predictions
    y_trend_fit = pd.DataFrame(
        model.predict(X_train),
        index=y_train.index,
        columns=y_train.columns,
    )
    y_trend_pred = pd.DataFrame(
        model.predict(X_test),
        index=y_test.index,
        columns=y_test.columns,
    )

    # Add bases
    if base_fit is not None:
        y_trend_fit += base_fit
    if base_pred is not None:
        y_trend_pred += base_pred

    # Plot
    if ax is not None:
        ax0 = y_train.plot(color='0.25', sharex=True, ax=ax, label='y_train')
        ax0 = y_test.plot(color='0.25', sharex=True, ax=ax0, label='y_test')
        ax0 = y_trend_fit.plot(color='C0', sharex=True, ax=ax0, label='y_fit')
        ax0 = y_trend_pred.plot(color='C3', sharex=True, ax=ax0, label='y_pred')
        ax0.legend(['y_train', 'y_test','y_trend_fit','y_trend_pred'])

    return y_trend_fit, y_trend_pred


def fit_cycle(y, idx_train, idx_test, lags, base_fit=None, base_pred=None, ax=None, model=None, other_features=None):

    X = make_lags(y, lags=lags)

    if other_features is not None:
        X = pd.concat([X, make_lags(other_features, lags=1)], axis=1)

    X = X.fillna(0.0)
    X_train, X_test = X.iloc[idx_train, :], X.iloc[idx_test, :]
    y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]

    # Fit seasonal model on residual
    if model is None:
        model = LinearRegression()

    y_residual = y_train
    if base_fit is not None:
        y_residual = y_train - base_fit

    # print(X_train)
    # print(y_residual)
    model.fit(X_train, y_residual.values.ravel())

    # Make predictions
    y_cycle_fit = pd.DataFrame(
        model.predict(X_train),
        index=y_train.index,
        columns=y_train.columns,
    )
    y_cycle_pred = pd.DataFrame(
        model.predict(X_test),
        index=y_test.index,
        columns=y_test.columns,
    )

    # Add bases
    if base_fit is not None:
        y_cycle_fit += base_fit
    if base_pred is not None:
        y_cycle_pred += base_pred

    # Plot
    if ax is not None:
        ax1 = y_train.plot(color='0.25', sharex=True, ax=ax, label='y_train')
        ax1 = y_test.plot(color='0.25', sharex=True, ax=ax1, label='y_test')
        ax1 = y_cycle_fit.plot(color='C0', sharex=True, ax=ax1, label='y_fit')
        ax1 = y_cycle_pred.plot(color='C3', sharex=True, ax=ax1, label='y_pred')
        ax1.legend(['y_train', 'y_test','y_cycle_fit','y_cycle_pred'])

    return y_train, y_test, y_cycle_fit, y_cycle_pred

# Compute Fourier features to the 4th order (8 new features) for a
# series y with daily observations and annual seasonality:
#
# fourier_features(y, freq=365.25, order=4)
# annotations: https://stackoverflow.com/a/49238256/5769929
def seasonal_plot(X, y, period, freq, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    palette = sns.color_palette("husl", n_colors=X[period].nunique(),)
    ax = sns.lineplot(
        x=freq,
        y=y,
        hue=period,
        data=X,
        ci=False,
        ax=ax,
        palette=palette,
        legend=False,
    )
    ax.set_title(f"Seasonal Plot ({period}/{freq})")
    for line, name in zip(ax.lines, X[period].unique()):
        y_ = line.get_ydata()[-1]
        ax.annotate(
            name,
            xy=(1, y_),
            xytext=(6, 0),
            color=line.get_color(),
            xycoords=ax.get_yaxis_transform(),
            textcoords="offset points",
            size=14,
            va="center",
        )
    return ax


def plot_periodogram(ts, detrend='linear', ax=None):
    from scipy.signal import periodogram
    fs = pd.Timedelta("1Y") / pd.Timedelta("1D")
    freqencies, spectrum = periodogram(
        ts,
        fs=fs,
        detrend=detrend,
        window="boxcar",
        scaling='spectrum',
    )
    if ax is None:
        _, ax = plt.subplots()
    ax.step(freqencies, spectrum, color="purple")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels(
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)",
            "Biweekly (26)",
            "Weekly (52)",
            "Semiweekly (104)",
        ],
        rotation=30,
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    return ax


def plot_trend(y, title):

    # print(y)

    fig, axs = plt.subplots(10, 1, figsize=(11, 6))
    _ = plt.suptitle(title)

    for trend_order in range(10):

        # print("trend order:", trend_order)

        dp = DeterministicProcess(
            index=y.index,  # dates from the training data
            order=trend_order + 1,  # trend
            drop=True,  # drop terms to avoid collinearity
        )
        X = dp.in_sample()  # features for the training data

        # print(X)

        model = LinearRegression()
        model.fit(X, y)

        # Make predictions
        y_trend_fit = pd.DataFrame(
            model.predict(X),
            index=y.index,
            columns=y.columns,
        )

        # plt.plot(X.index, y, color="blue")
        # plt.plot(X.index, y_trend_fit, color="red")
        # plt.show()

        y.plot(color='0.25', sharex=True, ax=axs[trend_order], label='y')
        y_trend_fit.plot(color='C0', sharex=True, ax=axs[trend_order], label='y_fit')

        at = AnchoredText(
            f"{r2_score(y_trend_fit, y):.2f}",
            prop=dict(size="large"),
            frameon=True,
            loc="lower right",
        )
        at.patch.set_boxstyle("square, pad=0.0")
        axs[trend_order].add_artist(at)

    plt.show()


def performance_hybrid_model(df, model_trend, model_cycle, plot, target_column, step_trend, trend_order,
                             add_features, features, title):

    y = df[[target_column]].dropna()

    if plot:
        # fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(11, 6))
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 6))
        _ = plt.suptitle(title)
    else:
        ax0, ax1, ax2 = None, None, None
    if not add_features:
        features = None

    # idx_train, idx_test = train_test_split(y.index, test_size=test_size, shuffle=False)

    train_r2, y_test_list, y_cycle_pred_list = [], [], []
    tscv = TimeSeriesSplit(n_splits=2, test_size=36)
    for idx_train, idx_test in tscv.split(y):

        # Trend + Season
        if step_trend:
            y_trend_fit, y_trend_pred = fit_trend(y, idx_train, idx_test, ax=ax0, trend_order=trend_order + 1,
                                                  seasonal_order=target_seasonal[target_column], model=model_trend)
        else:
            y_trend_fit, y_trend_pred = None, None

        y_train, y_test, y_cycle_fit, y_cycle_pred = fit_cycle(y, idx_train, idx_test, lags=target_lags[target_column],
                                                               base_fit=y_trend_fit, base_pred=y_trend_pred,
                                                               ax=ax1, model=model_cycle, other_features=features)

        # Forecast
    # from sklearn.multioutput import MultiOutputRegressor
    # y_train, y_test, y_forecast_fit, y_forecast_pred = forecasting(y, idx_train, idx_test, lags=target_lags[t], steps=6,
    #                                               base_fit=y_cycle_fit, base_pred=y_cycle_pred,
    #                                               ax=ax2, model=MultiOutputRegressor(GradientBoostingRegressor()))


    # train_rmse = mean_squared_error(y_train, y_forecast_fit, squared=False)
    # test_rmse = mean_squared_error(y_test, y_forecast_pred, squared=False)

        y_test_list.append(y_test)
        y_cycle_pred_list.append(y_cycle_pred)
        train_r2.append(r2_score(y_train, y_cycle_fit))

    test_r2 = r2_score(pd.concat(y_test_list), pd.concat(y_cycle_pred_list))
    train_r2 = np.mean(train_r2)

    # print((f"Train RMSE: {train_rmse:.2f}\n" f"Test RMSE: {test_rmse:.2f}"))

    # if plot:
    #     plt.show()

    return train_r2, np.mean(test_r2), y_train, y_test, y_cycle_fit, y_cycle_pred


def cv_hybrid_models():

    warnings.filterwarnings("ignore")

    """
    feature101 -> EMVOVERALLEMV
    feature105 -> HOUST
    feature154 -> RECPROUSM156N
    feature177 -> FEDFUNDS
    feature191 -> FRBKCLMCILA
    feature42 -> Inflation (CPI) | Total | Annual growth rate (%)
    feature48 -> Inflation (CPI) | Total | 2015=100
    feature89 -> TRESEGUSM052N
    feature98 -> AMTMTI
    """
    features = ["feature101", "feature105", "feature154", "feature177", "feature191", "feature42", "feature48",
                "feature89", "feature98"]
    df = get_df_from_table("pivot")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values(by="date").set_index("date")
    df = df.apply(pd.to_numeric)
    df = df.asfreq("MS")
    features = df[features].dropna()
    name_df = get_df_from_table("indicator_name").set_index("id")
    target_columns = [c for c in df.columns if 'target' in c]

    plot = False
    test_size = 3 * 12
    model_trend = [LinearRegression(), Ridge(), Lasso()]
    model_cycle = [LinearRegression(), GradientBoostingRegressor()]

    for t in target_columns:
        mongo_doc = []
        title = get_indicator_name(name_df, t)
        print(f"{title} - {t}")
        for step_trend in [True, False]:
            for trend_order in range(5):
                if not step_trend and trend_order > 0:
                    continue
                for add_features in [True, False]:
                    for m_c in model_cycle:
                        for idx, m_t in enumerate(model_trend):

                            if not step_trend and idx > 0:
                                continue

                            train_r2, test_r2 = performance_hybrid_model(df, m_t, m_c, plot, t, step_trend,
                                                     trend_order,
                                                     add_features, features, title)

                            doc = {
                                "target": {"id":t, "name":title},
                                "trend":str(m_t),
                                "cycle":str(m_c),
                                "step_trend":step_trend,
                                "trend_order": trend_order+1,
                                "exo_features": add_features,
                                "train_r2": round(train_r2,2),
                                "test_r2": round(test_r2,2)
                            }
                            # print(doc)
                            mongo_doc.append(doc)
                    # print()

        doc = {"data": mongo_doc}
        insert_document("cv_hybrid_models", doc)


        # plot_lags_and_autocorrelation(t)
        # plot_seasonality(t)
        # return
        # continue
        # plot_trend(y, t)
        # continue


def analyze_cv_hybrid():

    doc = get_collection_documents("cv_hybrid_models")
    df = []
    for d in doc:
        for data in d["data"]:
            target_id = data["target"]["id"]
            target_name = data["target"]["name"]
            trend = data["trend"]
            cycle = data["cycle"]
            step_trend = data["step_trend"]
            trend_order = data["trend_order"]
            exo_features = data["exo_features"]
            train_r2 = data["train_r2"]
            test_r2 = data["test_r2"]
            df.append([target_id, target_name, trend, cycle, step_trend, trend_order, exo_features, train_r2, test_r2])

    df = pd.DataFrame(df, columns=["target_id","target_name","trend","cycle","step_trend","trend_order","exo_features","train_r2","test_r2"])
    # print(df)

    df["train_r2"] = pd.to_numeric(df["train_r2"])
    df["test_r2"] = pd.to_numeric(df["test_r2"])
    df = df.groupby(["trend","cycle","step_trend","trend_order","exo_features"])[["train_r2","test_r2"]].mean().reset_index()
    df = df.sort_values(by="test_r2", ascending=False)

    print(df)


def forecasting(y, idx_train, idx_test, model=None, lags=4, steps=8, base_fit=None, base_pred=None, ax=None):

    X = make_lags(y, lags=lags).fillna(0.0)
    original_y = y.copy()
    y = make_multistep_target(y, steps=steps).dropna()

    y, X = y.align(X, join='inner', axis=0)

    # print(y.tail(40))

    idx_test = idx_test[:-steps]
    # print("idx_train:", idx_train)
    # print("idx_test:", idx_test)
    X_train, X_test = X.loc[idx_train, :], X.loc[idx_test, :]
    y_train, y_test = y.loc[idx_train], y.loc[idx_test]

    # print(y_train.tail(10))

    # print(X_train.tail(10))
    # print(y_train.tail(10))

    if model is None:
        model = LinearRegression()

    y_residual = y_train
    if base_fit is not None:
        first_steps_rows = base_pred.iloc[:steps-1,:]
        base_fit = pd.concat([base_fit, first_steps_rows])
        base_fit = make_multistep_target(base_fit, steps=steps).dropna()
        y_residual = y_train - base_fit

    # print(base_fit.tail(100))
    # print(y_train.tail(100))
    # print(y_residual.tail(100))
    # print(X_train[X_train.isna().any(axis=1)])
    # print(y_residual[y_residual.isna().any(axis=1)])
    model.fit(X_train, y_residual)

    y_fit = pd.DataFrame(model.predict(X_train), index=X_train.index, columns=y.columns)
    y_pred = pd.DataFrame(model.predict(X_test), index=X_test.index, columns=y.columns)

    # print(y_test.tail(10))
    # print(y_fit.tail(10))
    # print(y_pred.tail(10))

    # Add bases
    if base_fit is not None:

        # print("y fit")
        # print(y_fit.tail(20))
        # print("base fit")
        # print(base_fit.tail(20))
        y_fit += base_fit
        # print("y fit")
        # print(y_fit.tail(20))

    if base_pred is not None:
        # print(base_pred.head(10))
        # print(y_pred.head(10))
        base_pred = make_multistep_target(base_pred, steps=steps).dropna()
        y_pred += base_pred
        # print(y_pred.head(10))

    if ax is not None:
        palette = dict(palette='husl', n_colors=64)

        # print(y_fit)
        # print(y_pred)

        ax1 = original_y.plot(color='0.25', sharex=True, ax=ax, label='y')
        ax1 = plot_multistep(y_fit, ax=ax1, palette_kwargs=palette)
        ax1 = plot_multistep(y_pred, ax=ax1, palette_kwargs=palette)

        ax1.legend(['y', 'y_fit', 'y_pred'])

    return y_train, y_test, y_fit, y_pred


def plot_multistep(y, every=1, ax=None, palette_kwargs=None):
    palette_kwargs_ = dict(palette='husl', n_colors=16, desat=None)
    if palette_kwargs is not None:
        palette_kwargs_.update(palette_kwargs)
    palette = sns.color_palette(**palette_kwargs_)
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_prop_cycle(plt.cycler('color', palette))
    for date, preds in y[::every].iterrows():
        preds.index = pd.period_range(start=date, periods=len(preds), freq="M")
        preds.plot(ax=ax)
    return ax


def test_with_stationarity():
    warnings.filterwarnings("ignore")

    """
    feature101 -> EMVOVERALLEMV
    feature105 -> HOUST
    feature154 -> RECPROUSM156N
    feature177 -> FEDFUNDS
    feature191 -> FRBKCLMCILA
    feature42 -> Inflation (CPI) | Total | Annual growth rate (%)
    feature48 -> Inflation (CPI) | Total | 2015=100
    feature89 -> TRESEGUSM052N
    feature98 -> AMTMTI
    """
    features = ["feature101", "feature105", "feature154", "feature177", "feature191", "feature42", "feature48",
                "feature89", "feature98"]
    df = get_df_from_table("pivot")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values(by="date").set_index("date")
    df = df.apply(pd.to_numeric)
    df = df.asfreq("MS")
    original_df = df.copy()
    order = 1
    df = convert_to_stationary(df, 'log_minus_mean', order)

    # t = 'target266'
    # tdf = df[[t]].dropna()
    # yt_df = invert_transformation(original_df[[t]].dropna(), tdf, order)
    # print(original_df[[t]].dropna().head(5))
    # print(tdf.head(5))
    # print(yt_df.head(10))

    features = df[features].dropna()
    name_df = get_df_from_table("indicator_name").set_index("id")
    target_columns = [c for c in df.columns if 'target' in c]

    plot = True
    for t in target_columns:
        train_r2, test_r2, y_train, y_test, y_cycle_fit, y_cycle_pred = performance_hybrid_model(df, Ridge(), GradientBoostingRegressor(), plot, t, False, 1, False, features, t)
        # print(f"TRAIN {train_r2}, TEST {test_r2}")
        # print(ytrain.tail(20))
        # print(ytest)
        # print(original_df.tail(40))
        # print(y_train.index, y_cycle_fit.index)
        # print(y_train.shape, y_cycle_fit.shape)
        # print(y_test.index, y_cycle_pred.index)
        # print(y_test.shape, y_cycle_pred.shape)
        y_train = y_cycle_fit
        y_test = y_cycle_pred

        yt = pd.concat([y_train, y_test], axis=0)
        # yt = y_train.copy()
        # yt = ytest.copy()
        # print(yt.head(10))
        # yt_df = invert_transformation(original_df[[t]].dropna(), yt, order)
        # yt_df.index = pd.to_datetime(original_df[[t]].dropna().index)
        # print(yt_df.head(40))
        # print(yt_df.tail(40))
        # train_df = invert_transformation(original_df[[t]].loc[ytrain.index], ytrain, order)
        # train_df = train_df[order:]
        # train_df.index = pd.to_datetime(train_df.index)
        #
        # test_df = invert_transformation(original_df[[t]].loc[ytest.index], ytest, order)
        # test_df = test_df[order:]
        # test_df.index = pd.to_datetime(test_df.index)
        ax = original_df[[t]].dropna().plot(color='0.25', sharex=True, label='y_original')
        # yt_df.plot(color='C2', linestyle="--", ax=ax, sharex=True, label='y_train')
        # ax = ytrain.plot(color='C2', sharex=True, ax=ax, linestyle="--", label='y_train')
        yt.loc[y_train.index].plot(color='C2', linestyle="--", ax=ax, sharex=True, label='y_train')
        yt.loc[y_test.index].plot(color='C1', linestyle="--", ax=ax, sharex=True, label='y_test')
        plt.show()


def test_pickle():
    import pandas
    from sklearn import model_selection
    from sklearn.linear_model import LogisticRegression
    import pickle
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    dataframe = pandas.read_csv(url, names=names)
    array = dataframe.values
    X = array[:, 0:8]
    Y = array[:, 8]
    test_size = 0.33
    seed = 7
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
    # Fit the model on training set
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    # save the model to disk
    filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))


def train_model(y, t_col, other_features=None):
    X = get_features_trend(y, t_col)

    model_trend = LinearRegression()
    model_trend.fit(X, y.values.ravel())
    base_fit = pd.DataFrame(
        model_trend.predict(X),
        index=y.index,
        columns=y.columns,
    )
    X = get_features_cycle(y, other_features, t_col)
    y_residual = y - base_fit
    y, X = y.align(X, join='inner', axis=0)

    model_cycle = GradientBoostingRegressor()
    model_cycle.fit(X, y_residual.values.ravel())

    outcome = pd.DataFrame(
        model_cycle.predict(X),
        index=y.index,
        columns=y.columns,
    )
    return model_trend, model_cycle


def get_features_trend(y, t_col):
    if 'p' in t_col:
        t_col = t_col.split('p')[0]

    seasonal_order = target_seasonal[t_col]
    trend_order = target_trend[t_col]

    fourier = CalendarFourier(freq="A", order=seasonal_order)
    dp = DeterministicProcess(
        index=y.index,  # dates from the training data
        order=trend_order,  # trend
        additional_terms=[fourier],
        drop=True,  # drop terms to avoid collinearity
    )
    X = dp.in_sample()
    return X


def get_features_cycle(y, other_features, t_col):
    if 'p' in t_col:
        t_col = t_col.split('p')[0]
    lags = target_lags[t_col]

    X = make_lags(y, lags=lags)
    if other_features is not None:
        X = pd.concat([X, make_lags(other_features, lags=1)], axis=1)

    X = X.fillna(0.0)
    return X
