import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import pandas as pd
import seaborn as sns

betterment_colors = [
    '#79CCFF',  # SHY
    '#ADE7FF',  # TIP
    '#B5FFCB',  # VTI
    '#41F39A',  # IVE
    '#30EC9B',  # IWN
    '#19DB9A',  # IWS
    '#218080',  # EFA
    '#112F42',  # EEM
]


def minkowski_distance(arr_1, arr_2, p):
    """
    An implementation of the metric for the Lebesgue spaces. The Minkowski distance generalizes to many
    well-known metrics for specific choices of p. For example:

    p = 1: Manhattan Distance

    p = 2: Euclidian distance

    p -> infinity: Chebyshev distance

    For a given p, the function will return the distance between the two points at arr_1 and arr_2 in L^p space

    :param arr_1: The location of the first point
    :type arr_1: array-like
    :param arr_2: The location of the second point
    :type arr_2: array-like
    :param p: The paramater specifying which p-norm will be used
    :type p: float
    :return: The distance between arr_1 and arr_2 in L^p space
    """
    if len(arr_1) != len(arr_2):
        raise ValueError
    return sum(abs(arr_1 - arr_2) ** p) ** (1 / p)


def within_tolerance(target, current, drift):
    """

    :param target:
    :type target: array-like
    :param current:
    :type current: array-like
    :param drift:
    :type drift: float
    :return:
    """
    return minkowski_distance(target, current, MINKOWSKI_P) < drift


if __name__ == '__main__':
    sns.set_style('whitegrid')
    cm.register_cmap('betterment', cmap=colors.ListedColormap(betterment_colors))

    STARTING_CASH = 100000
    MAX_DRIFT = 0.05
    MINKOWSKI_P = 2
    RELATIVE_PATH = 'C:/Users/Tiger/PycharmProjects/BettermentCaseStudy'

    returns_df = pd.read_csv(
        filepath_or_buffer='{0}/portfolio_returns.csv'.format(RELATIVE_PATH),
        index_col=0,
        parse_dates=True
    )
    tickers = returns_df.columns
    dates = returns_df.index
    target_weights = pd.Series(
        data=[0.25, 0.25, 0.125, 0.125, 0.04, 0.035, 0.125, 0.05],
        index=tickers
    )

    # NON REBALANCED PORTFOLIO
    cumulative_returns_df = (returns_df + 1).cumprod()
    buy_and_hold_df = (cumulative_returns_df * STARTING_CASH).mul(target_weights, axis=1)
    buy_and_hold_df_alloc = (buy_and_hold_df.div(buy_and_hold_df.sum(axis=1), axis=0))
    buy_and_hold_df_returns = buy_and_hold_df.sum(axis=1).pct_change(1)

    # REBALANCED PORTFOLIO
    rebalance_df = buy_and_hold_df.copy()
    rebalance_df_alloc = buy_and_hold_df_alloc.copy()

    for date in dates[1:]:
        if within_tolerance(target_weights.values, rebalance_df_alloc.shift(1).loc[date].values, MAX_DRIFT):
            rebalance_df.loc[date] = rebalance_df.shift(1).loc[date].mul(1 + returns_df.loc[date])
        else:
            rebalance_df.loc[date] = target_weights * sum(rebalance_df.shift(1).loc[date].mul(1 + returns_df.loc[date]))
        rebalance_df_alloc.loc[date] = (rebalance_df.loc[date].div(rebalance_df.loc[date].sum()))

    rebalance_df_returns = rebalance_df.sum(axis=1).pct_change(1)

    # SAVE DATASETS TO FILE
    buy_and_hold_df.sum(axis=1).to_csv('values_buy_and_hold')
    rebalance_df.sum(axis=1).to_csv('values_rebalance')
    buy_and_hold_df_alloc.to_csv('allocation_buy_and_hold')
    rebalance_df_alloc.to_csv('allocation_rebalance')
    buy_and_hold_df_returns.to_csv('returns_buy_and_hold')
    rebalance_df_returns.to_csv('returns_rebalance')

    # MAKE AND SAVE PLOTS

    # DAILY RETURNS
    buy_and_hold_ax = plt.subplot2grid((2, 2), (0, 0))
    rebalance_ax = plt.subplot2grid((2, 2), (1, 0))
    hist_ax = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
    buy_and_hold_df_returns.plot(
        ax=buy_and_hold_ax,
        figsize=(12, 6),
        title='Daily returns of buy-and-hold portfolio',
    )
    rebalance_df_returns.plot(
        ax=rebalance_ax,
        title='Daily returns of rebalanced portfolio',
    )
    buy_and_hold_df_returns.plot(
        ax=hist_ax,
        kind='hist',
        bins=100,
        alpha=0.5,
        title='Histogram of returns',
        label='Buy and Hold'
    )
    rebalance_df_returns.plot(
        ax=hist_ax,
        kind='hist',
        bins=100,
        alpha=0.5,
        label='Rebalance'
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig('daily_returns.png')
    plt.gcf().clear()
    plt.close()

    # ALLOCATIONS
    fig_alloc, axes_alloc = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)

    buy_and_hold_df_alloc.plot(
        ax=axes_alloc[0],
        figsize=(12, 6),
        title='Weight of portfolio assets of buy-and-hold portfolio',
        kind='area',
        legend=False,
        ylim=(0, 1),
        colormap=cm.get_cmap('betterment')
    )
    rebalance_df_alloc.plot(
        ax=axes_alloc[1],
        title='Weight of portfolio assets of rebalanced portfolio',
        kind='area',
        legend=False,
        colormap=cm.get_cmap('betterment')
    )
    axes_alloc[0].set_xlim(dates[0], dates[-1])
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.2), ncol=8)
    plt.savefig('asset_allocations.png')
    plt.gcf().clear()
    plt.close()

    # PORTFOLIO VALUES

    fig_values, axes_values = plt.subplots()

    buy_and_hold_df.sum(axis=1).plot(
        ax=axes_values,
        figsize=(12, 6),
        title='Portfolio values',
        label='Value of buy-and-hold portfolio',
    )
    rebalance_df.sum(axis=1).plot(
        ax=axes_values,
        label='Value of rebalanced portfolio'
    )
    axes_values.set_xlim(dates[0], dates[-1])
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)
    plt.savefig('values.png')
    plt.gcf().clear()
    plt.close()
