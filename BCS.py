import sys
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import pandas as pd
import seaborn as sns
from tqdm import tqdm

betterment_palette = [
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
    cm.register_cmap('betterment', cmap=colors.ListedColormap(betterment_palette))

    STARTING_CASH = 100000
    MAX_DRIFT = 0.05
    MINKOWSKI_P = 5.0
    PATH = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

    returns_df = pd.read_csv(
        filepath_or_buffer=os.path.join(PATH, 'portfolio_returns.csv'),
        index_col=0,
        parse_dates=True
    )
    tickers = returns_df.columns
    dates = returns_df.index
    returns_df.index.name = 'Date'
    returns_df.columns = pd.MultiIndex.from_tuples(list(zip(['daily'] * 8, tickers)))
    target_weights = pd.Series(
        data=[0.25, 0.25, 0.125, 0.125, 0.04, 0.035, 0.125, 0.05],
        index=tickers
    )

    returns_df[zip(['cumulative']*8, tickers)] = (returns_df['daily'] + 1).cumprod()

    # NON REBALANCED PORTFOLIO
    buy_and_hold_df = pd.DataFrame(
        data=(returns_df['cumulative'] * STARTING_CASH).mul(target_weights, axis=1).values,
        index=dates,
        columns=pd.MultiIndex.from_tuples(list(zip(['values'] * 8, tickers)))
    )
    buy_and_hold_df[zip(['allocations']*8, tickers)] = (buy_and_hold_df['values'].div(buy_and_hold_df['values'].sum(axis=1), axis=0))
    buy_and_hold_df['returns'] = (buy_and_hold_df['values'].sum(axis=1)).pct_change(1)



    # REBALANCED PORTFOLIO
    rebalance_df = buy_and_hold_df.copy()

    for date in tqdm(dates[1:]):
        if within_tolerance(target_weights.values, rebalance_df.shift(1).loc[date, 'allocations'].values, MAX_DRIFT):
            rebalance_df.loc[date, 'values'] = \
                rebalance_df.shift(1).loc[date, 'values'].mul(1 + returns_df.loc[date, 'daily']).values
        else:
            rebalance_df.loc[date, 'values'] = \
                (sum(rebalance_df.shift(1).loc[date, 'values'].mul(1 + returns_df.loc[date, 'daily'])) *
                 target_weights).values
        rebalance_df.loc[date, 'allocations'] = \
            (rebalance_df.loc[date, 'values'].div(rebalance_df.loc[date, 'values'].sum())).values

    rebalance_df['returns'] = rebalance_df['values'].sum(axis=1).pct_change(1)

    # SAVE DATASETS TO FILE
    buy_and_hold_df['values'].sum(axis=1).to_csv('values_buy_and_hold.csv')
    rebalance_df['values'].sum(axis=1).to_csv('values_rebalance.csv')
    buy_and_hold_df['allocations'].to_csv('allocation_buy_and_hold.csv')
    rebalance_df['allocations'].to_csv('allocation_rebalance.csv')
    buy_and_hold_df['returns'].to_csv('returns_buy_and_hold.csv')
    rebalance_df['returns'].to_csv('returns_rebalance.csv')

    # MAKE AND SAVE PLOTS

    # DAILY RETURNS
    buy_and_hold_ax = plt.subplot2grid((2, 2), (0, 0))
    rebalance_ax = plt.subplot2grid((2, 2), (1, 0))
    hist_ax = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
    buy_and_hold_df['returns'].plot(
        ax=buy_and_hold_ax,
        figsize=(12, 6),
        title='Daily returns of buy-and-hold portfolio',
    )
    rebalance_df['returns'].plot(
        ax=rebalance_ax,
        title='Daily returns of rebalanced portfolio',
    )
    buy_and_hold_df['returns'].plot(
        ax=hist_ax,
        kind='hist',
        bins=100,
        alpha=0.5,
        title='Histogram of returns',
        label='Buy and Hold'
    )
    rebalance_df['returns'].plot(
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

    buy_and_hold_df['allocations'].plot(
        ax=axes_alloc[0],
        figsize=(12, 6),
        title='Weight of portfolio assets of buy-and-hold portfolio',
        kind='area',
        legend=False,
        ylim=(0, 1),
        colormap=cm.get_cmap('betterment', 8)
    )
    rebalance_df['allocations'].plot(
        ax=axes_alloc[1],
        title='Weight of portfolio assets of rebalanced portfolio',
        kind='area',
        legend=False,
        colormap=cm.get_cmap('betterment')
    )
    axes_alloc[0].set_xlim(dates[0], dates[-1])
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.2), ncol=8)
    plt.savefig('asset_allocations.png', dpi=300)
    plt.gcf().clear()
    plt.close()

    # PORTFOLIO VALUES

    fig_values, axes_values = plt.subplots()

    buy_and_hold_df['values'].sum(axis=1).plot(
        ax=axes_values,
        figsize=(12, 6),
        title='Portfolio values',
        label='Value of buy-and-hold portfolio',
    )
    rebalance_df['values'].sum(axis=1).plot(
        ax=axes_values,
        label='Value of rebalanced portfolio'
    )
    axes_values.set_xlim(dates[0], dates[-1])
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)
    plt.savefig('values.png')
    plt.gcf().clear()
    plt.close()
