import os

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from Strategy import Strategy
from betterment_colors import BETTERMENT_BLUE, BETTERMENT_GRAY, BETTERMENT_PALETTE


def minkowski_distance(arr_1, arr_2, p):
    """
    An implementation of the metric for the Lebesgue spaces. The Minkowski distance generalizes to many
    well-known metrics for specific choices of p. For example:

    p = 1: Manhattan Distance

    p = 2: Euclidian distance

    p -> 0: Hamming distance

    p -> infinity: Chebyshev distance

    For a given p, the function will return the distance between the two points at arr_1 and arr_2 in L^p space

    :param arr_1: The location of the first point
    :type arr_1: array-like
    :param arr_2: The location of the second point
    :type arr_2: array-like
    :param p: The parameter specifying which p-norm will be used
    :type p: float
    :return: The distance between arr_1 and arr_2 in L^p space
    """
    return sum(abs(arr_1 - arr_2) ** p) ** (1 / p)



def save_images(strategy_1, strategy_2):
    """
    General function for plotting images of the dataframes created by the main code of the file.

    :param strategy_1: The first strategy to plot and compare. This is the benchmark.
    :param strategy_2: The second strategy to plot and compare. This is the portfolio.
    :return:
    """

    # RETURNS PLOT
    df_1 = strategy_1.df
    df_2 = strategy_2.df
    df_trades = strategy_2.trades
    df_1_ax = plt.subplot2grid((2, 2), (0, 0))
    df_2_ax = plt.subplot2grid((2, 2), (1, 0))
    hist_ax = plt.subplot2grid((2, 2), (0, 1), rowspan=2)

    df_1['returns'].plot(
        ax=df_1_ax,
        figsize=(12, 6),
        title='Daily returns of buy-and-hold portfolio',
        color=BETTERMENT_GRAY
    )
    df_2['returns'].plot(
        ax=df_2_ax,
        title='Daily returns of rebalanced portfolio',
        color=BETTERMENT_BLUE
    )
    sns.distplot(
        a=df_1['returns'],
        color=BETTERMENT_GRAY,
        ax=hist_ax,
        bins=50,
        label='Buy and Hold'
    )
    sns.distplot(
        a=df_2['returns'],
        color=BETTERMENT_BLUE,
        ax=hist_ax,
        bins=50,
        label='Rebalance'
    )
    plt.title('Frequency of returns')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('images', 'daily_returns.png'), dpi=300)
    plt.gcf().clear()

    # ALLOCATIONS PLOT
    fig_alloc, axes_alloc = plt.subplots(nrows=2, ncols=1, sharex='all', sharey='all')

    df_1['allocations'].plot(
        ax=axes_alloc[0],
        figsize=(12, 6),
        title='weight of portfolio assets of buy-and-hold portfolio',
        kind='area',
        legend=False,
        ylim=(0, 1),
        colormap=cm.get_cmap('betterment', 8)
    )
    df_2['allocations'].plot(
        ax=axes_alloc[1],
        title='Weight of portfolio assets of rebalanced portfolio',
        kind='area',
        legend=False,
        colormap=cm.get_cmap('betterment')
    )
    axes_alloc[0].set_xlim(df_1.index[0], df_1.index[-1])
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.3), ncol=8)
    plt.savefig(os.path.join('images', 'asset_allocations.png'), dpi=300)
    plt.gcf().clear()

    # PORTFOLIO VALUE PLOT
    fig_values, axes_values = plt.subplots()

    df_1['values'].sum(axis=1).plot(
        ax=axes_values,
        figsize=(12, 6),
        title='Portfolio values',
        label='Value of buy-and-hold portfolio',
        color=BETTERMENT_GRAY
    )
    df_2['values'].sum(axis=1).plot(
        ax=axes_values,
        label='Value of rebalanced portfolio',
        color=BETTERMENT_BLUE
    )
    axes_values.set_xlim(df_1.index[0], df_1.index[-1])
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.savefig(os.path.join('images', 'values.png'), dpi=300)
    plt.gcf().clear()

    # TRADES PLOT
    fig_trades, axes_trades = plt.subplots()

    df_trades.cumsum().reindex(df_1.index, method='ffill').plot(
        ax=axes_trades,
        figsize=(12, 6),
        title='Cumulative investment per ETF',
        cmap='betterment'
    )
    axes_trades.set_xlim(df_1.index[0], df_1.index[-1])
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.2), ncol=8)
    plt.savefig(os.path.join('images', 'trades.png'), dpi=300)
    plt.gcf().clear()
    plt.close()


def run(max_drift=0.05, minkowski_p=4, starting_cash=100000, commission=0.005):
    """

    :param max_drift: Our max allowed drift from the target portfolio, in percentage points.
    :param minkowski_p: The p-parameter for the Minkowski distance function
    :param starting_cash: The initial cash for the backtest
    :param commission: The commission per trade, expressed in percentage points
    :return: None
    """
    cm.register_cmap('betterment', cmap=colors.ListedColormap(BETTERMENT_PALETTE))
    sns.set(style='whitegrid')
    returns_df = pd.read_csv(
        filepath_or_buffer='portfolio_returns.csv',
        index_col=0,
        parse_dates=True
    )
    tickers = returns_df.columns
    dates = returns_df.index
    returns_df.index.name = 'Date'
    returns_df.columns = pd.MultiIndex.from_product([['daily'], tickers])
    target_weights = pd.Series(
        data=[0.25, 0.25, 0.125, 0.125, 0.04, 0.035, 0.125, 0.05],
        index=tickers
    )

    returns_df[list(zip(['cumulative'] * 8, tickers))] = (returns_df['daily'] + 1).cumprod()

    # BUY AND HOLD PORTFOLIO
    # The buy-and-hold portfolio serves as our baseline. As expected from the name, the buy-and-hold portfolio buys
    # the target ETF portfolio and then holds it for the period.
    buy_and_hold = Strategy('buy_and_hold', dates, tickers, returns_df, target_weights, starting_cash, commission)

    # REBALANCED PORTFOLIO
    # The rebalanced portfolio is our 'active' portfolio for this case study. It rebalances its holdings whenever the
    # allocation drifts too far from the target.
    rebalanced = Strategy('rebalanced', dates, tickers, returns_df, target_weights, starting_cash, commission)
    rebalanced.trade(trigger_function=minkowski_distance, trigger_point=max_drift, p=minkowski_p)

    # SUMMARY STATISTICS
    buy_and_hold_stats = buy_and_hold.summary_stats()
    rebalanced_stats = rebalanced.summary_stats()
    stats = pd.DataFrame(
        data=[buy_and_hold_stats, rebalanced_stats],
        index=['Buy and Hold', 'Rebalanced']
    )
    stats['Capital Gains'] = stats['Capital Gains'].round(2)
    stats.iloc[:, 1:5] = stats.iloc[:, 1:5].round(3)
    stats.index.name = 'Strategy'
    stats.to_csv(os.path.join('datasets', 'stats.csv'))

    save_images(buy_and_hold, rebalanced)
    buy_and_hold.save_to_csv()
    rebalanced.save_to_csv()


if __name__ == '__main__':
    run(max_drift=0.05, minkowski_p=4, starting_cash=100_000, commission=0.005)
