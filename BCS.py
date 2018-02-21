import os
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from functools import partial

STARTING_CASH = 100_000
MAX_DRIFT = 0.05
MINKOWSKI_P = 5
PATH = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
BETTERMENT_BLUE = '#1F4AB4'
BETTERMENT_GRAY = '#30363D'
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
    :param p: The parameter specifying which p-norm will be used
    :type p: float
    :return: The distance between arr_1 and arr_2 in L^p space
    """
    return sum(abs(arr_1 - arr_2) ** p) ** (1 / p)


def within2_tolerance(target, current, p, drift):
    """
    Checks whether the Minkowski distance between two points is within the specified tolerance

    :param target: The point-location of the ideal asset allocation
    :type target: array-like
    :param current: The point-location of the current asset allocation
    :type current: array-like
    :param p: The parameter specifying which p-norm will be used
    :type p: float
    :param drift: The allowed drift from the ideal asset allocation
    :type drift: float
    :return: Boolean specifying whether the current allocation is within tolerance
    """
    return minkowski_distance(target, current, p) < drift


def generate_portfolios(returns, p, tolerance):
    tickers = returns.columns
    dates = returns.index
    returns.index.name = 'Date'
    returns.columns = pd.MultiIndex.from_product([['daily'], tickers])
    target_weights = pd.Series(data=[0.25, 0.25, 0.125, 0.125, 0.04, 0.035, 0.125, 0.05], index=tickers)

    current_drift = partial(minkowski_distance, arr_2=target_weights.values, p=p)

    returns[list(zip(['cumulative'] * 8, tickers))] = (returns['daily'] + 1).cumprod()

    # BUY AND HOLD PORTFOLIO
    # The buy-and-hold portfolio serves as our baseline. As expected from the name, the buy-and-hold portfolio buys
    # the target ETF portfolio and then holds it for the period.
    buy_and_hold_df = pd.DataFrame(
        data=(returns_df['cumulative'] * STARTING_CASH).mul(target_weights, axis=1).values,
        index=dates,
        columns=pd.MultiIndex.from_product([['values'], tickers])
    )
    buy_and_hold_df[list(zip(['allocations'] * 8, tickers))] = \
        (buy_and_hold_df['values'].div(buy_and_hold_df['values'].sum(axis=1), axis=0))
    buy_and_hold_df['returns'] = (buy_and_hold_df['values'].sum(axis=1)).pct_change(1)

    # REBALANCED PORTFOLIO
    # The rebalanced portfolio is our main portfolio of interest. Like the buy-and-hold portfolio, we initialize the
    # rebalanced portfolio using the target ETF portfolio, but unlike the former, we rebalance the portfolio whenever
    # the current allocation strays too far from our target.
    # The definition of 'too far' is given by the Minkowski distance function. See docstring for minkowski_distance
    # for more info.
    rebalance_df = pd.DataFrame(np.zeros_like(buy_and_hold_df),index=dates,columns=buy_and_hold_df.columns)
    rebalance_df.iloc[0] = buy_and_hold_df.iloc[0]

    for date in tqdm(dates[1:]):  # TQDM is a library for progress bars - provides some nice visual feedback!
        end_of_day_values = rebalance_df.shift(1).loc[date, 'values'].mul(1 + returns_df.loc[date, 'daily']).values
        if current_drift(rebalance_df.shift(1).loc[date, 'allocations'].values) < tolerance:
            # If we are within tolerance, we just set the current value of the portfolio to the portfolio value at the
            # end of the day.
            rebalance_df.loc[date, 'values'] = end_of_day_values
        else:
            # If we are not within tolerance, we rebalance. Rebalancing is done at the end of the trading day,
            # which is why we still grow the portfolio by the daily returns.
            rebalance_df.loc[date, 'values'] = (sum(end_of_day_values) * target_weights).values
        # Once we have calculated the end-of-day value of the portfolio, we set the allocation by looking at the
        # dollars invested in each ETF
        rebalance_df.loc[date, 'allocations'] = \
            (rebalance_df.loc[date, 'values'].div(rebalance_df.loc[date, 'values'].sum())).values

    rebalance_df['returns'] = rebalance_df['values'].sum(axis=1).pct_change(1)

    #save_to_file(buy_and_hold_df, rebalance_df)
    #make_images(buy_and_hold_df, rebalance_df)

    annualized_returns = (rebalance_df['values'].iloc[-1].sum() / STARTING_CASH) ** (31_536_000 / ((dates[-1] - dates[0]).total_seconds())) - 1
    annualized_volatility = rebalance_df['returns'].std() * (252 ** 0.5)
    sharpe = annualized_returns / annualized_volatility

    return sharpe


def save_to_file(df_1, df_2):
    df_1['values'].sum(axis=1).to_csv('values_buy_and_hold.csv')
    df_2['values'].sum(axis=1).to_csv('values_rebalance.csv')
    df_1['allocations'].to_csv('allocation_buy_and_hold.csv')
    df_2['allocations'].to_csv('allocation_rebalance.csv')
    df_1['returns'].to_csv('returns_buy_and_hold.csv')
    df_2['returns'].to_csv('returns_rebalance.csv')


def make_images(df_1, df_2):
    """
    General function for plotting images of the dataframes created by the main code of the file.

    :param df_1: The first dataframe to plot. This is the benchmark.
    :param df_2: The second dataframe to plot. This is the portfolio.
    :return:
    """

    # RETURNS PLOT
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
    df_1['returns'].plot(
        ax=hist_ax,
        kind='hist',
        bins=50,
        alpha=0.5,
        title='Histogram of returns',
        label='Buy and Hold',
        color=BETTERMENT_GRAY
    )
    df_2['returns'].plot(
        ax=hist_ax,
        kind='hist',
        bins=50,
        alpha=0.5,
        label='Rebalance',
        color=BETTERMENT_BLUE
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig('daily_returns.png')
    plt.gcf().clear()

    # ALLOCATIONS PLOT
    fig_alloc, axes_alloc = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)

    df_1['allocations'].plot(
        ax=axes_alloc[0],
        figsize=(12, 6),
        title='Weight of portfolio assets of buy-and-hold portfolio',
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
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.2), ncol=8)
    plt.savefig('asset_allocations.png', dpi=300)
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
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)
    plt.savefig('values.png')
    plt.gcf().clear()
    plt.close()


if __name__ == '__main__':
    sns.set_style('whitegrid')
    cm.register_cmap('betterment', cmap=colors.ListedColormap(betterment_palette))
    returns_df = pd.read_csv(
        filepath_or_buffer=os.path.join(PATH, 'portfolio_returns.csv'),
        index_col=0,
        parse_dates=True
    )

    generate_portfolios(returns_df, 5, 0.05)
