import os
from functools import partial
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# Declaration of constants
STARTING_CASH = 100_000
COMMISSION = 0.005  # This is a form of commission (i.e. fees paid per trade), expressed in percentage.
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


def generate_sensitivity_plot(returns, df, target):
    df = df.copy()
    min_p, max_p, step_p = 1, 10, 1
    min_tol, max_tol, step_tol = 0.02, 0.2, 0.01
    sharpe_df = pd.DataFrame(
        columns=np.round(np.arange(min_tol, max_tol, step_tol), 2),
        index=np.arange(min_p, max_p, step_p)
    )
    for i, p in tqdm(enumerate(np.arange(min_p, max_p, step_p))):
        for j, tol in enumerate(np.arange(min_tol, max_tol, step_tol)):
            rebalanced, _ = generate_rebalanced(returns, df, p, target, tol)
            _, _, sharpe = calculate_summary_statistics(rebalanced)
            sharpe_df.loc[p, tol] = sharpe
    sharpe_df.to_csv(os.path.join('datasets', 'sharpe.csv'))
    sharpe_df.columns.name = 'Threshold'
    sharpe_df.index.name = 'Minowski p'

    mask = np.zeros_like(sharpe_df)
    mask[sharpe_df == sharpe_df.min()] = True

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df, linewidths=0.1, ax=ax, cmap="Blues", xticklabels=2, yticklabels=2, mask=mask)
    plt.tight_layout()
    plt.savefig(os.path.join('images', 'heatmap.png'))
    plt.gcf().clear()
    plt.close()

def generate_rebalanced(returns, df, p, target, tolerance):
    # REBALANCED PORTFOLIO
    # The rebalanced portfolio is our main portfolio of interest. Like the buy-and-hold portfolio, we initialize the
    # rebalanced portfolio using the target ETF portfolio, but unlike the former, we rebalance the portfolio whenever
    # the current allocation strays too far from our target.
    # The definition of 'too far' is given by the Minkowski distance function. See docstring for minkowski_distance
    # for more info.
    df = df.copy()
    dates = df.index
    trades = pd.DataFrame(columns=tickers)
    trades.index.name = 'Date'
    trades.loc[dates[0]] = df.loc[dates[0], 'values'].values
    current_drift = partial(minkowski_distance, arr_2=target.values, p=p)
    for date in tqdm(dates[1:]):  # TQDM is a library for progress bars - provides some nice visual feedback!
        end_of_day_values = df.shift(1).loc[date, 'values'].mul(1 + returns_df.loc[date, 'daily'])
        if current_drift(df.shift(1).loc[date, 'allocations'].values) > tolerance:
            # If we are not within tolerance, we rebalance. Rebalancing is done at the end of the trading day,
            # which is why we still grow the portfolio by the daily returns.
            previous_values = df.loc[date, 'values'].copy()
            position_value = sum(end_of_day_values.values) * target
            trading_cost = abs(end_of_day_values.div(sum(end_of_day_values.values)) - target) * \
                           sum(end_of_day_values.values) * COMMISSION
            df.loc[date, 'values'] = (position_value - trading_cost).values
            df.loc[date:, 'values'] = returns.loc[date:, 'cumulative'].div(
                returns.loc[date, 'cumulative']).mul(df.loc[date, 'values'], axis=1).values
            trade = pd.Series(df.loc[date, 'values'] - previous_values)
            trades.loc[date] = trade
            # Once we have calculated the end-of-day value of the portfolio, we set the allocation by looking at the
            # dollars invested in each ETF
            df.loc[date:, 'allocations'] = df.loc[date:, 'values'].div(df.loc[date:, 'values'].sum(axis=1), axis=0).values
    df['returns'] = df['values'].sum(axis=1).pct_change(1).fillna(0)

    return df, trades


def save_datasets(df_1, df_2, trades_df):
    path = partial(os.path.join, 'datasets')
    df_1['values'].sum(axis=1).to_csv(path('values_buy_and_hold.csv'))
    df_2['values'].sum(axis=1).to_csv(path('values_rebalance.csv'))
    df_1['allocations'].to_csv(path('allocation_buy_and_hold.csv'))
    df_2['allocations'].to_csv(path('allocation_rebalance.csv'))
    df_1['returns'].to_csv(path('returns_buy_and_hold.csv'))
    df_2['returns'].to_csv(path('returns_rebalance.csv'))
    trades_df.to_csv(path('trades.csv'))


def save_images(df_1, df_2, trades_df):
    """
    General function for plotting images of the dataframes created by the main code of the file.

    :param df_1: The first dataframe to plot. This is the benchmark.
    :param df_2: The second dataframe to plot. This is the portfolio.
    :return:
    """

    root = 'images'

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
        title='Frequency of returns',
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
    plt.savefig(os.path.join(root, 'daily_returns.png'))
    plt.gcf().clear()

    # ALLOCATIONS PLOT
    fig_alloc, axes_alloc = plt.subplots(nrows=2, ncols=1, sharex='all', sharey='all')

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
    plt.savefig(os.path.join(root, 'asset_allocations.png'), dpi=300)
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
    plt.savefig(os.path.join(root, 'values.png'))
    plt.gcf().clear()
    plt.close()

    # TRADES PLOT
    fig_trades, axes_trades = plt.subplots()

    trades_df.cumsum().reindex(df_1.index, method='ffill').plot(
        ax=axes_trades,
        figsize=(12, 6),
        title='Cumulative investment per ETF',
        cmap='betterment'
    )
    axes_trades.set_xlim(df_1.index[0], df_1.index[-1])
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.2), ncol=8)
    plt.savefig(os.path.join(root, 'trades.png'))
    plt.gcf().clear()
    plt.close()


def calculate_summary_statistics(df):
    annualized_returns = (df['values'].iloc[-1].sum() / STARTING_CASH) ** (
            60 * 60 * 24 * 365 / ((df.index[-1] - df.index[0]).total_seconds())) - 1
    annualized_volatility = df['returns'].std() * (252 ** 0.5)
    sharpe = annualized_returns / annualized_volatility
    return annualized_returns, annualized_volatility, sharpe

if __name__ == '__main__':
    max_drift = 0.05
    minkowski_p = 6
    cm.register_cmap('betterment', cmap=colors.ListedColormap(betterment_palette))
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
    target_weights = pd.Series(data=[0.25, 0.25, 0.125, 0.125, 0.04, 0.035, 0.125, 0.05], index=tickers)

    returns_df[list(zip(['cumulative'] * 8, tickers))] = (returns_df['daily'] + 1).cumprod()

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
    buy_and_hold_df['returns'] = (buy_and_hold_df['values'].sum(axis=1)).pct_change(1).fillna(0)

    rebalance_df, trades_df = generate_rebalanced(returns_df, buy_and_hold_df, minkowski_p, target_weights, max_drift)

    save_datasets(buy_and_hold_df, rebalance_df, trades_df)
    save_images(buy_and_hold_df, rebalance_df, trades_df)