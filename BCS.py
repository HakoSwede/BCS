import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial


def minkowski_distance(arr_1, arr_2, p):
    return sum(abs(arr_1 - arr_2)**p)**(1/p)


def within_tolerance(target, current, drift):
    return minkowski_distance(target, current, MINKOWSKI_P) < drift


if __name__ == '__main__':
    sns.set_style('whitegrid')

    STARTING_CASH = 100000
    MAX_DRIFT = 0.05
    MINKOWSKI_P = 1
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
    cumulative_returns_df = (returns_df+1).cumprod()
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
            rebalance_df.loc[date] = sum(rebalance_df.shift(1).loc[date].mul(1 + returns_df.loc[date])) * target_weights
        rebalance_df_alloc.loc[date] = (rebalance_df.loc[date].div(rebalance_df.loc[date].sum()))

    rebalance_df_returns = rebalance_df.sum(axis=1).pct_change(1)

    # MAKE AND SAVE PLOTS
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
        colormap='Blues_r'
    )
    rebalance_df_alloc.plot(
        ax=axes_alloc[1],
        title='Weight of portfolio assets of rebalanced portfolio',
        kind='area',
        legend=False,
        colormap='Blues_r'
    )
    axes_alloc[0].set_xlim(dates[0], dates[-1])
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.2), ncol=8)
    plt.savefig('asset_allocations.png')
    plt.gcf().clear()
    plt.close()

    # PORTFOLIO RETURNS

    fig_values, axes_values = plt.subplots()

    buy_and_hold_df.sum(axis=1).plot(
        ax=axes_values,
        figsize=(12, 6),
        title='Portfolio values',
        label='Value of buy-and-hold portfolio'
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
