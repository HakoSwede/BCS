import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial


def minkowski_distance(arr_1, arr_2, p):
    return sum(abs(arr_1 - arr_2)**p)**(1/p)


if __name__ == '__main__':
    sns.set()
    sns.set_palette(sns.color_palette('Blues_r', 8))

    STARTING_CASH = 100000
    ALLOWED_DRIFT = 0.05
    MINKOWSKI_P = 2
    relative_path = 'C:/Users/Tiger/PycharmProjects/BettermentCaseStudy'

    returns_df = pd.read_csv(
        filepath_or_buffer=relative_path + '/portfolio_returns.csv',
        index_col=0,
        parse_dates=True
    )
    tickers = returns_df.columns
    dates = returns_df.index
    initial_weights = pd.DataFrame(
        data=dict(zip(tickers, [0.25, 0.25, 0.125, 0.125, 0.04, 0.035, 0.125, 0.05])),
        index=[dates[0]]
    )
    print(initial_weights)
    minkowski = partial(
        minkowski_distance,
        arr_2=initial_weights.values[0],
        p=MINKOWSKI_P)

    # NON REBALANCED PORTFOLIO
    cum_returns_df = (returns_df+1).cumprod()
    buy_and_hold_df = (cum_returns_df * STARTING_CASH) * initial_weights.values[0]
    buy_and_hold_df_alloc = (buy_and_hold_df.div(buy_and_hold_df.sum(axis=1), axis=0))
    buy_and_hold_df_returns = buy_and_hold_df.sum(axis=1).pct_change(1)

    # REBALANCED PORTFOLIO
    rebalance_df = buy_and_hold_df.copy()
    rebalance_df_alloc = buy_and_hold_df_alloc.copy()

    for date in dates[1:]:
        if minkowski(rebalance_df_alloc.loc[date]) > ALLOWED_DRIFT:
            rebalance_df.loc[date] = sum(rebalance_df.shift(1).loc[date] * (1 + returns_df.loc[date])) * initial_weights.values[0]
        else:
            rebalance_df.loc[date] = rebalance_df.shift(1).loc[date] * (1 + returns_df.loc[date])
        rebalance_df_alloc.loc[date] = (rebalance_df.loc[date].div(rebalance_df.loc[date].sum(), axis=0))

    rebalance_df_returns = rebalance_df.sum(axis=1).pct_change(1)

    # MAKE AND SAVE PLOTS

    # DAILY RETURNS
    buy_and_hold_df_returns.plot(
        figsize=(12, 6),
        title='Daily returns of buy-and-hold portfolio',
        label='Daily returns')
    plt.savefig('buy_and_hold_daily_returns.png')
    plt.gcf().clear()
    plt.close()

    rebalance_df_returns.plot(
        figsize=(12, 6),
        title='Daily returns of rebalanced portfolio',
        label='Daily returns'
    )
    plt.savefig('rebalance_daily_returns.png')
    plt.gcf().clear()
    plt.close()

    # WEIGHT OF ASSETS
    buy_and_hold_df_alloc.plot(
        figsize=(12, 6),
        title='Weight of portfolio assets of buy-and-hold portfolio',
        kind='area',
        ylim=(0, 1)
    )
    plt.savefig('buy_and_hold_weight')
    plt.gcf().clear()
    plt.close()

    rebalance_df_alloc.plot(
        figsize=(12, 6),
        title='Weight of portfolio assets of rebalanced portfolio',
        kind='area',
        ylim=(0, 1)
    )
    plt.savefig('rebalance_weight')
    plt.gcf().clear()
    plt.close()

    # PORTFOLIO RETURNS
    buy_and_hold_df.sum(axis=1).plot(
        figsize=(12, 6),
        title='Value of portfolio',
        label='Value of portfolio'
    )
    plt.savefig('buy_and_hold_value')
    plt.gcf().clear()
    plt.close()

    rebalance_df.sum(axis=1).plot(
        figsize=(12,6),
        title='Value of portfolio',
        label='Value of portfolio'
    )
    plt.savefig('rebalance_value')
    plt.gcf().clear()
    plt.close()
